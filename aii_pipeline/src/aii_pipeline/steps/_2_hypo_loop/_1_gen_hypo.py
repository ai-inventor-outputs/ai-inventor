#!/usr/bin/env python3
"""
Hypothesis Generation Module - Multi-Model Parallel LLM.

Generates research hypotheses using:
- Multiple models (hypos_per_llm × num_models = total hypotheses)
- asyncio.Semaphore for concurrent execution with max_parallel limit
- Structured JSON output (Hypothesis schema)
- Seed prompts from seed_hypo module

Supports two backends:
- OpenRouter (default): Uses chat() with tool loop
- Claude agent: Uses Agent with SDK native output_format for structured output
"""

import asyncio
import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from aii_lib.abilities.aii_ability import abilities_to_openai_tools
from aii_lib.abilities.endpoint_names import (
    AII_WEB_FETCH,
    AII_WEB_FETCH_GREP,
    AII_WEB_SEARCH,
)
from aii_lib.agent_backend import Agent
from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import ParallelTModule

from aii_lib import (
    OpenRouterClient,
    chat,
    get_model_short,
)
from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.out_schema import (
    GenHypoOut,
    Hypothesis,
)
from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.s_prompt import (
    get as get_gen_hypo_sysprompt,
)
from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.u_prompt import (
    get as get_gen_hypo_prompt,
)
from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.u_prompt import (
    get_force_output_prompt,
)
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig


async def _run_task_openrouter(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    timeout: int,
    reasoning_effort: str | None,
    tools: list[dict] | None,
    local_tool_handlers: dict[str, Callable] | None = None,
) -> dict | None:
    """Run a single hypothesis generation task with OpenRouter."""
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        effective_model = OpenRouterClient.resolve_model(model)

        async with OpenRouterClient(
            api_key=api_key,
            model=effective_model,
            timeout=timeout,
        ) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                tools=tools,
                response_format=Hypothesis,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                task_id=task_id,
                task_name=task_name,
                emit_summary=True,
                local_tool_handlers=local_tool_handlers,
            )

            output_json = client.extract_json_from_response(result.response)
            if output_json:
                # Match ``_run_task_claude_agent``: emit ``task_output``
                # so replay synthesis can recover the structured payload
                # on a future fork.
                parsed = json.loads(output_json)
                emit.task_output(task_id=task_id, output=parsed)
                emit.end_task(task_id, status="done", name=task_name)
                return parsed

            emit.end_task(task_id, status="done", name=task_name, text="No output")
            return None

    except Exception as e:
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


async def _run_task_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    task_name: str | None = None,
    parent_module_id: str | None = None,
) -> dict | None:
    """Run a single hypothesis generation task with Claude agent."""
    from aii_lib import build_options

    task_id: str | None = None
    if task_name and parent_module_id:
        task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        options = build_options(
            agent_cfg,
            cwd,
            task_id=task_id or "",
            task_name=task_name or "",
            system_prompt=system_prompt,
            output_format=Hypothesis.to_struct_output(),
            force_output_prompt=get_force_output_prompt(),
        )

        response = await Agent(options).run(prompt)

        if response.structured_output:
            if task_id and task_name:
                # Mirror ``response.structured_output`` to ``task.output`` so
                # replay-execute synthesis (Stage 4 — see ``aii_lib.agent_backend
                # .claude_agent_sdk.replay.synthesize_agent_response_from_task``)
                # can recover it on a subsequent fork. Without this, the
                # task's ``output`` stays ``None`` (only the aggregated
                # ``module_output`` carries the hypotheses list), and
                # forking from a parent that hasn't passed gen_hypo's
                # boundary fails with "gen_hypo emitted 0 hypotheses".
                # Done per-substep instead of universally at Agent.run
                # because some substeps (gen_viz / gen_art) emit a typed
                # post-processed result on ``task.output`` and would be
                # preempted by an agent-layer auto-emit.
                #
                # Coerce the agent's raw dict through ``Hypothesis(**dict)``
                # so the discriminator default (``kind="hypothesis"``)
                # populates before it lands on ``task.output: AnyOutput``
                # — pydantic's tagged-union dispatch runs BEFORE field
                # defaults, so a bare dict without ``kind`` would fail
                # ``union_tag_not_found``. Same machinery the wrapper
                # ``GenHypoOut(...)`` constructor below relies on.
                parsed_output = Hypothesis.model_validate(response.structured_output)
                emit.task_output(
                    task_id=task_id,
                    output=parsed_output,
                )
                emit.end_task(task_id, status="done", name=task_name)
            return response.structured_output

        if task_id and task_name:
            emit.end_task(task_id, status="done", name=task_name, text="No output")
        return None

    except Exception as e:
        if task_id and task_name:
            emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


@dataclass
class GenHypoCtx(ModuleCtx):
    """Substep ctx for gen_hypo."""

    agent_prompts: list | None = None
    run_dir: Path | None = None
    previous_hypothesis: dict | None = None
    previous_review_feedback: dict | None = None
    iteration: int = 1
    user_uploads_path: str = ""
    parent_id: str = ""


class GenHypoModule(ParallelTModule):
    """gen_hypo substep — multi-LLM parallel hypothesis generation.

    Pre-instantiated by the scaffold under each ``hypo_loop`` iter.
    The phase orchestrator (``HypoLoopGroup.execute``) looks up this typed
    instance from ``iter.children`` and calls
    ``await module.execute(...)``.
    """

    kind: Literal["gen_hypo_module"] = "gen_hypo_module"
    """Per-subclass discriminator. The pipeline's ``AnyModule`` union
    (rebound in ``aii_pipeline/run/__init__.py``) routes to this
    typed class on ``model_validate``; without a unique tag, fork-init
    seed hydration would collapse to the base ``ParallelTModule`` and
    lose ``execute()``."""

    name: Literal["gen_hypo"] = "gen_hypo"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        agent_prompts=None,
        run_dir=None,
        previous_hypothesis: dict | None = None,
        previous_review_feedback: dict | None = None,
        iteration: int = 1,
        user_uploads_path: str = "",
        parent_id: str,
    ) -> GenHypoCtx:
        return GenHypoCtx(
            config=config,
            output_dir=run_dir,
            agent_prompts=agent_prompts,
            run_dir=run_dir,
            previous_hypothesis=previous_hypothesis,
            previous_review_feedback=previous_review_feedback,
            iteration=iteration,
            user_uploads_path=user_uploads_path,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        agent_prompts=None,
        run_dir=None,
        previous_hypothesis: dict | None = None,
        previous_review_feedback: dict | None = None,
        iteration: int = 1,
        user_uploads_path: str = "",
        parent_id: str,
    ):
        with ctx_scope(
            self.get_context(
                config=config,
                agent_prompts=agent_prompts,
                run_dir=run_dir,
                previous_hypothesis=previous_hypothesis,
                previous_review_feedback=previous_review_feedback,
                iteration=iteration,
                user_uploads_path=user_uploads_path,
                parent_id=parent_id,
            )
        ):
            """Run hypothesis generation with parallel LLM calls.

            Uses asyncio.Semaphore for managed concurrent execution.
            Supports OpenRouter (default) or Claude agent backend.
            """
            # Create output directory (ensure absolute path)
            # When called from hypo_loop, run_dir is already the gen_hypo subdir.
            # When called standalone, run_dir is None and we create our own.
            if run_dir:
                output_dir = Path(run_dir).resolve()
            else:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                output_dir = Path(f"{config.outputs_directory}/{timestamp}_gen_hypo").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            mid = emit.start_single_module(
                name="gen_hypo",
                parent_id=parent_id,
            )

            # Get config
            aii_prompt = config.prompt
            seeded_per_llm = config.gen_hypo.seeded_hypos_per_llm
            unseeded_per_llm = config.gen_hypo.unseeded_hypos_per_llm
            use_claude_agent = config.gen_hypo.use_claude_agent
            tool_names = [AII_WEB_SEARCH, AII_WEB_FETCH, AII_WEB_FETCH_GREP]

            # =====================================================================
            # SETUP BACKEND
            # =====================================================================
            if use_claude_agent:
                claude_cfg = config.gen_hypo.claude_agent
                max_parallel = claude_cfg.max_concurrent_agents

                # Step subdir with claude_agent/ as CWD
                agent_cwd = (output_dir / "claude_agent").resolve()
                agent_cwd.mkdir(parents=True, exist_ok=True)

                models = [
                    {
                        "model": claude_cfg.model,
                        "model_short": get_model_short(claude_cfg.model),
                    }
                ]
                llm_provider = "claude_agent"
            else:
                llm_client = config.gen_hypo.llm_client
                max_parallel = config.gen_hypo.max_parallel
                llm_provider = "openrouter"

                # Convert tool names to OpenRouter format
                or_tools = abilities_to_openai_tools(tool_names)

                # Add user folder tools if configured
                uf_handlers = None
                if user_uploads_path and Path(user_uploads_path).exists():
                    from aii_pipeline.llm_tools import make_user_folder_tools

                    uf_tool_defs, uf_handlers = make_user_folder_tools(user_uploads_path)
                    or_tools.extend(uf_tool_defs)

                # Build models list
                models = []
                for m in llm_client.models:
                    models.append(
                        {
                            "model": m.model,
                            "model_short": get_model_short(m.model),
                            "suffix": m.suffix,
                            "reasoning_effort": m.reasoning_effort,
                        }
                    )

                if not models:
                    emit.status_public_error("No models configured in gen_hypo.llm_client.models")
                    return None

            num_models = len(models)
            total_seeded = seeded_per_llm * num_models
            total_unseeded = unseeded_per_llm * num_models
            total_hypotheses = total_seeded + total_unseeded

            agent_prompts = agent_prompts or [[] for _ in range(total_seeded)]
            while len(agent_prompts) < total_seeded:
                agent_prompts.append([])

            if use_claude_agent:
                tools_str = f"WebSearch, WebFetch, {AII_WEB_FETCH_GREP}"
            else:
                tools_str = ", ".join(tool_names)
            emit.status_public_progress(
                f"GenHypo - Generating {total_hypotheses} hypotheses ({seeded_per_llm} seeded + {unseeded_per_llm} unseeded per LLM × {num_models} models)"
            )
            emit.status_private_info(
                f"Provider: {llm_provider} | Tools: {tools_str} | Max Parallel: {max_parallel or 'unlimited'}"
            )

            system_prompt_seeded = get_gen_hypo_sysprompt(seeded=True)
            system_prompt_unseeded = get_gen_hypo_sysprompt(seeded=False)
            if max_parallel is not None and max_parallel <= 0:
                max_parallel = 1
            sem = asyncio.Semaphore(max_parallel) if max_parallel else None

            # Build task configs. Each task gets a 1-based ``gen_hypo_<n>``
            # identifier — the FE's ``NODETREE_DISPLAY_NAMES`` keys off
            # this suffix to render "Idea 1", "Idea 2", … under the
            # ``Create Idea`` module instead of all tasks collapsing to
            # the same parent label. The counter spans seeded + unseeded
            # so every task in this iteration has a unique slot.
            task_configs = []
            seeded_idx = 0
            task_counter = 0

            for model_config in models:
                model_name = model_config["model"]

                for _hypo_idx in range(seeded_per_llm):
                    task_counter += 1
                    task_id = f"gen_hypo_{task_counter}"
                    seeded_idx += 1
                    agent_inspiration = (
                        agent_prompts[seeded_idx - 1]
                        if (seeded_idx - 1) < len(agent_prompts)
                        else []
                    )
                    seeds = [
                        {"id": p.get("id", "?"), "prompt": p.get("prompt", "")}
                        for p in agent_inspiration
                        if isinstance(p, dict)
                    ]

                    task_configs.append(
                        {
                            "task_id": task_id,
                            "prompt": get_gen_hypo_prompt(
                                agent_inspiration,
                                aii_prompt,
                                web_search=True,
                                previous_hypothesis=previous_hypothesis,
                                previous_review_feedback=previous_review_feedback,
                                user_folder_path=user_uploads_path,
                            ),
                            "model": model_name,
                            "model_config": model_config,
                            "is_seeded": True,
                            "seeds": seeds,
                        }
                    )

            for model_config in models:
                model_name = model_config["model"]

                for _hypo_idx in range(unseeded_per_llm):
                    task_counter += 1
                    task_id = f"gen_hypo_{task_counter}"

                    task_configs.append(
                        {
                            "task_id": task_id,
                            "prompt": get_gen_hypo_prompt(
                                [],
                                aii_prompt,
                                web_search=True,
                                previous_hypothesis=previous_hypothesis,
                                previous_review_feedback=previous_review_feedback,
                                user_folder_path=user_uploads_path,
                            ),
                            "model": model_name,
                            "model_config": model_config,
                            "is_seeded": False,
                            "seeds": [],
                        }
                    )

            # =====================================================================
            # RUN TASKS
            # =====================================================================
            async def run_task(task_cfg: dict):
                async with sem if sem else nullcontext():
                    task_id = task_cfg["task_id"]
                    sys_prompt = (
                        system_prompt_seeded if task_cfg["is_seeded"] else system_prompt_unseeded
                    )

                    if use_claude_agent:
                        result = await _run_task_claude_agent(
                            prompt=task_cfg["prompt"],
                            system_prompt=sys_prompt,
                            agent_cfg=claude_cfg,
                            cwd=agent_cwd,
                            task_name=task_id,
                            parent_module_id=mid,
                        )
                    else:
                        model_config = task_cfg["model_config"]
                        result = await _run_task_openrouter(
                            task_name=task_id,
                            parent_module_id=mid,
                            prompt=task_cfg["prompt"],
                            system_prompt=sys_prompt,
                            model=model_config["model"],
                            api_key=config.api_keys.openrouter,
                            timeout=llm_client.llm_timeout,
                            reasoning_effort=model_config.get("reasoning_effort"),
                            tools=or_tools,
                            local_tool_handlers=uf_handlers,
                        )

                    return {
                        "task_id": task_id,
                        "model": task_cfg["model"],
                        "is_seeded": task_cfg["is_seeded"],
                        "seeds": task_cfg["seeds"],
                        "result": result,
                    }

            # v26: no skip-mask — send_user_msg replays whole modules from
            # scratch, no sibling preloading.
            dispatched = await asyncio.gather(
                *[run_task(cfg) for cfg in task_configs],
                return_exceptions=True,
            )

            # Collect results.
            hypotheses: list[dict] = []
            for tr in dispatched:
                if isinstance(tr, BaseException):
                    emit.status_public_warning(f"GenHypo task failed: {tr}")
                    continue
                if tr["result"]:
                    hypotheses.append(
                        {
                            "hypothesis_id": tr["task_id"],
                            "model": tr["model"],
                            "is_seeded": tr["is_seeded"],
                            "seeds": tr["seeds"],
                            **tr["result"],
                        }
                    )

            emit.status_public_success(
                f"GenHypo completed - {len(hypotheses)}/{total_hypotheses} hypotheses generated"
            )

            # Build return value (for in-memory pipeline flow)
            module_output = GenHypoOut(
                output_dir=str(output_dir),
                hypotheses=hypotheses,
            )

            # Emit the typed module output. ``GenHypoOut.hypotheses``
            # carries the per-task hypothesis dict list — readers walk
            # ``module.output.hypotheses`` instead of the legacy
            # ``outputs=[...]`` plural payload.
            emit.module_output(
                module_id=mid,
                name="gen_hypo",
                output=module_output,
            )
            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return module_output
