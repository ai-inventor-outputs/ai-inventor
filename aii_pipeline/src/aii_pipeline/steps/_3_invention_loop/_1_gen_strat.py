"""GEN_STRAT Step - Generate research strategies from multiple LLMs.

Each strategy contains:
- title, objective, rationale: What we're doing and why
- artifact_directions: Artifacts to create THIS iteration (IDs assigned by code after LLM output)
- expected_outcome: What we'll have after this iteration
Each strategy's artifact directions are elaborated into detailed plans.

Supports two backends:
- OpenRouter (default): Uses chat() with structured output
- Claude agent: Uses Agent with SDK native output_format for structured output

Verification + Retry (similar to verify_citations in audit_hypo):
- Verifies: strategy count, valid dependencies
- Retries with conversation continuation if verification fails

Uses aii_lib for:
- OpenRouterClient: LLM calls (OpenRouter backend)
- Agent/AgentOptions: Claude agent calls
"""

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import ParallelTModule

from aii_lib import get_model_short
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
    ArtifactDep,
    ArtifactDirection,
    Strategy,
)
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.s_prompt import (
    get as get_gen_strat_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.u_prompt import (
    get as get_gen_strat_prompt,
)
from aii_pipeline.steps.base import ModuleCtx

from .invention_loop import LoopCtx
from .utils.gen_strat_tasks import gen_strat, gen_strat_claude_agent

# =============================================================================
# MAIN STEP
# =============================================================================


@dataclass
class GenStratCtx(ModuleCtx):
    """Substep ctx for gen_strat."""

    parent_ctx: LoopCtx | None = None
    iteration: int = 1
    previous_strategies: list[dict] | None = None
    reviewer_feedback_text: str | None = None
    paper_text: str | None = None
    parent_id: str = ""


class GenStratModule(ParallelTModule):
    """gen_strat substep — multi-LLM strategy generation per iteration."""

    kind: Literal["gen_strat_module"] = "gen_strat_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_strat"] = "gen_strat"

    def get_context(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        previous_strategies: list[dict] | None = None,
        reviewer_feedback_text: str | None = None,
        paper_text: str | None = None,
        parent_id: str,
    ) -> GenStratCtx:
        return GenStratCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            previous_strategies=previous_strategies,
            reviewer_feedback_text=reviewer_feedback_text,
            paper_text=paper_text,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        previous_strategies: list[dict] | None = None,
        reviewer_feedback_text: str | None = None,
        paper_text: str | None = None,
        parent_id: str,
    ) -> list[Strategy]:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                output_dir=output_dir,
                previous_strategies=previous_strategies,
                reviewer_feedback_text=reviewer_feedback_text,
                paper_text=paper_text,
                parent_id=parent_id,
            )
        ):
            """Run the GEN_STRAT step.

            Generates strategies from multiple LLMs. Each strategy contains
            artifact directions that will be elaborated into plans.
            """
            config = ctx.config
            hypothesis = ctx.hypothesis
            artifacts = ctx.invention_loop_group.get_artifacts()
            user_uploads_path = ctx.user_uploads_path

            mid = emit.start_single_module(
                name="gen_strat",
                parent_id=parent_id,
            )

            # Get config
            gen_strat_cfg = config.invention_loop.gen_strat
            max_iterations = config.invention_loop.max_iterations
            allowed_artifacts = config.invention_loop.allowed_artifacts
            strats_per_call = gen_strat_cfg.strats_per_call  # Strategies generated per LLM call
            calls_per_llm = gen_strat_cfg.calls_per_llm  # Parallel calls per model
            max_parallel = gen_strat_cfg.claude_agent.max_concurrent_agents
            use_claude_agent = gen_strat_cfg.use_claude_agent

            # Verification config
            verify_cfg = gen_strat_cfg.verify_artifacts
            verify_retries = verify_cfg.retry
            min_valid_artifacts = verify_cfg.min_valid_artifacts
            art_limit = gen_strat_cfg.art_limit
            artifact_context_per_type = gen_strat_cfg.artifact_context_per_type

            # =========================================================================
            # SETUP BACKEND
            # =========================================================================
            # Step subdir within iteration dir (always created regardless of backend)
            if output_dir:
                step_dir = (output_dir / "gen_strat").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            if use_claude_agent:
                claude_cfg = gen_strat_cfg.claude_agent

                models = [
                    {
                        "model": claude_cfg.model,
                        "model_short": get_model_short(claude_cfg.model),
                    }
                ]
                llm_provider = "claude_agent"
                llm_timeout = claude_cfg.seq_prompt_timeout
            else:
                llm_cfg = gen_strat_cfg.llm_client
                llm_timeout = llm_cfg.llm_timeout
                llm_provider = "openrouter"

                # Parse models from config
                models = [
                    {
                        "model": m.model,
                        "reasoning_effort": m.reasoning_effort,
                        "suffix": m.suffix,
                    }
                    for m in llm_cfg.models
                ]

            openrouter_key = config.api_keys.openrouter

            num_models = len(models)
            total_calls = calls_per_llm * num_models
            total_strategies = strats_per_call * total_calls

            # Build {id} set and {id: type} map for verification of strategies' deps.
            existing_artifact_ids = {a.id for a in artifacts}
            artifact_pool_map = {a.id: a.type for a in artifacts}

            model_names = [m["model"] for m in models]
            emit.status_private_info(f"Provider: {llm_provider}")
            emit.status_private_info(f"Models: {model_names}")
            emit.status_private_info(
                f"Strategies: {total_strategies} ({strats_per_call}/call x {calls_per_llm} calls x {num_models} models)"
            )
            # ``Iteration: N of M`` was a duplicate of ``run.py:124``'s
            # ``ITERATION N/M``; ``Paper texts available: K`` was noise (the
            # FE doesn't distinguish artifact-pool size from iteration
            # bookkeeping). Both removed per design feedback.
            emit.status_public_info(
                f"Allowed artifacts: {allowed_artifacts if allowed_artifacts else 'all'}"
            )
            emit.status_private_info(
                f"Verify: retries={verify_retries}, min_valid={min_valid_artifacts}"
            )
            emit.status_private_info(f"Art limit: {art_limit if art_limit else 'none'}")
            emit.status_private_info(f"Timeout: {f'{llm_timeout}s' if llm_timeout else 'None'}")

            # Build system prompt
            system_prompt = get_gen_strat_system_prompt(allowed_artifacts)

            # Build task configs - calls_per_llm tasks per model. Task name
            # is the literal "gen_strat" for all parallel siblings; replay-mode
            # slot-claim distinguishes them by sequence position in
            # ``parent.children`` (asyncio.gather argument-order is preserved
            # for the synchronous prefix of each coroutine, so children land
            # in the same order the for-loop builds task_configs). See
            # REPLAY_EXECUTE_AUDIT_0_2.md for the determinism analysis. We
            # intentionally do NOT use the f"gen_strat_{model_short}_{idx}"
            # discriminator pattern that gen_plan uses, because doing so
            # would break backward compatibility with already-recorded
            # clone logs whose task_starts emit name="gen_strat".
            task_configs = []

            for model_cfg in models:
                for _call_idx in range(calls_per_llm):
                    task_id = "gen_strat"

                    prompt = get_gen_strat_prompt(
                        hypothesis=hypothesis,
                        artifacts=artifacts,
                        current_iteration=iteration,
                        max_iterations=max_iterations,
                        previous_strategies=previous_strategies,
                        allowed_artifacts=allowed_artifacts,
                        num_strategies=strats_per_call,
                        art_limit=art_limit,
                        artifact_context_per_type=artifact_context_per_type,
                        reviewer_feedback_text=reviewer_feedback_text,
                        paper_text=paper_text,
                        user_folder_path=user_uploads_path,
                    )

                    task_configs.append((task_id, prompt, model_cfg))

            # Run all generators in parallel (with optional semaphore for Claude agent).
            # The pre-fanout "Running N strategy generators..." emit was
            # removed — the FE already shows the parallel task count via
            # the module's task list, so the line was redundant chrome.
            sem = asyncio.Semaphore(max_parallel) if max_parallel else None

            async def run_task(task_name: str, prompt: str, model_cfg: dict):
                async with sem if sem else nullcontext():
                    if use_claude_agent:
                        # Per-task CWD so parallel agents don't collide
                        task_cwd = (
                            (output_dir / task_name)
                            if output_dir
                            else Path.cwd().resolve() / task_name
                        )
                        task_cwd.mkdir(parents=True, exist_ok=True)
                        return task_name, await gen_strat_claude_agent(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            agent_cfg=claude_cfg,
                            cwd=task_cwd,
                            output_dir=output_dir,
                            iteration=iteration,
                            existing_artifact_ids=existing_artifact_ids,
                            artifact_pool_map=artifact_pool_map,
                            num_strategies=strats_per_call,
                            verify_retries=verify_retries,
                            min_valid_artifacts=min_valid_artifacts,
                            allowed_artifacts=allowed_artifacts,
                            art_limit=art_limit,
                            task_name=task_name,
                            parent_module_id=mid,
                        )
                    return task_name, await gen_strat(
                        task_name=task_name,
                        parent_module_id=mid,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model_cfg["model"],
                        api_key=openrouter_key,
                        iteration=iteration,
                        existing_artifact_ids=existing_artifact_ids,
                        artifact_pool_map=artifact_pool_map,
                        num_strategies=strats_per_call,
                        reasoning_effort=model_cfg.get("reasoning_effort", "medium"),
                        suffix=model_cfg.get("suffix"),
                        llm_timeout=llm_timeout,
                        verify_retries=verify_retries,
                        min_valid_artifacts=min_valid_artifacts,
                        allowed_artifacts=allowed_artifacts,
                        art_limit=art_limit,
                    )

            # v26: no skip-mask — send_user_msg replays whole modules from
            # scratch, no sibling preloading.
            dispatched = await asyncio.gather(
                *[
                    run_task(task_id, prompt, model_cfg)
                    for task_id, prompt, model_cfg in task_configs
                ],
                return_exceptions=True,
            )

            # Collect all strategy dicts with their source task_id.
            all_strategy_items: list[tuple[str, dict]] = []
            for tr in dispatched:
                if isinstance(tr, BaseException):
                    emit.status_public_warning(f"GenStrat task failed: {tr}")
                    continue
                task_id, strategy_dicts = tr
                if strategy_dicts is None:
                    continue
                for s_dict in strategy_dicts:
                    all_strategy_items.append((task_id, s_dict))

            # Convert to Strategy objects — use task_id + idx as the id
            all_strategies: list[Strategy] = []
            task_idx_counters: dict[str, int] = {}

            for task_id, s_dict in all_strategy_items:
                try:
                    task_idx_counters[task_id] = task_idx_counters.get(task_id, 0) + 1
                    idx = task_idx_counters[task_id]
                    strategy_id = f"{task_id}_idx{idx}"

                    # Convert artifact directions
                    directions = []
                    for a_dict in s_dict.get("artifact_directions", []):
                        deps_raw = a_dict.get("depends_on", []) or []
                        deps = [
                            ArtifactDep(id=d.get("id", ""), label=d.get("label", ""))
                            if isinstance(d, dict)
                            else ArtifactDep(id=str(d), label="")
                            for d in deps_raw
                        ]
                        direction = ArtifactDirection(
                            id=a_dict.get("id", ""),
                            type=a_dict.get("type", "research"),
                            objective=a_dict.get("objective", ""),
                            approach=a_dict.get("approach", ""),
                            depends_on=deps,
                        )
                        directions.append(direction)

                    strategy = Strategy(
                        id=strategy_id,
                        title=s_dict.get("title", ""),
                        summary=s_dict.get("summary", ""),
                        objective=s_dict.get("objective", ""),
                        rationale=s_dict.get("rationale", ""),
                        artifact_directions=directions,
                        expected_outcome=s_dict.get("expected_outcome", ""),
                    )

                    all_strategies.append(strategy)

                except Exception as e:
                    emit.status_public_warning(f"Failed to create strategy: {e}")
                    raise

            emit.status_public_success(
                f"GEN_STRAT complete: {len(all_strategies)} strategies generated"
            )

            from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
                GenStratOut,
            )

            emit.module_output(
                module_id=mid,
                name="gen_strat",
                output=GenStratOut(strategies=all_strategies),
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return all_strategies
