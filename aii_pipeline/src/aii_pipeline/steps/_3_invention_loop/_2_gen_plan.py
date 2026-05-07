"""GEN_PLAN Step - Elaborate artifact_directions into detailed plans.

Takes artifact_directions from ALL strategies and elaborates each into
detailed plans. For each (artifact_direction, llm) combination, we generate
`plans_per_strat` plans.

Each artifact type has its own plan schema:
- proof: informal_proof_draft, rationale
- research: research_plan, rationale
- dataset: ideal_dataset_criteria, dataset_search_plan
- experiment: implementation_pseudocode, fallback_plan, dependencies, testing_plan
- evaluation: metrics_descriptions, metrics_justification

Total plans = total_artifact_directions_across_strats × num_models × plans_per_strat

Supports two backends:
- OpenRouter (default): Uses chat() with structured output
- Claude agent: Uses Agent with SDK native output_format for structured output

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
    ArtifactDirection,
    Strategy,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
    BasePlan,
    PlanType,
    get_plan_schema,
    verify_compute_profile,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.s_prompt import (
    get as get_plan_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.u_prompt import (
    get as get_planner_prompt,
)
from aii_pipeline.steps.base import ModuleCtx

from .invention_loop import LoopCtx
from .utils.gen_plan_helpers import (
    _create_testing_strategy,
    gen_plan_for_art,
    gen_plan_for_art_claude_agent,
)


@dataclass
class GenPlanCtx(ModuleCtx):
    """Substep ctx for gen_plan."""

    parent_ctx: LoopCtx | None = None
    iteration: int = 1
    strategies: list[Strategy] | None = None
    parent_id: str = ""


class GenPlanModule(ParallelTModule):
    """gen_plan substep — elaborate strategies' artifact_directions into plans."""

    kind: Literal["gen_plan_module"] = "gen_plan_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_plan"] = "gen_plan"

    def get_context(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        strategies: list[Strategy] | None = None,
        parent_id: str,
    ) -> GenPlanCtx:
        return GenPlanCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            strategies=list(strategies) if strategies else None,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        strategies: list[Strategy] | None = None,
        parent_id: str,
    ) -> list[BasePlan]:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                output_dir=output_dir,
                strategies=strategies,
                parent_id=parent_id,
            )
        ):
            """Run the GEN_PLAN step.

            Takes artifact_directions from ALL strategies and elaborates each into
            detailed plans. All strategies' directions are processed in parallel.
            """
            config = ctx.config
            hypothesis = ctx.hypothesis
            artifacts = ctx.invention_loop_group.get_artifacts()
            user_uploads_path = ctx.user_uploads_path

            mid = emit.start_parallel_module(
                name="gen_plan",
                parent_id=parent_id,
            )

            emit.status_public_info(
                f"GEN_PLAN - Elaborating artifact directions for iteration {iteration}"
            )

            # Check for testing mode
            testing_mode = config.invention_loop.test_all_artifacts
            allowed_artifacts = config.invention_loop.allowed_artifacts

            if testing_mode:
                emit.status_public_info("TESTING MODE: Creating synthetic strategy")
                test_strat = _create_testing_strategy(artifacts, allowed_artifacts, iteration)
                strategies = [test_strat]
                emit.status_private_info(
                    f"Created test strategy with {len(test_strat.artifact_directions)} artifact directions",
                )

            if not strategies:
                emit.status_public_error("No strategies provided - cannot generate plans")
                return []

            # Collect artifact_directions from ALL strategies, tracking which strategy each came from
            artifact_directions = []
            direction_to_strategy: dict[str, str] = {}  # direction_id -> strategy_id
            for strat in strategies:
                for direction in strat.artifact_directions:
                    direction_to_strategy[direction.id] = strat.id
                artifact_directions.extend(strat.artifact_directions)

            if not artifact_directions:
                emit.status_public_warning(
                    "Strategies have no artifact_directions - skipping gen_plan"
                )
                return []

            gen_plan_cfg = config.invention_loop.gen_plan
            use_claude_agent = gen_plan_cfg.use_claude_agent
            plans_per_strat = gen_plan_cfg.plans_per_strat
            openrouter_key = config.api_keys.openrouter

            # Setup backend
            # Step subdir within iteration dir (always created regardless of backend)
            if output_dir:
                step_dir = (output_dir / "gen_plan").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            if use_claude_agent:
                claude_cfg = gen_plan_cfg.claude_agent
                max_parallel = claude_cfg.max_concurrent_agents

                models = [
                    {
                        "model": claude_cfg.model,
                        "model_short": get_model_short(claude_cfg.model),
                    }
                ]
                llm_provider = "claude_agent"
                llm_timeout = claude_cfg.seq_prompt_timeout
            else:
                llm_cfg = gen_plan_cfg.llm_client
                llm_timeout = llm_cfg.llm_timeout
                max_parallel = None
                llm_provider = "openrouter"
                models = [
                    {
                        "model": m.model,
                        "reasoning_effort": m.reasoning_effort,
                        "suffix": m.suffix,
                    }
                    for m in llm_cfg.models
                ]

            num_models = len(models)
            num_directions = len(artifact_directions)
            total_tasks = num_directions * num_models * plans_per_strat

            emit.status_private_info(f"Provider: {llm_provider}")
            emit.status_private_info(
                f"Strategies: {len(strategies)} ({', '.join(s.id for s in strategies)})"
            )
            emit.status_private_info(f"Artifact directions (combined): {num_directions}")
            emit.status_private_info(f"Models: {[m['model'] for m in models]}")
            emit.status_private_info(f"Plans per strat: {plans_per_strat}")
            emit.status_private_info(f"Total tasks: {total_tasks}")
            emit.status_private_info(f"Timeout: {f'{llm_timeout}s' if llm_timeout else 'None'}")

            # Compute profile info (only relevant in runpod mode, but always included in prompt)
            exec_cfg = config.execute_env
            runpod_cfg = exec_cfg.runpod
            compute_profiles_dict: dict | None = None
            artifact_type_profiles: dict[str, list[str]] | None = None
            if exec_cfg.mode == "runpod" and runpod_cfg.compute_profiles:
                compute_profiles_dict = runpod_cfg.compute_profiles
                artifact_type_profiles = runpod_cfg.artifact_type_profiles

            # Build task configs: one task per (artifact_direction, model, plan_num) combination
            task_configs = []
            task_counter = 0
            # Per-type 1-based index, scoped to this gen_plan invocation. Used in
            # the wire-level task name so the FE tree can render "Plan Dataset 1",
            # "Plan Experiment 2", etc. without a separate type lookup. Same
            # pattern as gen_art / gen_art_demo.
            type_idx_counters: dict[str, int] = {}

            for direction in artifact_directions:
                # Get type-specific system prompt for this artifact type
                system_prompt = get_plan_system_prompt(direction.type)

                for model_cfg in models:
                    for _plan_num in range(plans_per_strat):
                        task_counter += 1
                        type_short = direction.type
                        type_idx_counters[type_short] = type_idx_counters.get(type_short, 0) + 1
                        task_id = f"gen_plan_{type_short}_{type_idx_counters[type_short]}"

                        prompt = get_planner_prompt(
                            hypothesis=hypothesis,
                            artifacts=artifacts,
                            artifact_direction=direction,
                            compute_profiles=compute_profiles_dict,
                            artifact_type_profiles=artifact_type_profiles,
                            user_folder_path=user_uploads_path,
                        )

                        task_configs.append((task_id, prompt, system_prompt, model_cfg, direction))

            emit.status_public_info(f"Running {len(task_configs)} planners...")

            # v26: no skip-mask. send_user_msg replays the whole module from
            # scratch on the same run; sibling preloading is gone with forks.
            preloaded_results: list[tuple[str, str, dict]] = []
            task_configs_to_dispatch = list(task_configs)

            # Run remaining planners in parallel
            sem = asyncio.Semaphore(max_parallel) if max_parallel else None

            async def run_task(
                task_name: str,
                prompt: str,
                system_prompt: str,
                model_cfg: dict,
                direction: ArtifactDirection,
            ):
                async with sem if sem else nullcontext():
                    if use_claude_agent:
                        # Per-task CWD so parallel agents don't collide
                        task_cwd = (
                            (output_dir / task_name)
                            if output_dir
                            else Path.cwd().resolve() / task_name
                        )
                        task_cwd.mkdir(parents=True, exist_ok=True)
                        return (
                            task_name,
                            direction.id,
                            await gen_plan_for_art_claude_agent(
                                prompt=prompt,
                                system_prompt=system_prompt,
                                agent_cfg=claude_cfg,
                                cwd=task_cwd,
                                iteration=iteration,
                                artifact_direction=direction,
                                task_name=task_name,
                                parent_module_id=mid,
                            ),
                        )
                    return (
                        task_name,
                        direction.id,
                        await gen_plan_for_art(
                            task_name=task_name,
                            parent_module_id=mid,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model=model_cfg["model"],
                            api_key=openrouter_key,
                            iteration=iteration,
                            artifact_direction=direction,
                            reasoning_effort=model_cfg.get("reasoning_effort", "medium"),
                            suffix=model_cfg.get("suffix"),
                            llm_timeout=llm_timeout,
                        ),
                    )

            dispatched = await asyncio.gather(
                *[
                    run_task(task_id, prompt, system_prompt, model_cfg, direction)
                    for task_id, prompt, system_prompt, model_cfg, direction in task_configs_to_dispatch
                ],
                return_exceptions=True,
            )
            # Preloaded results pass through the same post-processing as
            # dispatched results — Plan ids get reassigned for the new run, and
            # downstream code has no way to tell which were loaded vs run fresh.
            task_results = preloaded_results + list(dispatched)

            # Assemble Plan objects from task results — idx suffix for ordering visibility in logs
            all_plans: list[BasePlan] = []
            existing_artifact_ids = {a.id for a in artifacts}
            plan_idx = 0

            for tr in task_results:
                if isinstance(tr, BaseException):
                    emit.status_public_warning(f"GenPlan task failed: {tr}")
                    continue
                task_id, direction_id, p_dict = tr
                if p_dict is None:
                    continue
                try:
                    plan_idx += 1
                    p_type = PlanType(p_dict.get("type", "research"))

                    # Filter out non-existent artifact dependencies (each dep is {id, label})
                    raw_deps = p_dict.get("artifact_dependencies", []) or []
                    valid_deps: list[dict] = []
                    dropped: list[str] = []
                    for d in raw_deps:
                        dep_id = d.get("id", "") if isinstance(d, dict) else getattr(d, "id", "")
                        dep_label = (
                            d.get("label", "") if isinstance(d, dict) else getattr(d, "label", "")
                        )
                        if dep_id in existing_artifact_ids:
                            valid_deps.append({"id": dep_id, "label": dep_label})
                        else:
                            dropped.append(dep_id)
                    if dropped:
                        emit.status_private_info(
                            f"Dropped invalid deps from '{p_dict.get('title', 'Untitled')[:30]}': {dropped}",
                        )

                    plan_id = f"{task_id}_idx{plan_idx}"

                    # Build concrete subclass with flat kwargs
                    metadata_keys = {
                        "id",
                        "type",
                        "in_art_direction_id",
                        "in_strat_id",
                        "artifact_dependencies",
                    }
                    content_fields = {k: v for k, v in p_dict.items() if k not in metadata_keys}

                    cls = get_plan_schema(p_type.value)
                    plan = cls(
                        id=plan_id,
                        artifact_dependencies=valid_deps,
                        in_art_direction_id=p_dict.get("in_art_direction_id"),
                        in_strat_id=direction_to_strategy.get(direction_id),
                        **content_fields,
                    )

                    # Validate compute_profile (runpod mode only)
                    if exec_cfg.mode == "runpod" and artifact_type_profiles:
                        profile_errors = verify_compute_profile(
                            p_dict,
                            p_type.value,
                            artifact_type_profiles,
                        )
                        if profile_errors:
                            for err in profile_errors:
                                emit.status_public_warning(f"ComputeProfile: {err}")
                            # Default to first allowed profile for this type
                            allowed = artifact_type_profiles.get(p_type.value, [])
                            if allowed:
                                plan.runpod_compute_profile = allowed[0]
                                emit.status_private_info(
                                    f"Defaulting compute_profile to '{allowed[0]}' for {plan_id}",
                                )

                    # No more plan_pool.add — module_output emit below writes the data
                    # and ctx.invention_loop_group.get_plans() reconstitutes it.
                    all_plans.append(plan)

                except Exception as e:
                    emit.status_public_warning(f"Failed to create plan: {e}")
                    raise

            emit.status_public_success(f"GEN_PLAN complete: {len(all_plans)} plans generated")

            from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
                GenPlanOut,
            )

            emit.module_output(
                module_id=mid,
                name="gen_plan",
                output=GenPlanOut(plans=all_plans),
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return all_plans
