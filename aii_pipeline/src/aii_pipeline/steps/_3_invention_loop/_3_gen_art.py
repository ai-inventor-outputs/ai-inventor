"""GEN_ART Module - Execute selected plans via type-specific artifact executors.

Contains executors for each artifact type:
- RESEARCH: Research questions answered via web search (OpenRouter or Claude agent)
- EXPERIMENT: Methodology implementations (Claude Code agent)
- DATASET: HuggingFace/OWID dataset search and download (Claude Code agent)
- EVALUATION: Experiment result assessment (Claude Code agent)
- PROOF: Lean 4 formal proofs (Claude Code agent)

All artifact types execute in parallel, controlled by max_concurrent_agents semaphore.
This allows mixed execution (e.g., 2 research + 1 experiment + 2 datasets at once).

Telemetry structure:
- GEN_ART (module)
  └── Tasks for each artifact execution (FND_exec_*, EXP_exec_*, etc.)

Adds artifacts to pool (both successes AND failures).
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.run import current_run, emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import ParallelTModule

from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
    BasePlan,
    PlanType,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    ArtifactType,
    BaseArtifact,
)
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig


def make_artifact(
    *,
    id: str,  # noqa: A002 — matches BaseArtifact.id field on the data model
    plan: BasePlan,
    iteration: int = 0,
    title: str = "",
    summary: str = "",
    layman_summary: str = "",
    workspace_path: str | None = None,
    out_expected_files: list[str] | None = None,
    out_demo_files: list[ExpectedFile] | None = None,
    out_dependency_files: dict[str, str | list[str] | None] | None = None,
) -> BaseArtifact:
    """Build a ``BaseArtifact`` from a plan + executor results.

    Type derives from ``plan.type``; title defaults to ``plan.title``;
    ``in_dependencies`` is copied from ``plan.artifact_dependencies``.
    ``iteration`` is the invention_loop iter that produced this artifact
    (1-based) — downstream gen_paper_repo uses it to route per-iter.
    """
    return BaseArtifact(
        id=id,
        type=ArtifactType(plan.type.value),
        iteration=iteration,
        title=title or plan.title,
        in_plan_id=plan.id,
        in_dependencies=list(plan.artifact_dependencies),
        summary=summary,
        layman_summary=layman_summary,
        workspace_path=workspace_path,
        out_expected_files=out_expected_files or [],
        out_demo_files=out_demo_files or [],
        out_dependency_files=out_dependency_files or {},
    )


# Import artifact executors
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.out_schema import (
    DatasetArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.out_schema import (
    EvaluationArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.out_schema import (
    ExperimentArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.out_schema import (
    ProofArtifact,
)

# Import schema modules for file metadata functions
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.out_schema import (
    ResearchArtifact,
)

from .executors import (
    execute_dataset,
    execute_evaluation,
    execute_experiment,
    execute_proof,
    execute_research,
)

if TYPE_CHECKING:
    from .invention_loop import LoopCtx

# Map plan types to their artifact schema classes (for file metadata)
PLAN_TO_SCHEMA = {
    PlanType.RESEARCH: ResearchArtifact,
    PlanType.EXPERIMENT: ExperimentArtifact,
    PlanType.DATASET: DatasetArtifact,
    PlanType.EVALUATION: EvaluationArtifact,
    PlanType.PROOF: ProofArtifact,
}


async def exec_plan(
    plan: BasePlan,
    config: PipelineConfig,
    artifacts: list[BaseArtifact],
    iteration: int,
    run_dir: Path,
    research_idx: int = 0,
    dataset_idx: int = 0,
    experiment_idx: int = 0,
    evaluation_idx: int = 0,
    proof_idx: int = 0,
    task_id: str | None = None,
    task_name: str | None = None,
    parent_module_id: str = "",
    user_uploads_path: str = "",
) -> BaseArtifact | None:
    """
    Execute a single plan.

    Dispatches to the appropriate unit based on plan type.
    Handles inner refinement loops and failure cases.

    Returns:
        Created artifact (success or failure), or None on critical error
    """
    if task_id and task_name:
        emit.status_private_info(f"Executing plan: {plan.id} ({plan.type.value})")
        emit.status_private_info(f"  Title: {plan.title}")

    # Log missing dependencies — scan the artifacts list once
    _by_id = {a.id: a for a in artifacts}
    for dep in plan.artifact_dependencies:
        if dep.id not in _by_id:
            if task_id and task_name:
                emit.status_public_warning(f"Missing dependency: {dep.id}")

    # Dispatch to appropriate unit
    try:
        if plan.type == PlanType.RESEARCH:
            result, is_success = await execute_research(
                plan=plan,
                artifacts=artifacts,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                research_idx=research_idx,
                task_id=task_id,
                task_name=task_name,
                parent_module_id=parent_module_id,
                user_uploads_path=user_uploads_path,
            )

        elif plan.type == PlanType.EXPERIMENT:
            result, is_success = await execute_experiment(
                plan=plan,
                artifacts=artifacts,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                experiment_idx=experiment_idx,
                task_id=task_id,
                task_name=task_name,
                parent_module_id=parent_module_id,
                user_uploads_path=user_uploads_path,
            )

        elif plan.type == PlanType.DATASET:
            result, is_success = await execute_dataset(
                plan=plan,
                artifacts=artifacts,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                dataset_idx=dataset_idx,
                task_id=task_id,
                task_name=task_name,
                parent_module_id=parent_module_id,
                user_uploads_path=user_uploads_path,
            )

        elif plan.type == PlanType.EVALUATION:
            result, is_success = await execute_evaluation(
                plan=plan,
                artifacts=artifacts,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                evaluation_idx=evaluation_idx,
                task_id=task_id,
                task_name=task_name,
                parent_module_id=parent_module_id,
                user_uploads_path=user_uploads_path,
            )

        elif plan.type == PlanType.PROOF:
            result, is_success = await execute_proof(
                plan=plan,
                artifacts=artifacts,
                config=config,
                run_dir=run_dir,
                iteration=iteration,
                proof_idx=proof_idx,
                task_id=task_id,
                task_name=task_name,
                parent_module_id=parent_module_id,
                user_uploads_path=user_uploads_path,
            )

        else:
            if task_id and task_name:
                emit.status_public_error(f"Unknown plan type: {plan.type}")
            return None

    except Exception as e:
        if task_id and task_name:
            emit.status_public_error(f"Execution failed for {plan.id}: {e}")
        raise

    # Only add successful artifacts to pool (pool only stores successes).
    # task_end is emitted by the executor (end_task_failure / end_task_success)
    # in every code path; we just log the failure as a status message here.
    if not is_success:
        error_msg = result.get("error", "Unknown error")
        if task_id and task_name and plan.type != PlanType.RESEARCH:
            emit.status_public_warning(f"Failed: {error_msg}")
        return None

    # Get file metadata from schema class
    schema_class = PLAN_TO_SCHEMA.get(plan.type)
    if schema_class:
        # Convert ExpectedFile objects to strings (just paths)
        expected_files_raw = schema_class.get_expected_out_files()
        out_expected_files = [f.path if hasattr(f, "path") else str(f) for f in expected_files_raw]
        out_demo_files = schema_class.model_fields["out_demo_files"].default or []
    else:
        out_expected_files = []
        out_demo_files = []

    # Extract fields from result dict (populated by executors)
    artifact_title = result.get("title", "") or plan.title
    artifact_summary = result.get("summary", "")

    out_dependency_files: dict = {}
    if "file_list" in result:
        out_dependency_files["file_list"] = result["file_list"]
    elif out_expected_files:
        # Fallback: use expected output files as dependency file list
        out_dependency_files["file_list"] = out_expected_files
    if "data_file_paths" in result:
        out_dependency_files["data_file_paths"] = result["data_file_paths"]

    artifact = make_artifact(
        id=task_id,
        plan=plan,
        iteration=iteration,
        title=artifact_title,
        summary=artifact_summary,
        layman_summary=result.get("layman_summary", ""),
        workspace_path=result.get("workspace_path"),
        out_expected_files=out_expected_files,
        out_demo_files=out_demo_files,
        out_dependency_files=out_dependency_files,
    )

    # Emit success message (executors already call end_task_success()
    # which fires task_end; we only emit the human-readable status here).
    if task_id and task_name and plan.type != PlanType.RESEARCH:
        emit.status_public_success(f"{artifact.id} completed")

    return artifact


@dataclass
class GenArtCtx(ModuleCtx):
    """Substep ctx for gen_art."""

    parent_ctx: "LoopCtx | None" = None
    iteration: int = 1
    plans: list[BasePlan] | None = None
    parent_id: str = ""


class GenArtModule(ParallelTModule):
    """gen_art substep — execute selected plans via type-specific units."""

    kind: Literal["gen_art_module"] = "gen_art_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_art"] = "gen_art"

    def get_context(
        self,
        *,
        ctx: "LoopCtx",
        iteration: int,
        output_dir: Path | None = None,
        plans: list[BasePlan] | None = None,
        parent_id: str,
    ) -> GenArtCtx:
        return GenArtCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            plans=list(plans) if plans else None,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: "LoopCtx",
        iteration: int,
        output_dir: Path | None = None,
        plans: list[BasePlan] | None = None,
        parent_id: str,
    ) -> list[BaseArtifact]:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                output_dir=output_dir,
                plans=plans,
                parent_id=parent_id,
            )
        ):
            """Run the EXECUTE step.

            Executes all selected plans via their type-specific units.
            Can run independent plans in parallel.
            """
            config = ctx.config
            artifacts = ctx.invention_loop_group.get_artifacts()
            user_uploads_path = ctx.user_uploads_path
            run_dir = ctx.run_dir or ctx.output_dir

            # Step subdir within iteration dir (consistent with gen_strat, gen_plan, gen_paper_text)
            if output_dir:
                step_dir = (output_dir / "gen_art").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            # GEN_ART is a single module - all artifact executions are tasks within it
            mid = emit.start_parallel_module(
                name="gen_art",
                parent_id=parent_id,
            )

            allowed_artifacts = config.invention_loop.allowed_artifacts

            # Use explicitly passed plans, or all plans from pool for this iteration
            if plans:
                selected = plans
            else:
                selected = ctx.invention_loop_group.get_plans(iteration=iteration)

            # Filter by allowed artifacts if specified
            if allowed_artifacts:
                before_count = len(selected)
                selected = [p for p in selected if p.type.value in allowed_artifacts]
                if len(selected) < before_count:
                    emit.status_public_info(
                        f"Filtered to allowed artifacts {allowed_artifacts}: {before_count} -> {len(selected)}"
                    )

            if not selected:
                emit.status_public_warning("No plans selected for execution")
                emit.end_module(
                    parent_id=parent_id,
                    module_id=mid,
                )
                return []

            emit.status_public_info(f"Executing {len(selected)} plans:")
            for p in selected:
                deps_str = (
                    f" (deps: {', '.join(f'{d.id}[{d.label}]' if d.label else d.id for d in p.artifact_dependencies)})"
                    if p.artifact_dependencies
                    else ""
                )
                emit.status_public_info(f"  - [{p.id}] {p.type.value}: {p.title[:50]}{deps_str}")

            # Get concurrency limit from config
            max_concurrent = config.invention_loop.execute.max_concurrent_agents
            semaphore = asyncio.Semaphore(max_concurrent)
            emit.status_public_info(f"Max concurrent artifacts: {max_concurrent}")

            # Count plans by type for logging and idx tracking
            type_counts: dict[PlanType, int] = {}
            for p in selected:
                type_counts[p.type] = type_counts.get(p.type, 0) + 1

            # Log breakdown by type
            for ptype, count in sorted(type_counts.items(), key=lambda x: x[0].value):
                emit.status_public_info(f"  {ptype.value.upper()}: {count}")

            # Track idx per type (for workspace naming)
            type_idx_counters: dict[PlanType, int] = dict.fromkeys(PlanType, 0)

            # v26: no skip-mask — send_user_msg replays whole modules from
            # scratch, no sibling preloading.

            async def execute_with_semaphore(
                plan: BasePlan,
                task_counter: int,  # noqa: ARG001 — kept for parity with prior task-tracking sig
            ) -> BaseArtifact | None:
                """Execute a plan with semaphore-controlled concurrency."""
                async with semaphore:
                    # Per-type 1-based index (round-scoped: ``type_idx_counters``
                    # is created fresh on each ``execute_with_semaphore`` invocation).
                    # Used for both the on-disk workspace path AND the task name on
                    # the run bus, so the FE tree can render "Dataset 1" / "Experiment 2"
                    # by reading the slug — the type lives in the name itself.
                    type_idx = type_idx_counters[plan.type]
                    type_idx_counters[plan.type] += 1
                    task_id = f"gen_art_{plan.type.value}_{type_idx + 1}"
                    task_name = task_id

                    # Build kwargs with the correct idx parameter for this type
                    kwargs = {
                        "plan": plan,
                        "config": config,
                        "artifacts": artifacts,
                        "iteration": iteration,
                        "run_dir": output_dir
                        or run_dir,  # iter_N/gen_art/ — executors create workspaces here
                        "task_id": task_id,
                        "task_name": task_name,
                        "parent_module_id": mid,
                        # Set all idx to 0 by default, then override the specific one
                        "research_idx": 0,
                        "dataset_idx": 0,
                        "experiment_idx": 0,
                        "evaluation_idx": 0,
                        "proof_idx": 0,
                        "user_uploads_path": user_uploads_path,
                    }

                    # Override the specific idx for this type
                    idx_key = f"{plan.type.value}_idx"
                    kwargs[idx_key] = type_idx

                    artifact = await exec_plan(**kwargs)

                    # Surface the typed artifact on the run-tree task
                    # so downstream readers + fork replay reach it via
                    # ``task.output`` (no legacy ``outputs=[...]``
                    # plumbing). Task ``name`` is unique per iteration
                    # within the gen_art parallel module.
                    if artifact is not None:
                        run = current_run()
                        parent_module = run.find_module(mid)
                        if parent_module is not None:
                            for t in parent_module.children:
                                if t.name == task_id:
                                    emit.task_output(
                                        task_id=t.node_id,
                                        output=artifact,
                                    )
                                    break
                    return artifact

            # Execute plans in parallel (semaphore controls concurrency).
            tasks = []
            for task_counter, p in enumerate(selected, start=1):
                tasks.append(execute_with_semaphore(p, task_counter))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    emit.status_public_warning(f"GenArt task failed: {r}")
            artifacts: list[BaseArtifact] = [
                r for r in results if r is not None and not isinstance(r, Exception)
            ]

            # Summary - artifacts only contains successes (pool only stores successes)
            successes = len(artifacts)
            failures = len(selected) - successes

            emit.status_public_success(
                f"EXECUTE complete: {successes} successes, {failures} failures"
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return artifacts
