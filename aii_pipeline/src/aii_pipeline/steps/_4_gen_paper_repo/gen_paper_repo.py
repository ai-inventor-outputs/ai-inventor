"""Gen Paper Module - Post-loop paper generation.

Takes paper texts + artifacts from invention loop, produces:
- GitHub repo with code artifacts
- Prepared artifacts (notebooks, markdown, Lean playground links)
- Visualizations (image generation)
- Paper draft compiled to LaTeX/PDF

STRICTLY SEQUENTIAL EXECUTION:

    1: gen_repo         (resolve repo URL — no network)
    2: gen_viz          (image generation for paper figures)
    3: gen_demos        (per-artifact demo notebooks)
    4: gen_full_paper   (compile paper to LaTeX/PDF)
    5: deploy_gh        (clone, push src + demos + paper to GitHub)

No asyncio.gather, no background deploy phases — keeps the fork story
simple (every substep has one predecessor) at the cost of giving up
demos/viz parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aii_lib.run import current_run, emit
from aii_lib.run.context import ctx_scope, current_ctx
from aii_lib.run.mdgroup import SeqMdGroup

# Lazy imports to avoid circular: steps.__init__ → run.py → prompts → steps.base → steps.__init__
# Step runner modules and schemas are imported inside functions that use them.
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
    BaseDemo,
)
from aii_pipeline.steps._3_invention_loop.invention_loop import InventionLoopGroup
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import (
    DEFAULT_MIN_TOKEN_VALIDITY_SECONDS,
    PipelineConfig,
    rel_path,
)

if TYPE_CHECKING:
    from aii_pipeline.steps.base import StepContext


# Substep execution order — strictly sequential, viz before demos. Used
# by the resume / skip-ahead logic in :meth:`GenPaperRepoGroup.execute`
# and the substep-order validation in :meth:`get_context`.
GEN_PAPER_STEPS = ["gen_repo", "gen_viz", "gen_demos", "gen_full_paper", "deploy_gh"]


def _effective_index(step: str) -> int:
    return GEN_PAPER_STEPS.index(step)


# ---------------------------------------------------------------------------
# Phase ctx + phase MdGroup subclass — moved from aii_pipeline/run/scaffold.py
# per REFACTOR_PLAN §7.
# ---------------------------------------------------------------------------


@dataclass
class GenPaperRepoPhaseCtx:
    """Phase ctx for ``gen_paper_repo``.

    Distinct from the per-substep :class:`GenPaperCtx` (below) which
    carries substep-internal state. This phase ctx is just the boundary
    info ``GenPaperRepoGroup.execute`` needs. Pre-Stage-7 it also
    carried ``start_substep`` for resume skip-ahead; v27 replay-execute
    (Stages 1-9) made that field unnecessary, removed in Stage 10.
    """

    config: PipelineConfig
    run_dir: Path
    invention_loop_result: Any


class GenPaperRepoGroup(SeqMdGroup):
    """Phase 4 — gen_paper_repo (flat, no extra fields).

    NOTE: ``get_figures`` / ``get_demos`` walk for ``module_output``
    events with ``module_name == "gen_viz"`` / ``"gen_art_demo"``.
    """

    kind: Literal["gen_paper_repo_group"] = "gen_paper_repo_group"
    """Per-subclass discriminator (see ``HypoLoopGroup.kind``)."""

    def get_figures(self) -> list:
        """All figures produced by gen_viz.

        gen_viz is a parallel module whose tasks each surface one
        :class:`Figure` on ``task.output`` (set by per-task
        ``task_output`` events from ``_2_gen_viz.py``).
        """
        from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import (
            Figure,
        )

        out: list = []
        for m in self.children:
            if getattr(m, "name", None) != "gen_viz":
                continue
            for t in getattr(m, "children", []) or []:
                fig = getattr(t, "output", None)
                if isinstance(fig, Figure):
                    out.append(fig)
        return out

    def get_demos(self) -> list:
        """All demos produced by gen_art_demo (CodeDemo / LeanDemo / MarkdownDemo).

        Reads the module's ``output: GenArtDemoOut`` (aggregator typed
        Pydantic model) — populated by the ``module_output`` event in
        ``utils/step_runner.py:step_gen_demos``.
        """
        from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
            GenArtDemoOut,
        )

        for m in self.children:
            if getattr(m, "name", None) != "gen_art_demo":
                continue
            out = getattr(m, "output", None)
            if isinstance(out, GenArtDemoOut):
                return list(out.demos)
        return []

    def get_context(self) -> GenPaperRepoPhaseCtx:
        parent: StepContext = current_ctx()
        invention_loop_group = current_run().find_group_by_name("invention_loop")
        invention_loop_result = (
            invention_loop_group.output if invention_loop_group is not None else None
        )
        if not invention_loop_result:
            emit.status_public_error(
                "gen_paper_repo requires invention_loop result",
            )
            raise ValueError("gen_paper_repo requires invention_loop result")
        return GenPaperRepoPhaseCtx(
            config=parent.config,
            run_dir=parent.run_dir,
            invention_loop_result=invention_loop_result,
        )

    async def execute(self) -> Any:
        """Run the full gen_paper_repo phase strictly sequentially:

        1: gen_repo        (resolve URL — repo creation deferred to step 5)
        2: gen_viz         (image generation)
        3: gen_demos       (per-artifact demo notebooks)
        4: gen_full_paper  (compile LaTeX/PDF)
        5: deploy_gh       (clone repo, push src + demos + paper)
        """
        from .utils.step_runner import (
            step_deploy_gh,
            step_gen_demos,
            step_gen_full_paper,
            step_gen_repo,
            step_gen_viz,
        )

        with ctx_scope(self.get_context()) as outer:
            config = outer.config
            invention_loop_result = outer.invention_loop_result
            run_dir = outer.run_dir

            # Create output directory
            if run_dir:
                output_dir = run_dir / "4_gen_paper_repo"
            else:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                output_base = config.outputs_directory
                output_dir = Path(f"{output_base}/{timestamp}_gen_paper")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract data from invention loop result
            hypothesis = invention_loop_result.hypothesis

            # Read pools from the live Run's InventionLoopGroup. The pools
            # were populated by deserialized ``module_output`` events when
            # the Run was rebuilt from JSONL — no on-disk pools/ JSON
            # needed.
            inv_group = current_run().find_group_by_name("invention_loop")
            if not isinstance(inv_group, InventionLoopGroup):
                raise TypeError(
                    "gen_paper_repo expected an InventionLoopGroup on the live Run "
                    "(was the invention_loop phase replayed correctly?)",
                )
            all_artifacts = inv_group.get_artifacts()
            all_paper_texts = inv_group.get_paper_texts()

            # gen_paper_repo accumulates figures/demos as plain lists in the
            # step context — they're emitted as ``module_output`` events at
            # the end of each substep so downstream views can read them off
            # the run tree.
            figures: list = []
            demos: list = []

            # Read auth config for token validity checks between steps
            _auth_cfg = (
                config.raw.get("agent_backend", {})
                .get("claude_agent_sdk", {})
                .get("llm_backend", {})
                .get("claude_max", {})
                .get("auth", {})
            )
            min_token_validity = _auth_cfg.get(
                "min_token_validity_seconds",
                DEFAULT_MIN_TOKEN_VALIDITY_SECONDS,
            )

            # =========================================================================
            # STEP CONFIGURATION
            # =========================================================================
            last_step = None
            first_step: str | None = None
            run_gen_full_paper = True

            # Skip/stop helpers — strictly sequential, no concurrent-group fudging.
            # ``first_step`` / ``last_step`` are no longer threaded from
            # the pipeline boot (Stage 6 + 7 of v27 replay-execute);
            # they remain as local hooks in case a future feature needs
            # to slice this phase for testing.
            def should_skip(step: str) -> bool:
                """Skip if step is before first_step."""
                if not first_step:
                    return False
                return _effective_index(step) < _effective_index(first_step)

            def should_stop(step: str) -> bool:
                """Stop after this step if it matches last_step."""
                return bool(last_step) and last_step == step

            gen_paper_repo_gid = emit.start_seq_group(name="gen_paper_repo")
            assert gen_paper_repo_gid == self.node_id, (
                f"gen_paper_repo scaffold drift: gid={gen_paper_repo_gid} "
                f"self.node_id={self.node_id}"
            )

            emit.status_private_info(
                f"Hypothesis: {hypothesis.get('title', 'N/A')[:50]}",
            )
            emit.status_public_info(f"Artifacts: {len(all_artifacts)}")
            latest_paper = all_paper_texts[-1] if all_paper_texts else None
            emit.status_private_info(
                f"Paper texts: {len(all_paper_texts)} "
                f"(latest: {latest_paper.id if latest_paper else 'N/A'})",
            )
            emit.status_private_info(f"Output: {rel_path(output_dir)}")

            # Log step configuration
            step_range = f"{first_step or 'gen_repo'} → {last_step or 'deploy_gh'}"
            emit.status_private_info(f"Steps: {step_range}")

            # Build context bag for step functions
            ctx = GenPaperCtx(
                config=config,
                hypothesis=hypothesis,
                output_dir=output_dir,
                figures=figures,
                demos=demos,
                all_artifacts=all_artifacts,
                narrative=None,
                min_token_validity=min_token_validity,
                run_gen_full_paper=run_gen_full_paper,
                should_skip=should_skip,
                should_stop=should_stop,
                _all_paper_texts=all_paper_texts,
                gen_paper_repo_gid=gen_paper_repo_gid,
            )

            # Execute steps strictly sequentially. Each may return an early
            # GenPaperRepoOut when stop_after fires, in which case we bail.
            for step_fn in [
                step_gen_repo,
                step_gen_viz,
                step_gen_demos,
                step_gen_full_paper,
                step_deploy_gh,
            ]:
                early = await step_fn(ctx)
                if early is not None:
                    return early

            result = ctx.result

            emit.status_private_info(f"Output: {rel_path(output_dir)}")
            emit.status_private_info(
                f"Paper: {result.metadata.get('paper_pdf', result.metadata.get('paper_tex', 'N/A'))}",
            )
            emit.status_private_info(f"Figures: {len(ctx.figures)}")
            if result.gist_deployments:
                emit.status_private_info(
                    f"Gists: {len(result.gist_deployments)}",
                )
            if ctx.repo_url:
                emit.status_public_success(f"   GitHub Repo: {ctx.repo_url}")
                emit.status_private_info(
                    f"GitHub Repository: {ctx.repo_url}",
                )

            emit.end_group(gen_paper_repo_gid)

            # No more figure_pool/demo_pool sidecar JSON — figures/demos live
            # on the run tree as gen_viz / gen_art_demo module_output
            # events when those substeps emit them.

            return result


@dataclass
class GenPaperCtx(ModuleCtx):
    """Mutable context bag passed through gen_paper_repo substep helpers.

    Inherits config/output_dir from ModuleCtx; adds gen_paper-specific
    pools, callbacks, and accumulated state.
    """

    # Required gen_paper_repo inputs (no defaults — must be passed)
    gen_paper_repo_gid: str = ""
    """Node-id of the ``gen_paper_repo`` SeqMdGroup. Substeps thread this
    into ``start_*_module(parent_id=…)`` and ``end_module(parent_id=…)``
    so the dispatcher can attach modules under the right group node.
    Without this every substep would fall back to the literal string
    ``"gen_paper_repo"`` which the v26 ``_attach_module`` resolver
    rejects (it wants a real node_id, not a name)."""
    hypothesis: dict = None  # type: ignore[assignment]
    all_artifacts: list = None  # type: ignore[assignment]
    narrative: Any = None
    min_token_validity: int = 0
    run_gen_full_paper: bool = True
    should_skip: Any = None  # Callable[[str], bool]
    should_stop: Any = None  # Callable[[str], bool]
    _all_paper_texts: list | None = None
    # Accumulated results — plain lists, no pool wrappers
    figures: list = field(default_factory=list)
    demos: list[BaseDemo] = field(default_factory=list)
    repo_url: str | None = None
    repo_name: str | None = None
    repo_description: str | None = None
    paper_texts: list = field(default_factory=list)
    paper: Any = None  # PaperText | None
    prepared_artifacts: list = field(default_factory=list)
    gist_deployments: list = field(default_factory=list)
    result: Any = None  # GenPaperRepoOut | None
