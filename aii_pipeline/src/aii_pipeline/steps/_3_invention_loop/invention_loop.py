#!/usr/bin/env python3
"""
Invention Loop Module - Iterative Scientific Invention.

Implements the six-step invention loop:
1. GEN_STRAT       (module) - Generate strategic plans from multiple LLMs
2. GEN_PLAN        (module) - Generate plans from strategy's artifact directions
3. GEN_ART         (MODULE GROUP) - Execute selected plans to generate artifacts
   └── RESEARCH, DATASET, EXPERIMENT, EVALUATION, PROOF (sub-modules)
4. GEN_PAPER_TEXT   (module) - Generate paper draft from artifact pool
5. REVIEW_PAPER    (module) - External adversarial review
6. UPD_HYPO        (module) - Internal hypothesis revision

Telemetry hierarchy:
- INVENTION_LOOP (module group)
  └── iter_1 (module group per iteration)
      └── GEN_STRAT, GEN_PLAN, GEN_ART, GEN_PAPER_TEXT, REVIEW_PAPER, UPD_HYPO

The loop continues until:
- Max iterations reached
- Human signals exit

Five pools track state:
- StrategyPool: Strategic AII prompts
- PlanPool: Pending work
- ArtifactPool: Completed work (successes and failures)
- PaperTextPool: Paper drafts across iterations
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token_fresh
from aii_lib.run import current_run, emit
from aii_lib.run.context import ctx_scope, current_ctx
from aii_lib.run.loop_iteration import LoopIteration
from aii_lib.run.mdgroup import LoopMdGroup

from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
    format_critiques_for_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop.out_schema import InventionLoopOut
from aii_pipeline.run.utils.group_helpers import _module_output_for
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import (
    DEFAULT_MIN_TOKEN_VALIDITY_SECONDS,
    PipelineConfig,
    rel_path,
    retry_until_result,
)

if TYPE_CHECKING:
    from aii_pipeline.steps.base import StepContext


# ---------------------------------------------------------------------------
# Phase ctx + phase MdGroup subclass — moved from aii_pipeline/run/scaffold.py
# per REFACTOR_PLAN §7.
# ---------------------------------------------------------------------------


@dataclass
class InventionLoopCtx:
    """Phase ctx for ``invention_loop`` — phase-entry view.

    Distinct from the per-iteration :class:`LoopCtx` (below) which
    carries iter-internal state (accumulated review feedback, etc.).
    This phase ctx is just the boundary information
    ``InventionLoopGroup.execute`` needs to start: hypothesis from
    upstream, run dir, user uploads path. Pre-Stage-7 it also threaded
    ``start_iter`` / ``start_substep`` for resume skip-ahead; v27
    replay-execute (Stages 1-9) made those unnecessary, removed in
    Stage 10.
    """

    config: PipelineConfig
    run_dir: Path
    user_uploads_path: str
    hypothesis: dict


class InventionLoopGroup(LoopMdGroup):
    """Phase 3 — invention_loop with iterative substeps.

    Iterations of gen_strat → gen_plan → gen_art → gen_paper_text →
    review_paper → upd_hypo.
    """

    kind: Literal["invention_loop_group"] = "invention_loop_group"
    """Per-subclass discriminator (see ``HypoLoopGroup.kind``)."""

    def _iters(self, iteration: int | None) -> list[LoopIteration]:
        if iteration is None:
            return self.children
        it = self.find_iteration(iteration)
        return [it] if it else []

    def get_strategies(self, iteration: int | None = None) -> list:
        """All strategies produced by gen_strat, optionally filtered to one iter.

        Reads ``Module.output.strategies`` (typed :class:`GenStratOut`)
        — populated by the per-iteration ``module_output`` event.
        """
        from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
            GenStratOut,
        )

        out: list = []
        for it in self._iters(iteration):
            agg = _module_output_for(it.children, "gen_strat")
            if isinstance(agg, GenStratOut):
                out.extend(agg.strategies)
        return out

    def get_plans(self, iteration: int | None = None) -> list:
        """All plans produced by gen_plan, optionally filtered to one iter.

        Reads ``Module.output.plans`` (typed :class:`GenPlanOut`).
        """
        from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
            GenPlanOut,
        )

        out: list = []
        for it in self._iters(iteration):
            agg = _module_output_for(it.children, "gen_plan")
            if isinstance(agg, GenPlanOut):
                out.extend(agg.plans)
        return out

    def get_artifacts(self, iteration: int | None = None) -> list:
        """All artifacts produced by gen_art, optionally filtered to one iter.

        gen_art is a parallel module whose tasks each surface one
        :class:`BaseArtifact` (subclass) on ``task.output`` — walk
        every gen_art module's children and collect the per-task
        outputs. Readers downstream don't need the parent
        ``ModuleOutputMessage`` aggregate.
        """
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
            BaseArtifact,
        )

        out: list = []
        for it in self._iters(iteration):
            for m in it.children:
                if getattr(m, "name", None) != "gen_art":
                    continue
                for t in getattr(m, "children", []) or []:
                    art = getattr(t, "output", None)
                    if isinstance(art, BaseArtifact):
                        out.append(art)
        return out

    def get_paper_texts(self, iteration: int | None = None) -> list:
        """All paper drafts produced by gen_paper_text, optionally filtered to one iter.

        Reads ``Module.output`` (typed :class:`PaperText`) for each
        iteration's ``gen_paper_text`` module — single output per
        module, accumulated across iterations.
        """
        from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
            PaperText,
        )

        out: list = []
        for it in self._iters(iteration):
            t = _module_output_for(it.children, "gen_paper_text")
            if isinstance(t, PaperText):
                out.append(t)
        return out

    def get_context(self) -> "InventionLoopCtx":
        parent: StepContext = current_ctx()
        hypo_loop_group = current_run().find_group_by_name("hypo_loop")
        if hypo_loop_group is None:
            emit.status_public_error(
                "invention_loop requires hypo_loop group in the run tree",
            )
            raise ValueError("invention_loop requires hypo_loop group")
        # Prefer the typed :class:`HypoLoopOut` left on
        # ``hypo_loop_group.output`` by the ``mdgroup_output`` event
        # — falls back to the live tree-walk synthesizer when the
        # event hasn't been dispatched yet (mid-run before the phase
        # closes, or older logs predating the typed-output emit).
        hypo_loop = hypo_loop_group.output
        if hypo_loop is None:
            hypo_loop = hypo_loop_group.synthesize_result_from_tree()
        review_result = hypo_loop.review_hypo
        gen_hypo_result = hypo_loop.gen_hypo
        if review_result and getattr(review_result, "hypothesis", None):
            hypothesis = review_result.hypothesis
        elif gen_hypo_result and getattr(gen_hypo_result, "hypotheses", None):
            hypothesis = gen_hypo_result.hypotheses[0]
        else:
            emit.status_public_error(
                "invention_loop requires a hypothesis from hypo_loop",
            )
            raise ValueError("invention_loop requires a hypothesis from hypo_loop")
        return InventionLoopCtx(
            config=parent.config,
            run_dir=parent.run_dir,
            user_uploads_path=str(parent.run_dir / "user_uploads"),
            hypothesis=hypothesis,
        )

    async def execute(self) -> Any:
        """Run the full invention loop.

        The loop: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_PAPER_TEXT →
        REVIEW_PAPER → UPD_HYPO. Each iteration's substep is dispatched
        via ``await substeps[<name>].execute(...)`` against the typed
        Module subclasses the scaffold pre-instantiated under each iter.
        """
        with ctx_scope(self.get_context()) as outer:
            config = outer.config

            # Start the loop group (idempotent — scaffold pre-created
            # us). setup_loop reads the group off the live Run.
            loop_gid = emit.start_loop_group(name="invention_loop")
            assert loop_gid == self.node_id, (
                f"invention_loop scaffold drift: gid={loop_gid} self.node_id={self.node_id}"
            )

            # Setup: parse config, build the per-iter ctx.
            ctx = setup_loop(
                config,
                outer.hypothesis,
                outer.run_dir,
                user_uploads_path=outer.user_uploads_path,
            )

            # Local refs for read-only / iteration-control fields
            output_dir = ctx.output_dir
            max_iterations = ctx.max_iterations
            start_paper_text_at = ctx.start_paper_text_at
            min_token_validity = ctx.min_token_validity
            reviewer_feedback_text = (
                format_critiques_for_prompt(ctx.reviewer_feedback)
                if ctx.reviewer_feedback
                else None
            )

            # Review scheduling config
            upd_hypo_start_at = config.invention_loop.upd_hypo.start_at_iteration
            review_paper_start_at = config.invention_loop.review_paper.start_at_iteration

            # Log header
            emit.status_private_info(
                f"Hypothesis: {ctx.hypothesis.get('title', 'N/A')}",
            )
            emit.status_public_progress(
                f"Iterations: 1-{max_iterations}",
            )
            emit.status_public_info(
                f"Paper text starts at: iter {start_paper_text_at}",
            )
            emit.status_public_info(
                f"Review paper starts at: iter {review_paper_start_at}",
            )
            emit.status_public_info(
                f"Upd hypo starts at: iter {upd_hypo_start_at}",
            )
            emit.status_private_info(f"Output: {rel_path(output_dir)}")

            iterations_completed = 0

            for iteration in range(1, max_iterations + 1):
                emit.status_public_info(
                    f"ITERATION {iteration}/{max_iterations}",
                )
                # Start iteration
                iter_id = emit.start_iteration(
                    group_id=loop_gid,
                    iteration=iteration,
                )
                iter_node = current_run().find_node(iter_id)
                substeps = {m.name: m for m in iter_node.children}

                # Create iteration output directory
                iter_dir = output_dir / f"iter_{iteration}"
                iter_dir.mkdir(parents=True, exist_ok=True)

                # =================================================================
                # STEP 1: GEN_STRAT - Generate strategic plans
                # =================================================================
                emit.status_public_progress("\nSTEP 1: GEN_STRAT")
                prev_strats = [s.model_dump() for s in self.get_strategies(iteration=iteration - 1)]

                _papers = self.get_paper_texts()
                current_paper = _papers[-1].paper_text if _papers else ""

                strategies = await retry_until_result(
                    partial(
                        substeps["gen_strat"].execute,
                        ctx=ctx,
                        iteration=iteration,
                        output_dir=iter_dir,
                        previous_strategies=prev_strats or None,
                        reviewer_feedback_text=reviewer_feedback_text,
                        paper_text=current_paper,
                        parent_id=iter_id,
                    ),
                    retries=3,
                )

                emit.status_public_info(
                    f"Generated {len(strategies)} strategies",
                )

                if not strategies:
                    emit.status_public_warning(
                        "No strategies after 3 attempts, ending loop",
                    )
                    emit.end_iteration(
                        group_id=loop_gid,
                        iteration=iteration,
                    )
                    break

                if min_token_validity:
                    ensure_oauth_token_fresh(min_token_validity)

                # =================================================================
                # STEP 2: GEN_PLAN - Generate plans from ALL strategies
                # =================================================================
                emit.status_public_progress("\nSTEP 2: GEN_PLAN")

                plans = await retry_until_result(
                    partial(
                        substeps["gen_plan"].execute,
                        ctx=ctx,
                        iteration=iteration,
                        output_dir=iter_dir,
                        strategies=strategies,
                        parent_id=iter_id,
                    ),
                    retries=3,
                )

                emit.status_public_info(f"Generated {len(plans)} plans")

                if not plans:
                    emit.status_public_warning(
                        "No plans after 3 attempts, ending loop",
                    )
                    emit.end_iteration(
                        group_id=loop_gid,
                        iteration=iteration,
                    )
                    break

                if min_token_validity:
                    ensure_oauth_token_fresh(min_token_validity)

                # =================================================================
                # STEP 3: GEN_ART - Execute plans to generate artifacts
                # =================================================================
                emit.status_public_progress("\nSTEP 3: GEN_ART")
                artifacts = await substeps["gen_art"].execute(
                    ctx=ctx,
                    iteration=iteration,
                    output_dir=iter_dir,
                    plans=plans,
                    parent_id=iter_id,
                )
                emit.status_public_info(
                    f"Produced {len(artifacts)} artifacts",
                )

                if min_token_validity:
                    ensure_oauth_token_fresh(min_token_validity)

                # =================================================================
                # STEP 4: GEN_PAPER_TEXT - Generate paper draft
                # =================================================================
                paper_text_result = None
                emit.status_public_progress("\nSTEP 4: GEN_PAPER_TEXT")
                if iteration < start_paper_text_at:
                    emit.status_public_progress(
                        f"[SKIPPED — starts at iteration {start_paper_text_at}]",
                    )
                else:
                    _papers = self.get_paper_texts()
                    previous_paper = _papers[-1].paper_text if _papers else ""

                    try:
                        paper_text_result = await substeps["gen_paper_text"].execute(
                            ctx=ctx,
                            iteration=iteration,
                            output_dir=iter_dir,
                            previous_paper_text=previous_paper or None,
                            reviewer_feedback_text=reviewer_feedback_text,
                            parent_id=iter_id,
                        )
                    except Exception as e:
                        emit.status_public_error(
                            f"   gen_paper_text failed: {e}",
                        )
                        paper_text_result = None

                    if min_token_validity:
                        ensure_oauth_token_fresh(min_token_validity)

                # =================================================================
                # STEP 5: REVIEW_PAPER
                # =================================================================
                emit.status_public_progress("\nSTEP 5: REVIEW_PAPER")
                _papers = self.get_paper_texts()
                _latest_paper = _papers[-1] if _papers else None
                if _latest_paper is None:
                    emit.status_public_progress(
                        "[SKIPPED — no paper draft yet]",
                    )
                elif iteration < review_paper_start_at:
                    emit.status_public_progress(
                        f"[SKIPPED — starts at iteration {review_paper_start_at}]",
                    )
                else:
                    current_paper_text = _latest_paper.paper_text
                    try:
                        review_result = await substeps["review_paper"].execute(
                            ctx=ctx,
                            iteration=iteration,
                            paper_text=current_paper_text,
                            output_dir=iter_dir,
                            previous_critiques_text=reviewer_feedback_text,
                            parent_id=iter_id,
                        )
                        if review_result is not None:
                            ctx.reviewer_feedback = review_result.model_dump()
                            reviewer_feedback_text = format_critiques_for_prompt(review_result)
                            emit.status_public_info(
                                f"Review: score {review_result.score}/10, "
                                f"{len(review_result.critiques)} critiques",
                            )
                    except Exception as e:
                        emit.status_public_error(
                            f"   review_paper failed: {e}",
                        )

                # =================================================================
                # STEP 6: UPD_HYPO
                # =================================================================
                emit.status_public_progress("\nSTEP 6: UPD_HYPO")
                _papers = self.get_paper_texts()
                _latest_paper = _papers[-1] if _papers else None
                if _latest_paper is None:
                    emit.status_public_progress(
                        "[SKIPPED — no paper draft yet]",
                    )
                elif iteration < upd_hypo_start_at:
                    emit.status_public_progress(
                        f"[SKIPPED — starts at iteration {upd_hypo_start_at}]",
                    )
                else:
                    current_paper_text = _latest_paper.paper_text
                    try:
                        upd_result = await substeps["upd_hypo"].execute(
                            ctx=ctx,
                            iteration=iteration,
                            paper_text=current_paper_text,
                            output_dir=iter_dir,
                            reviewer_feedback_text=reviewer_feedback_text,
                            parent_id=iter_id,
                        )
                        if upd_result is not None:
                            ctx.hypothesis = upd_result
                            emit.status_public_info(
                                f"Hypothesis revised: {ctx.hypothesis.get('title', 'N/A')}",
                            )
                    except Exception as e:
                        emit.status_public_error(
                            f"   upd_hypo failed: {e}",
                        )

                    if min_token_validity:
                        ensure_oauth_token_fresh(min_token_validity)

                # Log iteration summary
                emit.status_public_info(f"Iteration {iteration} Summary:")
                emit.status_public_info(
                    f"It {iteration} Strategies: {len(strategies)}",
                )
                emit.status_public_info(
                    f"It {iteration} Plans: {len(plans)}",
                )
                emit.status_public_info(
                    f"It {iteration} Artifacts: {len(artifacts)}",
                )
                emit.status_public_info(
                    f"It {iteration} Paper: {'yes' if paper_text_result else 'no'}",
                )
                # Cumulative totals
                emit.status_private_info(
                    f"Total Strategies: {len(self.get_strategies())}",
                )
                emit.status_private_info(
                    f"Total Plans: {len(self.get_plans())}",
                )
                emit.status_private_info(
                    f"Total Artifacts: {len(self.get_artifacts())}",
                )
                all_papers = self.get_paper_texts()
                emit.status_private_info(f"Total Papers: {len(all_papers)}")
                latest_paper = all_papers[-1] if all_papers else None
                if latest_paper:
                    emit.status_private_info(
                        f"Latest paper: {latest_paper.id}",
                    )

                emit.end_iteration(group_id=loop_gid, iteration=iteration)

                iterations_completed += 1

                emit.status_private_info(
                    f"Iteration {iteration} state saved (live Run)",
                )

            # =====================================================================
            # POST-LOOP: Finalize
            # =====================================================================
            all_papers = self.get_paper_texts()
            all_artifacts = self.get_artifacts()
            latest_paper = all_papers[-1] if all_papers else None

            # Build final result — use the overall pipeline's last_step as the
            # invention-loop endpoint label for downstream metadata.
            end_step = config.init.pipeline.last_step
            result = InventionLoopOut(
                output_dir=str(output_dir),
                pools_dir=str(output_dir / "pools"),
                artifacts=all_artifacts,
                hypothesis=ctx.hypothesis,
                metadata={
                    "generated_at": datetime.now(UTC).isoformat(),
                    "module": "invention_loop",
                    "iterations_completed": iterations_completed,
                    "max_iterations": max_iterations,
                    "start_iteration": 1,
                    "end_step": end_step,
                    "total_strategies": len(self.get_strategies()),
                    "total_plans": len(self.get_plans()),
                    "total_artifacts": len(all_artifacts),
                    "total_paper_texts": len(all_papers),
                },
            )

            emit.status_public_success("Invention loop completed:")
            emit.status_public_info(f"Iterations: {iterations_completed}")
            emit.status_public_info(
                f"Artifacts: {len(self.get_artifacts())}",
            )
            emit.status_private_info(
                f"Paper texts: {len(self.get_paper_texts())}",
            )
            if latest_paper:
                emit.status_private_info(f"Latest paper: {latest_paper.id}")

            emit.end_group(loop_gid)

            return result


# Substep dispatch happens via the typed Module subclasses pre-instantiated
# under each iter by the scaffold (``GenStratModule`` / ``GenPlanModule`` /
# ``GenArtModule`` / ``GenPaperTextModule`` / ``ReviewPaperModule`` /
# ``UpdHypoModule``); each carries an ``execute()`` that delegates to the
# in-file ``run_*_module`` body. The orchestrator below builds a
# ``{name: substep_module}`` lookup per iter and dispatches via
# ``await substeps[<name>].execute(...)``.


@dataclass
class LoopCtx(ModuleCtx):
    """Context for the invention loop's per-iteration substep helpers.

    Inherits config/output_dir from ModuleCtx; adds invention-loop
    specific state. The actual strategies/plans/artifacts/paper_texts
    are read on demand via :attr:`invention_loop_group` accessors.

    Pre-Stage-7 this carried ``start_iteration`` / ``should_skip_step``
    / ``should_stop_at_step`` / ``resumed_from`` / ``resume_start_step``
    fields for resume skip-ahead. v27 replay-execute (Stages 1-9)
    made those unnecessary; Stage 10 deletion removed them.
    """

    # Required loop inputs (no defaults — must be passed)
    hypothesis: dict = None  # type: ignore[assignment]
    invention_loop_group: InventionLoopGroup = None  # type: ignore[assignment]
    max_iterations: int = 1
    start_paper_text_at: int = 1
    min_token_validity: int = 0
    run_dir: Path | None = None
    reviewer_feedback: dict | None = None
    revised_hypothesis: dict | None = None
    user_uploads_path: str = ""


def setup_loop(
    config: PipelineConfig,
    hypothesis: dict,
    run_dir: Path | None,
    user_uploads_path: str = "",
) -> LoopCtx:
    """Parse config, ensure the InventionLoopGroup exists.

    Strategies/plans/artifacts/paper_texts are reconstituted from the
    live Run's tree of ``module_output`` events on demand via the
    group's ``get_*`` accessors — no on-disk JSON sidecar files. Resume
    is handled uniformly by replay-execute (v27 Stages 1-9); this
    helper no longer accepts ``start_iter`` / ``start_substep``.
    """
    if run_dir is None:
        raise ValueError("invention_loop requires run_dir")
    output_dir = run_dir / "3_invention_loop"
    output_dir.mkdir(parents=True, exist_ok=True)

    invention_cfg = config.invention_loop
    max_iterations = invention_cfg.max_iterations
    start_paper_text_at = invention_cfg.gen_paper_text.start_at_iteration

    _auth_cfg = (
        config.raw.get("agent_backend", {})
        .get("claude_agent_sdk", {})
        .get("llm_backend", {})
        .get("claude_max", {})
        .get("auth", {})
    )
    min_token_validity = _auth_cfg.get(
        "min_token_validity_seconds", DEFAULT_MIN_TOKEN_VALIDITY_SECONDS
    )

    # The orchestrator MUST have already called
    # ``emit.start_loop_group(name="invention_loop")`` before
    # invoking setup_loop — we just retrieve the live group here.
    run = current_run()
    group = run.find_group_by_name("invention_loop")
    if not isinstance(group, InventionLoopGroup):
        # Defensive: free-standing fallback so setup doesn't crash.
        group = InventionLoopGroup(name="invention_loop")

    return LoopCtx(
        config=config,
        hypothesis=hypothesis,
        output_dir=output_dir,
        invention_loop_group=group,
        max_iterations=max_iterations,
        start_paper_text_at=start_paper_text_at,
        min_token_validity=min_token_validity,
        run_dir=run_dir,
        reviewer_feedback=None,
        revised_hypothesis=None,
        user_uploads_path=user_uploads_path,
    )
