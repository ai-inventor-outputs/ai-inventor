"""HYPO_LOOP — Iterative hypothesis generation with adversarial review.

Flow per iteration:
    1. gen_hypo  — generate hypotheses (with previous hypo + review feedback)
    2. review_hypo — adversarial review (feedback feeds into next gen_hypo)

Iteration 1: gen_hypo runs without feedback.
Iteration 2+: gen_hypo receives previous hypothesis + review feedback.

State lives on the run tree — every gen_hypo / review_hypo emits a
``module_output`` event whose ``outputs`` carries the typed result.
``HypoLoopGroup.get_hypotheses(...)`` / ``get_hypo_reviews(...)`` walk
the tree on demand. No more ledgers/*.json sidecar files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token_fresh
from aii_lib.run import current_run, emit
from aii_lib.run.context import ctx_scope, current_ctx
from aii_lib.run.mdgroup import LoopMdGroup

from aii_pipeline.pipeline import PipelineFailure
from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.out_schema import GenHypoOut
from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.out_schema import (
    ReviewHypoOut,
)
from aii_pipeline.prompts.steps._2_hypo_loop.out_schema import HypoLoopOut
from aii_pipeline.run.utils.group_helpers import _module_output_for
from aii_pipeline.utils import DEFAULT_MIN_TOKEN_VALIDITY_SECONDS, PipelineConfig

if TYPE_CHECKING:
    from aii_lib.run.loop_iteration import LoopIteration

    from aii_pipeline.steps.base import StepContext


@dataclass
class HypoLoopCtx:
    """Phase ctx for ``hypo_loop``.

    Carries the upstream agent_prompts produced by ``seed_hypo``.
    Pre-Stage-7 this also threaded ``start_iter`` / ``start_substep``
    for resume skip-ahead; v27 replay-execute (Stages 1-9) made those
    fields unnecessary, removed in Stage 10.
    """

    config: PipelineConfig
    run_dir: Path
    user_uploads_path: str
    agent_prompts: list = field(default_factory=list)


class HypoLoopGroup(LoopMdGroup):
    """Phase 2 — hypo_loop. Iterations of gen_hypo + review_hypo."""

    kind: Literal["hypo_loop_group"] = "hypo_loop_group"
    """Per-subclass discriminator. Pydantic's ``AnyMdGroup`` union (rebound
    in ``aii_pipeline/run/__init__.py`` to include phase classes) picks
    this value to deserialize ``HypoLoopGroup`` instances; without a
    unique tag, model_validate would collapse to the parent
    ``LoopMdGroup`` and lose ``execute()``."""

    def _iters(self, iteration: int | None) -> list[LoopIteration]:
        if iteration is None:
            return self.children
        it = self.find_iteration(iteration)
        return [it] if it else []

    def get_hypotheses(self, iteration: int | None = None) -> list:
        """All hypotheses produced by gen_hypo, optionally filtered to one iter.

        Reads from each iteration's ``gen_hypo`` Module's typed
        ``output: GenHypoOut`` slot (set by ``module_output``
        dispatch); ``GenHypoOut.hypotheses`` is the per-task list of
        hypothesis dicts.
        """
        out: list = []
        for it in self._iters(iteration):
            gen_hypo_out = _module_output_for(it.children, "gen_hypo")
            if gen_hypo_out is not None and getattr(gen_hypo_out, "hypotheses", None):
                out.extend(gen_hypo_out.hypotheses)
        return out

    def get_hypo_reviews(self, iteration: int | None = None) -> list:
        """All review_hypo outputs, optionally filtered to one iter.

        Reads ``Module.output`` (typed :class:`ReviewHypoOut`) for each
        iteration's ``review_hypo`` module — single output per module,
        accumulated into a list across iterations.
        """
        out: list = []
        for it in self._iters(iteration):
            r = _module_output_for(it.children, "review_hypo")
            if isinstance(r, ReviewHypoOut):
                out.append(r)
        return out

    def synthesize_result_from_tree(self) -> HypoLoopOut:
        """Reconstruct HypoLoopOut from the live tree's module outputs.

        Used as the in-process fallback path when :attr:`MdGroup.output`
        is not yet populated (live execution before the ``mdgroup_output``
        event fires) — also keeps callers stable across the migration.

        ``get_hypotheses()`` returns typed ``Hypothesis`` objects;
        :class:`GenHypoOut` expects ``list[dict]`` (the wire format
        downstream consumers see in normal flow), so dump them.
        Falls back to ``ReviewHypoOut(hypothesis=last_hypothesis)``
        when no review fired (matches ``execute`` 's own fallback).
        """
        hypos_typed = self.get_hypotheses()
        hypos = [h.model_dump() if hasattr(h, "model_dump") else h for h in hypos_typed]
        reviews = self.get_hypo_reviews()
        last_review = reviews[-1] if reviews else None
        if last_review is None and hypos:
            last_review = ReviewHypoOut(hypothesis=hypos[-1])
        return HypoLoopOut(
            gen_hypo=GenHypoOut(hypotheses=hypos),
            review_hypo=last_review,
            iterations_completed=len(self.children),
        )

    def get_context(self) -> HypoLoopCtx:
        parent: StepContext = current_ctx()
        seed_group = current_run().find_group_by_name("seed_hypo")
        seed_result = seed_group.output if seed_group is not None else None
        agent_prompts = (
            seed_result.agent_prompts
            if seed_result is not None and hasattr(seed_result, "agent_prompts")
            else []
        )
        return HypoLoopCtx(
            config=parent.config,
            run_dir=parent.run_dir,
            user_uploads_path=str(parent.run_dir / "user_uploads"),
            agent_prompts=agent_prompts,
        )

    async def execute(self) -> Any:
        """Run the hypothesis generation + review loop.

        Substep dispatch reads the typed Module subclasses (``GenHypoModule``
        / ``ReviewHypoModule``) the scaffold pre-instantiated under each
        iter and calls ``await module.execute(...)``. Resume is
        handled uniformly by replay-execute (Stages 1-9 of v27): the
        loop body is identical for fresh and resumed runs.
        """
        with ctx_scope(self.get_context()) as ctx:
            config = ctx.config
            agent_prompts = ctx.agent_prompts
            run_dir = ctx.run_dir
            user_uploads_path = ctx.user_uploads_path

            loop_cfg = config.gen_hypo_loop
            max_iterations = loop_cfg.max_iterations
            review_enabled = config.review_hypo.enabled

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

            loop_gid = emit.start_loop_group(name="hypo_loop")
            # Scaffold pre-created `self`; start_loop_group reuses our id.
            assert loop_gid == self.node_id, (
                f"hypo_loop scaffold drift: gid={loop_gid} self.node_id={self.node_id}"
            )

            emit.status_public_progress(
                f"Iterations: 1..{max_iterations}",
            )
            emit.status_private_info(f"Review enabled: {review_enabled}")

            # Track the most recent hypothesis/review across iterations —
            # derived from the run tree on each loop pass so resume / fork
            # restarts seed correctly without relying on Python locals
            # surviving the process boundary.
            previous_hypothesis: dict | None = None
            previous_review_feedback: dict | None = None

            all_hypos = self.get_hypotheses()
            all_reviews = self.get_hypo_reviews()
            if all_hypos:
                latest = all_hypos[-1]
                previous_hypothesis = (
                    latest.model_dump() if hasattr(latest, "model_dump") else dict(latest)
                )
            if all_reviews:
                latest_r = all_reviews[-1]
                previous_review_feedback = (
                    latest_r.final_review
                    if hasattr(latest_r, "final_review")
                    else (latest_r.model_dump() if hasattr(latest_r, "model_dump") else None)
                )

            results: dict[str, Any] = {}
            current_hypo: dict = previous_hypothesis or {}

            for iteration in range(1, max_iterations + 1):
                is_last = iteration == max_iterations
                iter_id = emit.start_iteration(
                    group_id=loop_gid,
                    iteration=iteration,
                )
                iter_node = current_run().find_node(iter_id)
                substeps = {m.name: m for m in iter_node.children}
                emit.status_public_progress(
                    f"--- Hypo loop iteration {iteration}/{max_iterations} ---",
                )

                iter_dir = run_dir / f"iter_{iteration}" if run_dir else None
                if iter_dir:
                    iter_dir.mkdir(parents=True, exist_ok=True)

                # Snapshot the prior-iteration hypothesis BEFORE gen_hypo
                # overwrites ``previous_hypothesis``. review_hypo uses this
                # to classify the H↔H Moulines edge from iter N-1's
                # hypothesis to iter N's.
                prior_iter_hypothesis = previous_hypothesis

                # --- GEN_HYPO ---
                gen_hypo_dir = iter_dir / "gen_hypo" if iter_dir else None
                if gen_hypo_dir:
                    gen_hypo_dir.mkdir(parents=True, exist_ok=True)

                gen_result = await substeps["gen_hypo"].execute(
                    config=config,
                    agent_prompts=agent_prompts,
                    run_dir=gen_hypo_dir or run_dir,
                    previous_hypothesis=previous_hypothesis,
                    previous_review_feedback=previous_review_feedback,
                    iteration=iteration,
                    user_uploads_path=user_uploads_path,
                    parent_id=iter_id,
                )
                if not gen_result:
                    # ``status_public_error`` is replay-skipped during
                    # fork/resume boot, so a bare emit silently drops
                    # and the caller sees only "Phase 'hypo_loop' returned
                    # falsy result" with no clue why. Raise so
                    # cli.py logs the actual reason inline.
                    msg = f"gen_hypo iter {iteration} returned falsy (no GenHypoOut)"
                    emit.status_public_error(msg)
                    raise PipelineFailure(msg)
                results["gen_hypo"] = gen_result

                hypotheses = gen_result.hypotheses if gen_result else []
                # Fail fast at the gen_hypo boundary on 0 hypotheses
                # rather than letting an empty placeholder dict silently
                # propagate downstream (current_hypo = {} → review skipped
                # → invention_loop crashes with the unhelpful "requires a
                # hypothesis from hypo_loop" message). Surfaces the
                # actual cause: model fallback chain exhausted, all
                # retries returned structured_output=None, etc.
                if not hypotheses:
                    msg = (
                        f"gen_hypo emitted 0 hypotheses (iter {iteration}); "
                        "all retries exhausted. Check the gen_hypo agent's "
                        "fallback model chain and structured-output schema."
                    )
                    emit.status_public_error(msg)
                    raise PipelineFailure(msg)
                current_hypo = hypotheses[0]
                previous_hypothesis = current_hypo

                # --- REVIEW_HYPO ---
                if review_enabled and current_hypo:
                    review_dir = iter_dir / "review_hypo" if iter_dir else None
                    if review_dir:
                        review_dir.mkdir(parents=True, exist_ok=True)

                    from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.out_schema import (
                        format_critiques_for_prompt,
                    )

                    prev_critiques_text = (
                        format_critiques_for_prompt(previous_review_feedback)
                        if previous_review_feedback
                        else None
                    )

                    review_result = await substeps["review_hypo"].execute(
                        config=config,
                        hypothesis=current_hypo,
                        iteration=iteration,
                        output_dir=review_dir,
                        previous_feedback_text=prev_critiques_text,
                        previous_hypothesis=prior_iter_hypothesis,
                        user_uploads_path=user_uploads_path,
                        parent_id=iter_id,
                    )
                    if review_result and review_result.final_review:
                        previous_review_feedback = review_result.final_review
                        results["review_hypo"] = review_result

                        if not is_last:
                            emit.status_public_progress(
                                "Review feedback collected for next iteration",
                            )

                # Close iteration
                emit.end_iteration(group_id=loop_gid, iteration=iteration)

                if not is_last and min_token_validity:
                    ensure_oauth_token_fresh(min_token_validity)

            # No more iteration_ledger — surface raw counts instead.
            iterations_completed = max_iterations

            if "review_hypo" not in results:
                results["review_hypo"] = ReviewHypoOut(hypothesis=current_hypo)

            out = HypoLoopOut(
                gen_hypo=results.get("gen_hypo"),
                review_hypo=results.get("review_hypo"),
                iterations_completed=iterations_completed,
            )

            emit.status_public_success(
                f"Hypo loop completed ({out.iterations_completed} iterations)",
            )
            emit.end_group(loop_gid)
            return out
