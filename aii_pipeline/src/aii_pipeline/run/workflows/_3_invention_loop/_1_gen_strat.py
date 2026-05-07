"""``gen_strat`` as a DBOS child workflow.

Wraps :class:`GenStratModule.execute` in a ``@DBOS.workflow``-decorated
free function with JSON-safe input + output. Caller builds a :class:`GenStratWorkflowInput`,
invokes this workflow, and re-validates the returned dicts as
:class:`Strategy` instances.

Implementation: this is a delegating wrapper — the workflow body
reconstructs a minimal :class:`LoopCtx` (with a stand-in
``invention_loop_group`` providing only ``get_artifacts``, the single
accessor ``GenStratModule.execute`` reads from it) and calls into the
legacy class method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.run.workflows._3_invention_loop._loop_group_stub import _LoopGroupStub
from aii_pipeline.steps._3_invention_loop._1_gen_strat import GenStratModule
from aii_pipeline.steps._3_invention_loop.invention_loop import LoopCtx
from aii_pipeline.utils import PipelineConfig


class GenStratWorkflowInput(BaseModel):
    """JSON-safe input to ``gen_strat_workflow``."""

    iteration: int
    """1-indexed iteration number within the invention loop."""

    parent_id: str
    """Node id of the owning iteration. Deterministic via path-derived
    id."""

    hypothesis: dict[str, Any]
    """The current hypothesis being investigated. Carried as a dict
    so the workflow can re-validate via ``Hypothesis.model_validate``
    if needed."""

    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    """List of existing artifacts in the invention pool, each as a
    serialised ``BaseArtifact`` dict. Empty for the first iteration."""

    user_uploads_path: str
    """Absolute path to the run's ``user_uploads/`` directory (string
    so the input remains JSON-serialisable)."""

    output_dir: str
    """Absolute path to the iteration's output directory (e.g.
    ``<run_dir>/3_invention_loop/iter_<n>/``). The legacy execute
    creates a ``gen_strat/`` subdir inside it."""

    config_dirs: list[str] = Field(default_factory=list)
    """Optional layered ``aii_config/pipeline/`` dirs to load on top
    of the canonical config (mirrors the cli ``--config-dir``
    flags). Empty for default canonical-only config."""

    previous_strategies: list[dict[str, Any]] | None = None
    """Strategies from the previous iteration (serialised
    ``Strategy`` dicts). ``None`` for the first iteration."""

    reviewer_feedback_text: str | None = None
    """Reviewer feedback text from the prior iteration, if any."""

    paper_text: str | None = None
    """Paper text from the prior iteration, if any."""


@DBOS.workflow()
async def gen_strat_workflow(wf_input: GenStratWorkflowInput) -> list[dict[str, Any]]:
    """Generate research strategies for one invention-loop iteration.

    JSON-safe child workflow:

    Args:
        input: :class:`GenStratWorkflowInput` — iteration index, parent
            iteration node id, hypothesis, artifact pool (as dicts),
            iteration-history fields.

    Returns:
        A list of serialised :class:`Strategy` dicts (callers re-validate
        via ``Strategy.model_validate(d)``).

    Determinism: workflow body reconstructs config + ctx deterministically
    (file reads, no time/random). The expensive LLM work happens inside
    ``Agent.run`` calls already wrapped as :func:`@DBOS.step`.
    """
    config = PipelineConfig.from_yaml(
        *[Path(d) for d in wf_input.config_dirs],
    )

    # Re-validate artifact dicts back into the typed discriminated-union
    # tree. ``bind_pipeline_typed_unions`` (called at pipeline boot)
    # widens ``BaseArtifact`` to a union over every concrete artifact
    # subclass; the type adapter then dispatches via the ``kind``
    # discriminator. Empty list → no-op.
    artifact_adapter = TypeAdapter(BaseArtifact)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]

    # Build the minimum-viable LoopCtx. ``gen_strat.execute`` reads:
    #   ctx.config / ctx.hypothesis / ctx.invention_loop_group.get_artifacts()
    #   ctx.user_uploads_path
    # plus the ModuleCtx base's config + output_dir. Everything else on
    # LoopCtx (max_iterations / start_paper_text_at / run_dir / reviewer_feedback / …)
    # is read by sibling modules, not gen_strat — leaving them at the
    # dataclass defaults is fine for this entry point.
    ctx = LoopCtx(
        config=config,
        output_dir=Path(wf_input.output_dir),
        hypothesis=wf_input.hypothesis,
        invention_loop_group=_LoopGroupStub(artifacts=artifacts),  # type: ignore[arg-type]
        user_uploads_path=wf_input.user_uploads_path,
        max_iterations=config.invention_loop.max_iterations,
        run_dir=Path(wf_input.output_dir).parent.parent,
    )

    module = GenStratModule()
    strategies = await module.execute(
        ctx=ctx,
        iteration=wf_input.iteration,
        output_dir=Path(wf_input.output_dir),
        previous_strategies=wf_input.previous_strategies,
        reviewer_feedback_text=wf_input.reviewer_feedback_text,
        paper_text=wf_input.paper_text,
        parent_id=wf_input.parent_id,
    )

    # Strategy is a Pydantic model; model_dump produces JSON-safe dicts.
    return [s.model_dump() for s in (strategies or [])]
