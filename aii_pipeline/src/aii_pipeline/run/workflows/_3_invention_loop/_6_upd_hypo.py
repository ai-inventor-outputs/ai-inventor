"""``upd_hypo`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.run.workflows._3_invention_loop._loop_group_stub import _LoopGroupStub
from aii_pipeline.steps._3_invention_loop._6_upd_hypo import UpdHypoModule
from aii_pipeline.steps._3_invention_loop.invention_loop import LoopCtx
from aii_pipeline.utils import PipelineConfig


class UpdHypoWorkflowInput(BaseModel):
    """JSON-safe input for ``upd_hypo_workflow``."""

    iteration: int
    parent_id: str
    hypothesis: dict[str, Any]
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    paper_text: str
    reviewer_feedback_text: str | None = None
    user_uploads_path: str
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def upd_hypo_workflow(
    wf_input: UpdHypoWorkflowInput,
) -> dict[str, Any] | None:
    """Internal hypothesis revision based on the iteration's paper + review."""
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    artifact_adapter = TypeAdapter(BaseArtifact)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]

    ctx = LoopCtx(
        config=config,
        output_dir=Path(wf_input.output_dir),
        hypothesis=wf_input.hypothesis,
        invention_loop_group=_LoopGroupStub(artifacts=artifacts),  # type: ignore[arg-type]
        user_uploads_path=wf_input.user_uploads_path,
        max_iterations=config.invention_loop.max_iterations,
        run_dir=Path(wf_input.output_dir).parent.parent,
    )

    module = UpdHypoModule()
    result = await module.execute(
        ctx=ctx,
        iteration=wf_input.iteration,
        paper_text=wf_input.paper_text,
        output_dir=Path(wf_input.output_dir),
        reviewer_feedback_text=wf_input.reviewer_feedback_text,
        parent_id=wf_input.parent_id,
    )
    # upd_hypo returns dict | None directly (legacy contract)
    return result if result is not None else None
