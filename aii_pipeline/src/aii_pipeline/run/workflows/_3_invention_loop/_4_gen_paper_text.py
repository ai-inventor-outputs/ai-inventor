"""``gen_paper_text`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.run.workflows._3_invention_loop._loop_group_stub import _LoopGroupStub
from aii_pipeline.steps._3_invention_loop._4_gen_paper_text import (
    GenPaperTextModule,
)
from aii_pipeline.steps._3_invention_loop.invention_loop import LoopCtx
from aii_pipeline.utils import PipelineConfig


class GenPaperTextWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_paper_text_workflow``."""

    iteration: int
    parent_id: str
    hypothesis: dict[str, Any]
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    user_uploads_path: str
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)
    previous_paper_text: str | None = None
    reviewer_feedback_text: str | None = None


@DBOS.workflow()
async def gen_paper_text_workflow(
    wf_input: GenPaperTextWorkflowInput,
) -> dict[str, Any] | None:
    """Generate the iteration's paper draft."""
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

    module = GenPaperTextModule()
    result = await module.execute(
        ctx=ctx,
        iteration=wf_input.iteration,
        output_dir=Path(wf_input.output_dir),
        previous_paper_text=wf_input.previous_paper_text,
        reviewer_feedback_text=wf_input.reviewer_feedback_text,
        parent_id=wf_input.parent_id,
    )
    return result.model_dump() if result is not None else None
