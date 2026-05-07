"""``review_hypo`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field

from aii_pipeline.steps._2_hypo_loop._2_review_hypo import ReviewHypoModule
from aii_pipeline.utils import PipelineConfig


class ReviewHypoWorkflowInput(BaseModel):
    """JSON-safe input for ``review_hypo_workflow``."""

    iteration: int
    parent_id: str
    hypothesis: dict[str, Any]
    previous_feedback_text: str | None = None
    previous_hypothesis: dict[str, Any] | None = None
    user_uploads_path: str = ""
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def review_hypo_workflow(
    wf_input: ReviewHypoWorkflowInput,
) -> dict[str, Any] | None:
    """Review a generated hypothesis. Returns serialised ``ReviewHypoOut``."""
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    module = ReviewHypoModule()
    result = await module.execute(
        config=config,
        hypothesis=wf_input.hypothesis,
        iteration=wf_input.iteration,
        output_dir=Path(wf_input.output_dir),
        previous_feedback_text=wf_input.previous_feedback_text,
        previous_hypothesis=wf_input.previous_hypothesis,
        user_uploads_path=wf_input.user_uploads_path,
        parent_id=wf_input.parent_id,
    )
    if result is None:
        return None
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)
