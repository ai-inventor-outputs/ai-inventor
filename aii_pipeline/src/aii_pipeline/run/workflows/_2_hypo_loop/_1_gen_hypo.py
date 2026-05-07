"""``gen_hypo`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field

from aii_pipeline.steps._2_hypo_loop._1_gen_hypo import GenHypoModule
from aii_pipeline.utils import PipelineConfig


class GenHypoWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_hypo_workflow``."""

    iteration: int
    parent_id: str
    agent_prompts: list[list[dict[str, Any]]] = Field(default_factory=list)
    """List of agent-prompt sets carried over from seed_hypo."""
    previous_hypothesis: dict[str, Any] | None = None
    previous_review_feedback: dict[str, Any] | None = None
    user_uploads_path: str = ""
    run_dir: str
    """Where gen_hypo writes its output (``<run_dir>/.../gen_hypo``)."""
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_hypo_workflow(
    wf_input: GenHypoWorkflowInput,
) -> dict[str, Any] | None:
    """Generate hypotheses for one hypo-loop iteration.

    Returns the legacy ``GenHypoOut`` as a dict (caller re-validates).
    Unlike the invention-loop modules, gen_hypo doesn't take a
    ``LoopCtx`` — its execute signature is direct kwargs.
    """
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    module = GenHypoModule()
    result = await module.execute(
        config=config,
        agent_prompts=wf_input.agent_prompts,
        run_dir=Path(wf_input.run_dir),
        previous_hypothesis=wf_input.previous_hypothesis,
        previous_review_feedback=wf_input.previous_review_feedback,
        iteration=wf_input.iteration,
        user_uploads_path=wf_input.user_uploads_path,
        parent_id=wf_input.parent_id,
    )
    if result is None:
        return None
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)
