"""``gen_art`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
    BasePlan,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.run.workflows._3_invention_loop._loop_group_stub import _LoopGroupStub
from aii_pipeline.steps._3_invention_loop._3_gen_art import GenArtModule
from aii_pipeline.steps._3_invention_loop.invention_loop import LoopCtx
from aii_pipeline.utils import PipelineConfig


class GenArtWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_art_workflow``."""

    iteration: int
    parent_id: str
    hypothesis: dict[str, Any]
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    plans: list[dict[str, Any]] = Field(default_factory=list)
    user_uploads_path: str
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_art_workflow(wf_input: GenArtWorkflowInput) -> list[dict[str, Any]]:
    """Execute plans and produce artifacts for one invention-loop iteration.

    Returns the FULL artifact pool after this iteration (legacy execute
    returns the cumulative pool, not just the new artifacts), as
    serialised :class:`BaseArtifact` dicts.
    """
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    artifact_adapter = TypeAdapter(BaseArtifact)
    plan_adapter = TypeAdapter(BasePlan)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]
    plans = [plan_adapter.validate_python(d) for d in wf_input.plans]

    ctx = LoopCtx(
        config=config,
        output_dir=Path(wf_input.output_dir),
        hypothesis=wf_input.hypothesis,
        invention_loop_group=_LoopGroupStub(  # type: ignore[arg-type]
            artifacts=artifacts,
            plans=plans,
        ),
        user_uploads_path=wf_input.user_uploads_path,
        max_iterations=config.invention_loop.max_iterations,
        run_dir=Path(wf_input.output_dir).parent.parent,
    )

    module = GenArtModule()
    out = await module.execute(
        ctx=ctx,
        iteration=wf_input.iteration,
        output_dir=Path(wf_input.output_dir),
        plans=plans,
        parent_id=wf_input.parent_id,
    )
    return [a.model_dump() for a in (out or [])]
