"""``gen_viz`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field

from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import (
    Figure,
)
from aii_pipeline.steps._4_gen_paper_repo._2_gen_viz import GenVizModule
from aii_pipeline.utils import PipelineConfig


class GenVizWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_viz_workflow``."""

    parent_module_id: str
    figures: list[dict[str, Any]] = Field(default_factory=list)
    """Figures from the paper, as serialised :class:`Figure` dicts."""
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_viz_workflow(wf_input: GenVizWorkflowInput) -> list[dict[str, Any]]:
    """Generate paper figures (image rendering)."""
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    figures = [Figure.model_validate(d) for d in wf_input.figures]

    module = GenVizModule()
    out_figures = await module.execute(
        config=config,
        figures=figures,
        output_dir=Path(wf_input.output_dir),
        parent_module_id=wf_input.parent_module_id,
    )
    return [f.model_dump() for f in (out_figures or [])]
