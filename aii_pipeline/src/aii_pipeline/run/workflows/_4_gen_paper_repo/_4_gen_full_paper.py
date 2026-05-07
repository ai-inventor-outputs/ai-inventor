"""``gen_full_paper`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field

from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
    PaperText,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import (
    Figure,
)
from aii_pipeline.steps._4_gen_paper_repo._4_gen_full_paper import (
    GenFullPaperModule,
)
from aii_pipeline.utils import PipelineConfig


class GenFullPaperWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_full_paper_workflow``."""

    parent_id: str
    paper: dict[str, Any] | None = None
    """Latest ``PaperText`` dict from invention_loop's last gen_paper_text iter."""
    figures: list[dict[str, Any]] = Field(default_factory=list)
    gist_deployments: list[dict[str, Any]] | None = None
    output_dir: str
    repo_url: str | None = None
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_full_paper_workflow(
    wf_input: GenFullPaperWorkflowInput,
) -> dict[str, Any]:
    """Compile the full LaTeX/PDF paper from the latest paper text + figures."""
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    paper_obj = PaperText.model_validate(wf_input.paper) if wf_input.paper is not None else None
    figures = [Figure.model_validate(d) for d in wf_input.figures]
    # GistDeployment: deserialised lazily — its location moves around in
    # current code. Pass through as raw dicts; legacy execute can still
    # accept them since it iterates and reads attribute-style fields.
    gist_deployments = wf_input.gist_deployments

    module = GenFullPaperModule()
    out = await module.execute(
        config=config,
        paper=paper_obj,
        figures=figures,
        gist_deployments=gist_deployments,
        output_dir=Path(wf_input.output_dir),
        repo_url=wf_input.repo_url,
        parent_id=wf_input.parent_id,
    )
    # GenPaperRepoOut is Pydantic
    return out.model_dump() if hasattr(out, "model_dump") else dict(out)
