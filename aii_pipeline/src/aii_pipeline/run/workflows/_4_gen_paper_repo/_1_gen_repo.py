"""``gen_repo`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field

from aii_pipeline.steps._4_gen_paper_repo._1_gen_repo import GenRepoModule
from aii_pipeline.utils import PipelineConfig


class GenRepoWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_repo_workflow``."""

    parent_id: str
    hypothesis: dict[str, Any]
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_repo_workflow(wf_input: GenRepoWorkflowInput) -> dict | None:
    """Resolve the GitHub repo URL/name/description from the hypothesis.

    Returns the legacy ``repo_info`` dict (``repo_url``, ``repo_name``,
    ``description``) or None if the module bailed.
    """
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    module = GenRepoModule()
    return await module.execute(
        config=config,
        hypothesis=wf_input.hypothesis,
        output_dir=Path(wf_input.output_dir),
        parent_id=wf_input.parent_id,
    )
