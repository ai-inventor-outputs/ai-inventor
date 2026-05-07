"""``gen_art_demo`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.steps._4_gen_paper_repo._3_gen_art_demo import GenArtDemoModule
from aii_pipeline.utils import PipelineConfig


class GenArtDemoWorkflowInput(BaseModel):
    """JSON-safe input for ``gen_art_demo_workflow``."""

    parent_module_id: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    artifact_workspaces: dict[str, str] = Field(default_factory=dict)
    """Map artifact id → workspace dir path (str)."""
    repo_url: str | None = None
    output_dir: str
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def gen_art_demo_workflow(
    wf_input: GenArtDemoWorkflowInput,
) -> list[dict[str, Any]]:
    """Generate per-artifact demo notebooks for the paper repo."""
    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.config_dirs])
    artifact_adapter = TypeAdapter(BaseArtifact)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]
    workspaces = {k: Path(v) for k, v in wf_input.artifact_workspaces.items()}

    module = GenArtDemoModule()
    demos = await module.execute(
        config=config,
        artifacts=artifacts,
        output_dir=Path(wf_input.output_dir),
        artifact_workspaces=workspaces,
        repo_url=wf_input.repo_url,
        parent_module_id=wf_input.parent_module_id,
    )
    return [d.model_dump() for d in (demos or [])]
