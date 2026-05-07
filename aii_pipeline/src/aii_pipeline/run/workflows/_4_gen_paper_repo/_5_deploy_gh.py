"""``deploy_gh`` as a DBOS child workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.steps._4_gen_paper_repo._5_deploy_gh import DeployGhModule


class DeployGhWorkflowInput(BaseModel):
    """JSON-safe input for ``deploy_gh_workflow``."""

    parent_id: str
    repo_url: str
    repo_name: str
    repo_description: str = ""
    output_dir: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    prepared_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    paper_pdf_path: str | None = None
    paper_latex_dir: str | None = None
    deploy_cfg: dict[str, Any] = Field(default_factory=dict)
    """Subset of pipeline config governing this deploy (passed by caller)."""
    github_cfg: dict[str, Any] = Field(default_factory=dict)
    """Github-specific config block."""
    max_file_size_mb: int | None = None
    """Bytes limit for repo pushes (taken from PipelineConfig.max_file_size_mb)."""
    config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def deploy_gh_workflow(wf_input: DeployGhWorkflowInput) -> list[dict[str, Any]]:
    """Clone the resolved repo, push src + demos + paper, return deployment info.

    ``deploy_cfg`` and ``github_cfg`` are passed as dicts (mirror the
    legacy execute's typed kwargs) — caller serialises the relevant
    config slices. The return value is the JSON-safe list of
    ``GistDeployment`` dicts (caller revives via
    ``TypeAdapter(GistDeployment).validate_python`` if needed).
    """
    artifact_adapter = TypeAdapter(BaseArtifact)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]
    # ``prepared_artifacts`` are typed differently in the legacy code
    # (subclass of BaseDemo); keep as dicts and let the module re-shape.
    prepared = list(wf_input.prepared_artifacts)

    module = DeployGhModule()
    extra_kwargs: dict[str, Any] = {}
    if wf_input.max_file_size_mb is not None:
        extra_kwargs["max_file_size_mb"] = wf_input.max_file_size_mb
    deployments = await module.execute(
        repo_url=wf_input.repo_url,
        repo_name=wf_input.repo_name,
        repo_description=wf_input.repo_description,
        output_dir=Path(wf_input.output_dir),
        artifacts=artifacts,
        prepared_artifacts=prepared,
        paper_pdf_path=Path(wf_input.paper_pdf_path) if wf_input.paper_pdf_path else None,
        paper_latex_dir=Path(wf_input.paper_latex_dir) if wf_input.paper_latex_dir else None,
        deploy_cfg=wf_input.deploy_cfg,
        github_cfg=wf_input.github_cfg,
        parent_id=wf_input.parent_id,
        **extra_kwargs,
    )
    return [d.model_dump() for d in deployments]
