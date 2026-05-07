"""Shared schemas for Gen Paper module.

Classes used across multiple gen_paper_repo steps live here.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GistDeployment(BaseModel):
    """Information about a deployed artifact in GitHub repo."""

    artifact_id: str = Field(description="ID of the artifact")
    iter: int = Field(default=0, description="Iteration number this artifact was produced in")
    gist_url: str = Field(description="URL to the artifact in GitHub repo")
    gist_id: str = Field(description="GitHub artifact ID")
    files: list[str] = Field(default_factory=list, description="Files deployed")
    colab_url: str | None = Field(default=None, description="Google Colab URL for notebooks")


class DeployGhOut(BaseModel):
    """Aggregate output of deploy_gh module with GitHub deployments.

    One ``GistDeployment`` per artifact pushed to GitHub. Used as the
    typed payload for ``module_output(output=...)``; readers walk
    ``module.output.deployments`` rather than the legacy plural list.
    """

    kind: Literal["deploy_gh_out"] = "deploy_gh_out"
    deployments: list[GistDeployment] = Field(default_factory=list)


class GenRepoOut(BaseModel):
    """Output of gen_repo module with repository information.

    Repo URL + metadata resolved via ``gh CLI``. Used as the typed
    payload for ``module_output(output=...)`` so readers (and resume)
    walk ``module.output`` instead of a free-form dict.
    """

    kind: Literal["gen_repo_out"] = "gen_repo_out"
    repo_url: str = ""
    repo_name: str = ""
    description: str = ""
    created: bool = False
    error: str = ""
