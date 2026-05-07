"""Gen paper repo config models."""

from pydantic import BaseModel, ConfigDict, Field

from .invention_loop import ClaudeAgentConfig, ClaudeAgentConfigNoConc


class GitHubIdentityConfig(BaseModel):
    """Non-secret GitHub identity for paper-repo creation + commits.

    The token (``AII_GH_TOKEN``) lives in ``.env``. The repo OWNER is
    auto-resolved from that token via ``gh api user``, so no username
    here. ``commit_author_*`` are written per-clone via ``git config``
    (never ``--global``) so they don't pollute the developer's identity.
    """

    commit_author_name: str = "AI Inventor"
    commit_author_email: str = "ai-inventor@noreply"
    repo_prefix: str = "ai-invention"
    model_config = ConfigDict(extra="allow")


class GenRepoConfig(BaseModel):
    """Repository generation configuration."""

    enabled: bool = True
    model_config = ConfigDict(extra="allow")


class DeployGHConfig(BaseModel):
    """GitHub deployment configuration."""

    enabled: bool = True
    chunk_max_mb: int = 1000  # Max MB per git push chunk
    push_timeout: int = 1200  # Seconds per push attempt
    min_push_interval: int = 2  # Seconds between consecutive pushes
    model_config = ConfigDict(extra="allow")


class VerifyVizConfig(BaseModel):
    """Visualization verification configuration."""

    max_retries: int = 2
    model_config = ConfigDict(extra="allow")


class FreeVizModelEntry(BaseModel):
    """Free visualization model entry configuration."""

    model: str
    llm_timeout: int = 120
    model_config = ConfigDict(extra="allow")


class FreeVizConfig(BaseModel):
    """Free visualization configuration."""

    client: str = "openrouter"
    max_concurrent: int = 10
    image_size: str | None = "2K"
    models: list[FreeVizModelEntry] = Field(
        default_factory=lambda: [
            FreeVizModelEntry(model="google/gemini-3-pro-image-preview", llm_timeout=240)
        ]
    )
    model_config = ConfigDict(extra="allow")


class VizGenConfig(BaseModel):
    """Visualization generation configuration."""

    use_claude_agent: bool = True
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    free_viz: FreeVizConfig = Field(default_factory=FreeVizConfig)
    verify_viz: VerifyVizConfig = Field(default_factory=VerifyVizConfig)
    model_config = ConfigDict(extra="allow")


class GenArtDemoConfig(BaseModel):
    """Artifact demo generation configuration."""

    enabled: bool = True
    max_notebook_total_runtime: int = 600
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    model_config = ConfigDict(extra="allow")


class GenFullPaperConfig(BaseModel):
    """Full paper generation configuration."""

    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    model_config = ConfigDict(extra="allow")


class GenPaperConfig(BaseModel):
    """Gen paper repo module configuration."""

    github: GitHubIdentityConfig = Field(default_factory=GitHubIdentityConfig)
    gen_repo: GenRepoConfig = Field(default_factory=GenRepoConfig)
    gen_art_demo: GenArtDemoConfig = Field(default_factory=GenArtDemoConfig)
    viz_gen: VizGenConfig = Field(default_factory=VizGenConfig)
    gen_full_paper: GenFullPaperConfig = Field(default_factory=GenFullPaperConfig)
    deploy_gh: DeployGHConfig = Field(default_factory=DeployGHConfig)
    model_config = ConfigDict(extra="allow")
