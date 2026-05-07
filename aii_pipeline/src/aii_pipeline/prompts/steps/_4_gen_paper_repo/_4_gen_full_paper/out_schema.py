"""Schema for full paper generation step.

Defines:
- FullPaper, FullPaperExpectedFiles: Structured output for LaTeX paper generation
- GenPaperRepoOut: Final output of gen_paper module
"""

from typing import Annotated, Literal

from aii_lib.prompts import (
    BaseExpectedFiles,
    LLMPrompt,
    LLMPromptModel,
    LLMStructOut,
    LLMStructOutModel,
)
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
    PaperText,
)
from aii_pipeline.steps.base import BaseStepOut
from pydantic import Field

from .._2_gen_viz.out_schema import Figure
from ..out_schema import GistDeployment

# =============================================================================
# STRUCTURED OUTPUT (agent output schema)
# =============================================================================


class FullPaperExpectedFiles(BaseExpectedFiles):
    """All expected output files from full paper generation."""

    paper_tex_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to LaTeX source file. Example: 'paper.tex'"
    )
    paper_pdf_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to compiled PDF. Example: 'paper.pdf'"
    )
    references_bib_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to BibTeX bibliography file. Example: 'references.bib'"
    )
    figure_paths: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']"
    )


class FullPaper(LLMPromptModel, LLMStructOutModel):
    """Full paper — structured output from paper generation."""

    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        json_schema_extra={"minLength": 40, "maxLength": 50},
        description="Short descriptive title for this paper generation task",
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        json_schema_extra={"minLength": 1200, "maxLength": 1500},
        description="Brief summary of the generated paper: sections written, figures included, compilation status",
    )
    out_expected_files: Annotated[FullPaperExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    )


# =============================================================================
# RESULT
# =============================================================================


class GenPaperRepoOut(BaseStepOut):
    """Final result of gen_paper module."""

    kind: Literal["gen_paper_repo_out"] = "gen_paper_repo_out"
    repo_url: str | None = Field(default=None, description="GitHub repo URL if created")

    # Artifacts
    gist_deployments: list[GistDeployment] = Field(default_factory=list)

    # Visualizations
    figures: list[Figure] = Field(default_factory=list)

    # Paper
    paper: PaperText | None = Field(default=None)
