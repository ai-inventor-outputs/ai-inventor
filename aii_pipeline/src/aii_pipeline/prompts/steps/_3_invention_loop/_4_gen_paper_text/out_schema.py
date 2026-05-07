"""Schema for paper text — PaperText, FigureSpec, and verification helpers.

PaperText is the structured output schema for paper writing.
Figures are provided as a structured `figures` array alongside `paper_text`
which contains simple [FIGURE:fig_id] markers for positioning.

Verification checks bidirectional consistency:
- Every marker in text has a matching figure in the array
- Every figure in the array has a matching marker in text
"""

import re
from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import Figure
from pydantic import Field

# =============================================================================
# FIGURE SPEC (structured output from LLM)
# =============================================================================


class FigureSpec(LLMPromptModel, LLMStructOutModel):
    """Figure specification — structured output from paper writing agent.

    The LLM fills these as a list in PaperText.figures.
    Later converted to Figure objects for viz gen.
    """

    id: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Figure ID matching the [FIGURE:id] marker in paper_text (e.g., 'fig1')"
    )
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Short descriptive figure title"
    )
    caption: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="LaTeX figure caption — appears below the figure in the paper. Should describe what the figure shows and highlight key takeaways."
    )
    image_gen_detailed_description: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Detailed image generation prompt — axes, labels, ALL numeric values, colors, aspect ratio, layout. The image generator cannot read files; this is its ONLY input."
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of what this figure communicates"
    )

    def to_figure(self) -> Figure:
        """Convert to a Figure object for viz gen pipeline."""
        return Figure(
            id=self.id,
            title=self.title,
            caption=self.caption,
            image_gen_detailed_description=self.image_gen_detailed_description,
            summary=self.summary,
        )


# =============================================================================
# SCHEMAS
# =============================================================================

FIGURE_MARKER_PATTERN = re.compile(r"\[FIGURE:([\w]+)\]")
ARTIFACT_MARKER_PATTERN = re.compile(r"\[ARTIFACT:([\w]+)\]")


class PaperText(LLMPromptModel, LLMStructOutModel):
    """Paper text — structured output from paper writing agent.

    Structured output fields (LLMPrompt + LLMStructOut):
    - title, abstract, paper_text, figures, summary

    paper_text contains [FIGURE:fig_id] markers for positioning.
    figures contains the full specs as structured objects.

    Metadata fields (plain, set by pipeline code):
    - id
    """

    kind: Literal["paper_text"] = "paper_text"
    # Structured output fields (agent fills these)
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Paper title - concise, descriptive, captures the main contribution"
    )
    abstract: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Paper abstract")
    paper_text: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Full paper body text with markdown section headers (# Introduction, # Methods, # Results, # Discussion, # Conclusion). Use [FIGURE:fig_id] markers (e.g. [FIGURE:fig1]) to indicate where each figure should appear."
    )
    figures: Annotated[list[FigureSpec], LLMPrompt, LLMStructOut] = Field(
        default_factory=list,
        description="List of figure specifications. Each must have an id matching a [FIGURE:id] marker in paper_text.",
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of the paper's main contribution and findings"
    )

    # Metadata fields (set by pipeline code, not by agent)
    id: str = Field(default="", description="Draft ID")


# =============================================================================
# MARKER EXTRACTION
# =============================================================================


def extract_figure_markers(text: str) -> list[str]:
    """Extract figure IDs from [FIGURE:id] markers in text."""
    return FIGURE_MARKER_PATTERN.findall(text)


def get_figures_from_data(data: dict) -> list[Figure]:
    """Get Figure objects from structured output data."""
    raw_figures = data.get("figures", [])
    figures = []
    for fig_data in raw_figures:
        if isinstance(fig_data, dict):
            spec = FigureSpec(**fig_data)
            figures.append(spec.to_figure())
        elif isinstance(fig_data, FigureSpec):
            figures.append(fig_data.to_figure())
    return figures


# =============================================================================
# VERIFICATION
# =============================================================================


def verify_figures(paper_text: str, figures: list[Figure]) -> dict:
    """Verify bidirectional consistency between text markers and figure specs."""
    marker_ids = extract_figure_markers(paper_text)
    figure_ids = [f.id for f in figures]

    marker_set = set(marker_ids)
    figure_set = set(figure_ids)

    missing_figures = sorted(marker_set - figure_set)
    orphan_figures = sorted(figure_set - marker_set)

    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for fid in figure_ids:
        if fid in seen:
            duplicate_ids.append(fid)
        seen.add(fid)

    field_errors: list[str] = []
    for fig in figures:
        if not fig.title:
            field_errors.append(f"{fig.id}: missing title")
        if not fig.caption:
            field_errors.append(f"{fig.id}: missing caption")
        if not fig.image_gen_detailed_description:
            field_errors.append(f"{fig.id}: missing image_gen_detailed_description")

    valid = not missing_figures and not orphan_figures and not duplicate_ids and not field_errors

    return {
        "valid": valid,
        "marker_ids": marker_ids,
        "figure_ids": figure_ids,
        "missing_figures": missing_figures,
        "orphan_figures": orphan_figures,
        "duplicate_ids": duplicate_ids,
        "field_errors": field_errors,
    }


# =============================================================================
# ARTIFACT MARKER RESOLUTION
# =============================================================================


def extract_artifact_markers(text: str) -> list[str]:
    """Extract artifact IDs from [ARTIFACT:id] markers in text."""
    return ARTIFACT_MARKER_PATTERN.findall(text)


def resolve_artifact_markers(paper_text: str, repo_url: str, artifacts: list) -> str:
    """Replace [ARTIFACT:id] markers with LaTeX footnotes linking to the artifact's GitHub folder.

    First occurrence of each artifact gets a footnote with the URL.
    Subsequent occurrences are removed (the footnote already pointed to the code).
    """
    from aii_pipeline.steps._4_gen_paper_repo.utils.naming import (
        get_readable_folder_name,
    )

    artifact_ids = extract_artifact_markers(paper_text)
    if not artifact_ids:
        return paper_text

    title_by_id = {a.id: getattr(a, "title", "") for a in artifacts}
    repo_url = repo_url.rstrip("/")
    seen: set[str] = set()

    def _replace_marker(match: re.Match) -> str:
        aid = match.group(1)
        title = title_by_id.get(aid, "")
        if aid not in title_by_id:
            return ""
        folder = get_readable_folder_name(aid, title)
        if aid not in seen:
            seen.add(aid)
            url = f"{repo_url}/tree/main/{folder}"
            return f"\\footnote{{Code: \\url{{{url}}}}}"
        return ""

    return ARTIFACT_MARKER_PATTERN.sub(_replace_marker, paper_text)
