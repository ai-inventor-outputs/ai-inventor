"""Schema for visualization generation.

Defines the unified Figure class and output file helpers for viz generation.

Figure lifecycle:
1. FigureSpec from structured output → converted to Figure (figure_path="")
2. viz gen fills in figure_path after image generation
3. gen_full_paper uses Figure with figure_path for LaTeX insertion
"""

from typing import Annotated, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.prompts import (
    BaseExpectedFiles,
    LLMPrompt,
    LLMPromptModel,
    LLMStructOut,
    LLMStructOutModel,
)
from pydantic import Field

# =============================================================================
# FIGURE
# =============================================================================


class Figure(LLMPromptModel):
    """A figure — from structured output FigureSpec, with optional generated image path.

    All fields are LLMPrompt-annotated for YAML serialization in prompts.
    """

    kind: Literal["figure"] = "figure"
    id: Annotated[str, LLMPrompt] = Field(
        description="Figure ID (e.g., 'fig1'). Links to [FIGURE:fig1] marker in paper text."
    )
    title: Annotated[str, LLMPrompt] = Field(description="Short descriptive figure title")
    caption: Annotated[str, LLMPrompt] = Field(
        default="",
        description="LaTeX figure caption — appears below the figure in the paper",
    )
    image_gen_detailed_description: Annotated[str, LLMPrompt] = Field(
        default="",
        description="Detailed image generation prompt — axes, labels, ALL numeric values, colors, layout. The image generator's ONLY input.",
    )
    aspect_ratio: Annotated[
        Literal["1:1", "4:3", "3:2", "16:9", "21:9", "3:4", "9:16"], LLMPrompt
    ] = Field(
        default="21:9",
        description="Aspect ratio for image generation. Pick by figure type: '21:9' for architecture diagrams / pipelines / flow charts (preferred for the paper's hero diagram), '16:9' for side-by-side comparisons / multi-panel results, '4:3' for dense charts, '1:1' for heatmaps / confusion matrices / scatter plots, '3:4' or '9:16' for vertical layouts.",
    )
    summary: Annotated[str, LLMPrompt] = Field(
        default="", description="Brief summary of what this figure shows"
    )
    figure_path: Annotated[str, LLMPrompt] = Field(
        default="",
        description="Path to the generated image file (e.g., 'figures/fig1_v0.jpg'). Empty before viz gen.",
    )


# =============================================================================
# STRUCTURED OUTPUT (agent output schema for expected files validation)
# =============================================================================


class VizExpectedFiles(BaseExpectedFiles):
    """Expected output files from viz generation."""

    image_path: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to the generated figure image file. Example: 'fig1_v0.jpg'"
    )


class VizFigureOutput(LLMPromptModel, LLMStructOutModel):
    """Structured output from viz figure generation agent."""

    kind: Literal["viz_figure_output"] = "viz_figure_output"
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        json_schema_extra={"minLength": 40, "maxLength": 50},
        description="Short descriptive title for the generated figure",
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        json_schema_extra={"minLength": 1200, "maxLength": 1500},
        description="Brief summary of the generated figure: what it shows, style, any issues fixed",
    )
    out_expected_files: Annotated[VizExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="Output file you created. Must include the generated figure image path."
    )


# =============================================================================
# CONSTANTS
# =============================================================================

VIZ_OUTPUT_FORMAT = "jpg"


# =============================================================================
# OUTPUT FILE HELPERS
# =============================================================================


def get_expected_out_file(figure_id: str, variation_idx: int) -> ExpectedFile:
    """Get expected output file for a single figure variation."""
    filename = f"{figure_id}_v{variation_idx}.{VIZ_OUTPUT_FORMAT}"
    return ExpectedFile(filename, f"Figure image for {figure_id} variation {variation_idx}")


def get_expected_out_files(figure_id: str, num_variations: int) -> list[ExpectedFile]:
    """Get all expected output files for a figure with multiple variations."""
    return [get_expected_out_file(figure_id, i) for i in range(num_variations)]


def get_output_filename(figure_id: str, variation_idx: int) -> str:
    """Get the output filename for a figure."""
    return f"{figure_id}_v{variation_idx}.{VIZ_OUTPUT_FORMAT}"


def get_iterations_folder(figure_id: str) -> str:
    """Get the subfolder name for iteration attempts (e.g. 'fig1_all')."""
    return f"{figure_id}_all"
