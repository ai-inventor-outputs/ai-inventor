"""Schema for artifact demo generation step.

Defines:
- DemoType enum and BaseDemo hierarchy (CodeDemo, LeanDemo, MarkdownDemo)
- Demo: Structured output from demo notebook generation
- DemoExpectedFiles: Expected output files
"""

from enum import StrEnum
from typing import Annotated, Literal

from aii_lib.prompts import (
    BaseExpectedFiles,
    LLMPrompt,
    LLMPromptModel,
    LLMStructOut,
    LLMStructOutModel,
)
from pydantic import BaseModel, Field

# =============================================================================
# DEMO TYPE HIERARCHY
# =============================================================================


class DemoType(StrEnum):
    CODE = "code"
    LEAN = "lean"
    MARKDOWN = "markdown"


class BaseDemo(BaseModel):
    """Base demo — common fields for all demo types."""

    kind: Literal["base_demo"] = "base_demo"
    id: str = Field(description="Artifact ID this demo belongs to")
    type: DemoType = Field(description="Demo type discriminator")
    iteration: int = Field(
        default=0,
        description="invention_loop iter that produced the source artifact (carried over from BaseArtifact.iteration so deploy_gh can route per-iter)",
    )
    title: str = Field(default="", description="Short descriptive title for this demo")
    summary: str = Field(default="", description="Brief summary of what this demo shows")
    original_path: str = Field(default="", description="Path to source workspace")
    demo_path: str = Field(default="", description="Path to demo output")


class DemoExpectedFiles(BaseExpectedFiles):
    """Expected output files from code demo notebook generation."""

    notebook: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        description="Path to the generated demo notebook. Example: 'code_demo.ipynb'",
    )
    mini_data_file: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        description="Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
    )


class LeanDemoExpectedFiles(BaseExpectedFiles):
    """Expected output files from lean demo generation."""

    lean_file: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="", description="Path to the Lean 4 proof file. Example: 'proof.lean'"
    )


class CodeDemo(BaseDemo, LLMPromptModel, LLMStructOutModel):
    """Dataset/experiment/evaluation → Jupyter notebook demo.

    Title and summary come from the parent artifact (gen_art step),
    not from the demo agent. The agent only outputs expected files.
    """

    kind: Literal["code_demo"] = "code_demo"
    type: Literal[DemoType.CODE] = DemoType.CODE
    out_expected_files: Annotated[DemoExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        default_factory=DemoExpectedFiles,
        description="All output files you created. Must include the demo notebook.",
    )
    notebook_path: str = Field(default="", description="Path to generated notebook")


class LeanDemo(BaseDemo):
    """Proof → markdown + Lean playground link."""

    kind: Literal["lean_demo"] = "lean_demo"
    type: Literal[DemoType.LEAN] = DemoType.LEAN
    out_expected_files: LeanDemoExpectedFiles = Field(
        default_factory=LeanDemoExpectedFiles,
        description="Expected output files from lean demo.",
    )
    playground_url: str = Field(default="", description="Lean playground URL")


class MarkdownDemo(BaseDemo):
    """Research → markdown summary."""

    kind: Literal["markdown_demo"] = "markdown_demo"
    type: Literal[DemoType.MARKDOWN] = DemoType.MARKDOWN


AnyDemo = CodeDemo | LeanDemo | MarkdownDemo


class GenArtDemoOut(BaseModel):
    """Aggregate output of gen_art_demo module with artifact demos.

    One entry per artifact's prepared demo (Lean / Markdown / Code paths
    all flow through this list). Used as the typed payload for
    ``module_output(output=...)`` so readers walk
    ``module.output.demos`` rather than the legacy
    ``ModuleOutputMessage.outputs`` list.
    """

    kind: Literal["gen_art_demo_out"] = "gen_art_demo_out"
    demos: list[CodeDemo | LeanDemo | MarkdownDemo] = Field(
        default_factory=list,
        description="All prepared demos across the 3 conversion paths.",
    )
