"""Schemas for artifact generation — base classes and pool objects.

Base Classes:
- BaseArtifact: Base for all artifact types (pool + per-type inheritance)
- BaseExpectedFiles: Base for per-type expected file specifications

Enums:
- ArtifactType: Enum for artifact types

Per-type subclasses live in their own subdirectories:
- research/schema.py, experiment/schema.py, dataset/schema.py, etc.
"""

from enum import StrEnum
from typing import Annotated, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.prompts import (
    BaseExpectedFiles,  # noqa: F401  (re-exported by per-type out_schema files)
    LLMPrompt,
    LLMPromptModel,
    LLMStructOut,
    LLMStructOutModel,
)
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
    ArtifactDep,
)
from pydantic import Field

# =============================================================================
# POOL SCHEMAS
# =============================================================================


class ArtifactType(StrEnum):
    """Types of artifacts that can be produced."""

    EXPERIMENT = "experiment"
    RESEARCH = "research"
    PROOF = "proof"
    EVALUATION = "evaluation"
    DATASET = "dataset"


class BaseArtifact(LLMPromptModel, LLMStructOutModel):
    """A completed artifact.

    Content fields (title, summary) have LLMPrompt + LLMStructOut markers.
    ``id`` and ``type`` are LLMPrompt only (visible in prompts, not LLM-generated).
    Other metadata fields are code-assigned (no markers, excluded from both).

    Only successful artifacts are stored in the pool.

    ID format: {type}_id{N}_it{iteration}__{model}
    """

    kind: Literal["base_artifact"] = "base_artifact"
    id: Annotated[str, LLMPrompt] = Field(
        default="", description="Unique artifact ID (e.g., experiment_id1_it1__sonnet)"
    )
    type: Annotated[ArtifactType, LLMPrompt] = Field(
        default=ArtifactType.RESEARCH, description="Type of artifact"
    )
    in_plan_id: str = Field(default="", description="ID of the plan this artifact was created from")
    in_dependencies: list[ArtifactDep] = Field(
        default_factory=list,
        description="Artifacts this artifact depended on at execution time, each with a short type label",
    )
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        json_schema_extra={"minLength": 40, "maxLength": 60},
        description="Descriptive title (40-60 characters). Must describe content, NOT a status message.",
    )
    layman_summary: Annotated[str, LLMStructOut] = Field(
        default="",
        json_schema_extra={"minLength": 100, "maxLength": 120},
        description="One-sentence plain-language summary of what this artifact does, accessible to non-experts. Used only in the per-artifact README, not in downstream prompts.",
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        json_schema_extra={"minLength": 1200, "maxLength": 1500},
        description="Summary for downstream artifacts: what this artifact provides",
    )
    iteration: int = Field(
        default=0,
        description="invention_loop iteration that produced this artifact (1-based; 0 means unset). Stamped at make_artifact time so downstream code (gen_paper_repo) can route per-iter without parsing paths.",
    )
    workspace_path: Annotated[str | None, LLMPrompt] = Field(
        default=None, description="Absolute path to artifact workspace"
    )
    out_expected_files: list[str] = Field(
        default_factory=list,
        description="Files executor should create (for verification)",
    )
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default_factory=list, description="Primary file(s) to convert to demo formats"
    )
    out_dependency_files: Annotated[dict[str, str | list[str] | None], LLMPrompt] = Field(
        default_factory=dict,
        description="Output files that dependent artifacts can consume.",
    )
