"""Schemas for plan generation — single inheritance hierarchy.

Plan (base) holds code-assigned metadata + common content fields.
Type-specific subclasses add their own typed fields.

Structured output:
- Content fields are marked with LLMStructOut (metadata excluded automatically)
- cls.plan_output_format() returns the Claude/OpenRouter output_format
  for a single plan object (one plan per artifact direction)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel

# Runtime import — used in ``BasePlan.artifact_dependencies: list[ArtifactDep]``
# annotation, which pydantic resolves against THIS module's globals at
# schema build (typed-union widening time). DO NOT move under
# TYPE_CHECKING (ruff TC001 will be tempted) — runtime symbol required.
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (  # noqa: TC002
    ArtifactDep,
)
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================


class PlanType(StrEnum):
    """Types of plans that can be generated."""

    EXPERIMENT = "experiment"
    RESEARCH = "research"
    PROOF = "proof"
    EVALUATION = "evaluation"
    DATASET = "dataset"


# =============================================================================
# BASE PLAN
# =============================================================================


class BasePlan(LLMPromptModel, LLMStructOutModel):
    """Base plan — common fields + code-assigned metadata.

    ID format: plan_{type}_id{N}_it{iteration}__{model}_idx{N}

    Content fields (LLMPrompt + LLMStructOut) are included in prompts and schemas.
    ``id`` and ``type`` are LLMPrompt only (visible in prompts, not LLM-generated).
    Other metadata fields (no markers) are excluded from both.
    """

    kind: Literal["base_plan"] = "base_plan"
    model_config = ConfigDict(extra="ignore")

    # Code-assigned metadata (LLMPrompt = visible in prompts, not LLM-generated)
    id: Annotated[str, LLMPrompt] = Field(default="", description="Unique plan ID")
    type: Annotated[PlanType, LLMPrompt] = Field(default=PlanType.EXPERIMENT)
    artifact_dependencies: list[ArtifactDep] = Field(default_factory=list)
    in_art_direction_id: str | None = Field(default=None)
    in_strat_id: str | None = Field(default=None)

    # Common content (LLM-filled)
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Short title for the plan")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="", description="Brief summary"
    )
    runpod_compute_profile: Annotated[str | None, LLMPrompt, LLMStructOut] = Field(
        default="cpu_light",
        description="Compute tier for execution — pick from the available profiles list (e.g., 'gpu', 'cpu_heavy', 'cpu_light'). Only used in RunPod mode.",
    )

    @classmethod
    def plan_output_format(cls) -> dict[str, Any]:
        """Build output_format for a single plan object.

        Uses LLMStructOut markers to auto-filter to content fields only.

        Returns:
            {"type": "json_schema", "schema": ...} ready for output_format= or
            access ["schema"] for response_format=.
        """
        return cls.to_struct_output()


# =============================================================================
# TYPE-SPECIFIC PLANS
# =============================================================================


class ProofPlan(BasePlan):
    """Plan for a PROOF artifact."""

    kind: Literal["proof_plan"] = "proof_plan"
    type: Annotated[Literal[PlanType.PROOF], LLMPrompt] = PlanType.PROOF

    informal_proof_draft: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Initial proof sketch in plain language - this is a first draft that may be refined or corrected during execution"
    )
    explanation: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Why this proof matters and how it advances the research"
    )


class ResearchPlan(BasePlan):
    """Plan for a RESEARCH artifact."""

    kind: Literal["research_plan"] = "research_plan"
    type: Annotated[Literal[PlanType.RESEARCH], LLMPrompt] = PlanType.RESEARCH

    question: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="", description="The specific research question to investigate"
    )
    research_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Step-by-step plan for web research to gather this research"
    )
    explanation: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Why this research matters and what question it answers"
    )


class DatasetPlan(BasePlan):
    """Plan for a DATASET artifact."""

    kind: Literal["dataset_plan"] = "dataset_plan"
    type: Annotated[Literal[PlanType.DATASET], LLMPrompt] = PlanType.DATASET

    ideal_dataset_criteria: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What makes an ideal dataset for this purpose - size, format, content requirements"
    )
    dataset_search_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Step-by-step plan for finding/creating this dataset - sources to check, fallback options"
    )
    target_num_datasets: Annotated[int, LLMPrompt, LLMStructOut] = Field(
        description="How many individual datasets should be delivered. Count each dataset separately, not collections — a benchmark suite of N datasets counts as N. This controls how broadly the executor searches, so setting it too low will under-collect."
    )


class ExperimentPlan(BasePlan):
    """Plan for an EXPERIMENT artifact."""

    kind: Literal["experiment_plan"] = "experiment_plan"
    type: Annotated[Literal[PlanType.EXPERIMENT], LLMPrompt] = PlanType.EXPERIMENT

    implementation_pseudocode: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="High-level pseudocode for the experiment implementation"
    )
    fallback_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What to do if the primary approach fails - alternative methods, simplified versions"
    )
    testing_plan: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="How to validate the experiment works: start with small/fast tests, look for confirmation signals before running full-scale experiments"
    )


class EvaluationPlan(BasePlan):
    """Plan for an EVALUATION artifact."""

    kind: Literal["evaluation_plan"] = "evaluation_plan"
    type: Annotated[Literal[PlanType.EVALUATION], LLMPrompt] = PlanType.EVALUATION

    metrics_descriptions: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What metrics will be computed and how they're defined"
    )
    metrics_justification: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Why these metrics are the right ones - what do they tell us about the hypothesis"
    )


class GenPlanOut(BaseModel):
    """Aggregate output of gen_plan module with all plans produced.

    Every plan produced across the parallel tasks in one iteration.
    Used as the typed payload for ``module_output(output=...)``; readers
    walk ``module.output.plans`` rather than the legacy plural list.
    """

    kind: Literal["gen_plan_out"] = "gen_plan_out"
    plans: list[ProofPlan | ResearchPlan | DatasetPlan | ExperimentPlan | EvaluationPlan] = Field(
        default_factory=list,
    )


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

PLAN_SCHEMAS: dict[str, type[BasePlan]] = {
    "proof": ProofPlan,
    "research": ResearchPlan,
    "dataset": DatasetPlan,
    "experiment": ExperimentPlan,
    "evaluation": EvaluationPlan,
}


def get_plan_schema(artifact_type: str) -> type[BasePlan]:
    """Get the plan schema class for a given artifact type."""
    return PLAN_SCHEMAS.get(artifact_type, ResearchPlan)


def verify_compute_profile(
    plan_dict: dict,
    artifact_type: str,
    allowed_profiles: dict[str, list[str]],
) -> list[str]:
    """Verify runpod_compute_profile is valid for the artifact type.

    Args:
        plan_dict: Plan dict with runpod_compute_profile field.
        artifact_type: Artifact type (e.g. "experiment").
        allowed_profiles: Map of artifact_type -> list of allowed profile names.

    Returns:
        List of error strings (empty = valid).
    """
    errors: list[str] = []
    profile = plan_dict.get("runpod_compute_profile", "")

    if not profile:
        errors.append("runpod_compute_profile is empty — must specify a profile name")
        return errors

    allowed = allowed_profiles.get(artifact_type, [])
    if not allowed:
        return errors

    if profile not in allowed:
        errors.append(
            f"runpod_compute_profile '{profile}' is not allowed for artifact type '{artifact_type}'. "
            f"Allowed profiles: {allowed}"
        )

    return errors
