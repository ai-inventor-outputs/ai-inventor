"""Schema for review_paper step — adversarial paper review."""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from pydantic import Field


class Critique(LLMPromptModel, LLMStructOutModel):
    """A single actionable critique from the reviewer."""

    id: Annotated[str, LLMPrompt] = Field(default="", description="Critique ID (code-assigned)")
    category: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Category: 'methodology', 'evidence', 'novelty', 'clarity', 'scope', or 'rigor'"
    )
    severity: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Severity: 'major' or 'minor'"
    )
    description: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Clear description of the issue"
    )
    suggested_action: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Concrete suggestion for how to address this critique"
    )


class DimensionScore(LLMPromptModel, LLMStructOutModel):
    """Score for a single review dimension with improvement suggestions."""

    dimension: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Dimension name: 'soundness', 'presentation', or 'contribution'"
    )
    score: Annotated[int, LLMPrompt, LLMStructOut] = Field(
        description="Score from 1 (poor) to 4 (excellent)"
    )
    justification: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief justification for this score"
    )
    improvements: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        default_factory=list,
        description="Specific improvements to raise the score (what + how + why)",
    )


class ReviewerFeedback(LLMPromptModel, LLMStructOutModel):
    """Adversarial review of the paper draft.

    ID format: review_it{iteration}__{model}
    """

    kind: Literal["reviewer_feedback"] = "reviewer_feedback"
    id: Annotated[str, LLMPrompt] = Field(default="", description="Review ID (code-assigned)")
    overall_assessment: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Overall assessment of the paper's quality and readiness"
    )
    strengths: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="Key strengths of the paper"
    )
    dimension_scores: Annotated[list[DimensionScore], LLMPrompt, LLMStructOut] = Field(
        default_factory=list,
        description="Scores (1-4) for: soundness, presentation, contribution",
    )
    critiques: Annotated[list[Critique], LLMPrompt, LLMStructOut] = Field(
        description="Actionable critiques — specific issues with concrete suggestions"
    )
    score: Annotated[int, LLMPrompt, LLMStructOut] = Field(
        description="Overall quality score from 1 (very strong reject) to 10 (award quality)"
    )
    confidence: Annotated[int, LLMPrompt, LLMStructOut] = Field(
        default=3,
        description="Confidence in assessment from 1 (educated guess) to 5 (absolutely certain)",
    )


def format_critiques_for_prompt(feedback: ReviewerFeedback | dict) -> str:
    """Format reviewer critiques for inclusion in gen_strat/gen_paper_text prompts.

    Accepts either a ReviewerFeedback object or a dict (from JSON resume).
    """
    if isinstance(feedback, dict):
        critiques = feedback.get("critiques", [])
    else:
        critiques = feedback.critiques

    if not critiques:
        return "No critiques from previous review."

    lines = []
    for c in critiques:
        if isinstance(c, dict):
            sev = c.get("severity", "minor").upper()
            cat = c.get("category", "")
            desc = c.get("description", "")
            action = c.get("suggested_action", "")
        else:
            sev = c.severity.upper()
            cat = c.category
            desc = c.description
            action = c.suggested_action
        lines.append(f"- [{sev}] ({cat}) {desc}")
        lines.append(f"  Action: {action}")
    return "\n".join(lines)
