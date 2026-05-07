"""Schemas for review_hypo step and hypo_loop ledger.

Reuses ReviewerFeedback/Critique from the invention loop's review_paper
and RevisedHypothesis from upd_hypo — same structures, same semantics.

Adds :class:`HypoReviewerFeedback` — a ReviewerFeedback subclass that
carries the H↔H Moulines typology (``relation_type`` +
``relation_rationale``) describing how the iteration's hypothesis
relates to the previous iteration's. Mirrors the upd_hypo H↔H edge so
:func:`get_trace` can build hypothesis_edges between consecutive
hypo_loop iterations the same way it does for invention_loop.
"""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut
from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
    Critique,
    ReviewerFeedback,
    format_critiques_for_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._6_upd_hypo.out_schema import (
    RevisedHypothesis,
)
from pydantic import BaseModel, Field


class HypoReviewerFeedback(ReviewerFeedback):
    """ReviewerFeedback + Moulines H↔H typology for hypo_loop iterations.

    Adds ``relation_type`` + ``relation_rationale`` so the trace projection
    can build a typed edge from the previous iteration's hypothesis to
    this iteration's. On iteration 1 (no previous), both fields are
    empty/None.
    """

    relation_type: Annotated[
        Literal["evolution", "embedding", "replacement"] | None,
        LLMPrompt,
        LLMStructOut,
    ] = Field(
        default=None,
        description=(
            "Moulines's structuralist typology classifying how this iteration's "
            "hypothesis relates to the previous iteration's: "
            "'evolution' — refining specialised claims while keeping the same "
            "conceptual frame; "
            "'embedding' — the previous hypothesis is now a special case of a "
            "broader frame; "
            "'replacement' — rejecting the previous frame entirely (Kuhnian "
            "shift). Leave null on the first iteration (no previous hypothesis)."
        ),
    )
    relation_rationale: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        max_length=50,
        description=(
            "Brief rationale (≤50 chars) for the relation_type. Empty on the first iteration."
        ),
    )


class HypoReviewLedgerEntry(LLMPromptModel):
    """One entry in the hypothesis review ledger."""

    id: Annotated[str, LLMPrompt] = Field(default="", description="Entry ID")
    iteration: Annotated[int, LLMPrompt] = Field(description="Loop iteration number")
    score: Annotated[int, LLMPrompt] = Field(description="Quality score 1-10")
    num_major: Annotated[int, LLMPrompt] = Field(description="Number of major critiques")
    num_minor: Annotated[int, LLMPrompt] = Field(description="Number of minor critiques")
    overall_assessment: Annotated[str, LLMPrompt] = Field(description="Overall assessment summary")
    confidence_delta: str = Field(default="", description="Change in confidence")
    title_before: str = Field(default="", description="Hypothesis title before this iteration")
    title_after: str = Field(default="", description="Hypothesis title after this iteration")
    was_revised: bool = Field(default=True, description="Whether the hypothesis was revised")


class HypoIterationLedgerEntry(LLMPromptModel):
    """One entry per hypo_loop iteration — tracks gen_hypo + review_hypo state.

    Replaces the raw dict entries previously stored in the `ledger` list.
    """

    id: Annotated[str, LLMPrompt] = Field(default="", description="Entry ID (hypo_it{N})")
    iteration: Annotated[int, LLMPrompt] = Field(description="Loop iteration number")
    title: Annotated[str, LLMPrompt] = Field(
        default="", description="Hypothesis title this iteration"
    )
    title_previous: str = Field(default="", description="Hypothesis title from previous iteration")
    had_review_feedback: bool = Field(
        default=False, description="Whether gen_hypo had prior review feedback"
    )
    review_score_before: int | None = Field(
        default=None, description="Review score from previous iteration"
    )
    review_score_after: int | None = Field(
        default=None, description="Review score from this iteration"
    )
    num_critiques: int | None = Field(
        default=None, description="Number of critiques from this iteration's review"
    )


class ReviewHypoOut(BaseModel):
    """Output of the review_hypo module."""

    kind: Literal["review_hypo_out"] = "review_hypo_out"
    hypothesis: dict = Field(default_factory=dict, description="Final hypothesis")
    final_review: dict | None = Field(default=None, description="Last review feedback")


__all__ = [
    "Critique",
    "HypoIterationLedgerEntry",
    "HypoReviewLedgerEntry",
    "HypoReviewerFeedback",
    "ReviewHypoOut",
    "ReviewerFeedback",
    "RevisedHypothesis",
    "format_critiques_for_prompt",
]
