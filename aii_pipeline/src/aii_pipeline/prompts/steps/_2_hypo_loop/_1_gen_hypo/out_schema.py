"""Pydantic schema for hypothesis generation structured output."""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from aii_pipeline.steps.base import BaseStepOut
from pydantic import Field


class TermDefinition(LLMPromptModel, LLMStructOutModel):
    """A technical term and its definition."""

    term: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="The technical term")
    definition: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Clear definition of the term"
    )


class Hypothesis(LLMPromptModel, LLMStructOutModel):
    """A research hypothesis with validation approach."""

    kind: Literal["hypothesis"] = "hypothesis"
    """Discriminator for ``AnyOutput`` — see ``aii_lib/run/typed_union.py``.
    No LLMStructOut annotation: invisible to the LLM's structured-output
    schema, populated via the default at deserialize time."""

    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Concise, self-explanatory title"
    )
    hypothesis: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="The core hypothesis statement"
    )
    motivation: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Why this hypothesis matters - significance and impact"
    )
    assumptions: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="Key assumptions that must hold for this hypothesis (2-5 items)"
    )
    investigation_approach: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="High-level approach to investigating this hypothesis"
    )
    success_criteria: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What outcomes would confirm or disconfirm this hypothesis?"
    )
    related_works: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="The most similar existing works found during research. Each entry describes one related work: what it does and how the proposed hypothesis fundamentally differs from it."
    )
    inspiration: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="How the provided seed inspiration(s) shaped this hypothesis - what patterns, techniques, or insights were adapted"
    )
    terms: Annotated[list[TermDefinition], LLMPrompt, LLMStructOut] = Field(
        description="Definitions of key technical terms used in the hypothesis"
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of the hypothesis in 1-2 sentences"
    )


class GenHypoOut(BaseStepOut):
    """Output of the gen_hypo module."""

    kind: Literal["gen_hypo_out"] = "gen_hypo_out"
    hypotheses: list[dict] = Field(default_factory=list, description="Generated hypotheses")
