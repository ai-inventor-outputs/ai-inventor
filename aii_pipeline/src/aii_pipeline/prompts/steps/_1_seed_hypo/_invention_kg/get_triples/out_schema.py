"""Pydantic schemas for invention_kg triples extraction."""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from pydantic import Field, field_validator


class Triple(LLMPromptModel, LLMStructOutModel):
    """A single knowledge triple extracted from a paper."""

    name: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        ..., description="Wikipedia article title for the entity"
    )
    relation: Annotated[Literal["uses", "proposes"], LLMPrompt, LLMStructOut] = Field(
        ..., description="How the paper relates to this entity"
    )
    entity_type: Annotated[
        Literal["task", "method", "data", "artifact", "tool", "concept", "other"],
        LLMPrompt,
        LLMStructOut,
    ] = Field(..., description="Type of entity")
    wikipedia_url: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        ...,
        description="Wikipedia URL (must start with https://en.wikipedia.org/wiki/)",
    )
    relevance: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        ..., description="1 sentence explaining why this matters"
    )

    @field_validator("wikipedia_url")
    @classmethod
    def validate_wikipedia_url(cls, v: str) -> str:
        if not v.startswith("https://en.wikipedia.org/wiki/"):
            raise ValueError("URL must start with https://en.wikipedia.org/wiki/")
        return v


class Triples(LLMPromptModel, LLMStructOutModel):
    """Structured output for triples extraction from a research paper."""

    paper_type: Annotated[Literal["contribution", "survey"], LLMPrompt, LLMStructOut] = Field(
        ...,
        description="contribution = proposes something new, survey = reviews existing work",
    )
    triples: Annotated[list[Triple], LLMPrompt, LLMStructOut] = Field(
        default_factory=list, description="List of extracted triples"
    )


__all__ = ["Triple", "Triples"]
