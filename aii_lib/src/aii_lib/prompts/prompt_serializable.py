"""LLMPromptModel — Pydantic BaseModel that knows how to render itself for LLM prompts.

Inherit from LLMPromptModel instead of BaseModel for any schema that will
be embedded in a prompt. Mark fields with LLMPrompt to control which appear:

    class Hypothesis(LLMPromptModel):
        title: Annotated[str, LLMPrompt] = Field(...)           # included
        hypothesis: Annotated[str, LLMPrompt] = Field(...)      # included
        hypothesis_id: str = Field(...)                         # excluded

Every LLMPromptModel must annotate its fields — only marked fields are
included. Unmarked fields are excluded.

Per-call override:
    hypo.to_prompt_yaml()                       # uses LLMPrompt markers
    hypo.to_prompt_yaml(include={"title"})      # overrides markers
"""

from __future__ import annotations

from pydantic import BaseModel

from .annotations import LLMPrompt
from .prompt_format import to_prompt_yaml, to_prompt_yaml_list


def _get_marked_fields(cls: type[BaseModel], marker: type) -> frozenset[str]:
    """Get fields marked with a given Annotated marker.

    Returns frozenset of field names (empty if none are marked).
    """
    return frozenset(
        name
        for name, info in cls.model_fields.items()
        if any(m is marker or isinstance(m, marker) for m in info.metadata)
    )


class LLMPromptModel(BaseModel):
    """BaseModel that can serialize itself to prompt-friendly YAML.

    Mark fields with Annotated[type, LLMPrompt] to include them.
    Unmarked fields are excluded.
    """

    def to_prompt_yaml(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        strip_nulls: bool = False,
    ) -> str:
        """Serialize this model to YAML for embedding in an LLM prompt.

        Args:
            include: Only include these fields. Overrides LLMPrompt markers.
            exclude: Remove these fields from the output.
            strip_nulls: Remove keys with None/empty values.

        Returns:
            Clean YAML string ready for prompt embedding.
        """
        fields = include if include is not None else _get_marked_fields(self.__class__, LLMPrompt)
        if exclude:
            fields = fields - exclude
        d = self.model_dump(mode="json", include=fields)
        return to_prompt_yaml(d, strip_nulls=strip_nulls)

    @staticmethod
    def list_to_prompt_yaml(
        items: list[LLMPromptModel],
        *,
        label: str = "Item",
        separator: str = "---",
        include: set[str] | None = None,
        strip_nulls: bool = False,
    ) -> str:
        """Serialize a list of models to labeled YAML blocks.

        Args:
            items: List of LLMPromptModel instances.
            label: Label for each block (e.g., "Strategy", "Hypothesis").
            separator: Block header separator.
            include: Only include these fields per item. Overrides LLMPrompt markers.
            strip_nulls: Remove keys with None/empty values.

        Returns:
            Labeled YAML blocks separated by blank lines.
        """
        dicts = [
            item.model_dump(
                mode="json",
                include=include
                if include is not None
                else _get_marked_fields(item.__class__, LLMPrompt),
            )
            for item in items
        ]
        return to_prompt_yaml_list(
            dicts,
            label=label,
            separator=separator,
            strip_nulls=strip_nulls,
        )
