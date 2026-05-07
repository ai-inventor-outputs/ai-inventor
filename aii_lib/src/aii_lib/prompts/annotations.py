"""Field markers for LLMPromptModel and LLMStructOutModel.

Use with Annotated to declare which fields appear in prompt YAML
and/or structured output JSON schemas.

Usage:
    from aii_lib.prompts import LLMPrompt, LLMStructOut

    class Hypothesis(LLMPromptModel, LLMStructOutModel):
        title: Annotated[str, LLMPrompt, LLMStructOut] = Field(...)      # both
        hypothesis: Annotated[str, LLMPrompt, LLMStructOut] = Field(...) # both
        hypothesis_id: str = Field(...)                                    # neither

Every LLMPromptModel/LLMStructOutModel must annotate its fields.
Only marked fields are included — unmarked fields are excluded.
"""


class LLMPrompt:
    """Marker: include this field in .to_prompt_yaml() output."""


class LLMStructOut:
    """Marker: include this field in .to_struct_output() JSON schema."""
