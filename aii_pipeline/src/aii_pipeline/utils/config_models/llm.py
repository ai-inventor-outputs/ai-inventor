"""LLM client and model config models."""

from pydantic import BaseModel, ConfigDict, Field


class ModelEntry(BaseModel):
    """Single model entry in a multi-model config."""

    model: str
    reasoning_effort: str = "medium"
    suffix: str = ""

    model_config = ConfigDict(extra="allow")


class MultiModelLLMClientConfig(BaseModel):
    """LLM client config with multiple models."""

    client: str = "openrouter"
    llm_timeout: int = 600
    suffix: str = ""
    models: list[ModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class GenHypoModelEntry(ModelEntry):
    """Model entry for gen_hypo with research-specific settings."""

    max_tool_iterations: int = 100


class GenHypoLLMClientConfig(BaseModel):
    """LLM client config for gen_hypo with multiple models."""

    client: str = "openrouter"
    llm_timeout: int = 1200
    models: list[GenHypoModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class NoveltyModelEntry(ModelEntry):
    """Model entry for novelty audits with web search settings."""

    max_tool_iterations: int = 20


class NoveltyLLMClientConfig(BaseModel):
    """LLM client config for novelty audits with multiple models and web search."""

    client: str = "openrouter"
    llm_timeout: int = 300
    suffix: str = ""
    models: list[NoveltyModelEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")
