"""Invention loop config models — agent configs, step configs, execution."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .llm import (
    GenHypoLLMClientConfig,
    MultiModelLLMClientConfig,
    NoveltyLLMClientConfig,
)


class ClaudeAgentConfig(BaseModel):
    """Claude agent configuration for structured output."""

    # ``llm_backend`` selects which entry under ``pipeline/harness/llm_backend.yaml.*``
    # this step talks to. The pipeline_config loader fills it in from
    # ``active.llm_backend`` for any block that doesn't set it explicitly.
    llm_backend: str = "claude_max"
    model: str = "claude-sonnet-4-6"
    effort: str = "high"
    max_turns: int | None = None
    agent_timeout: int | None = 7200
    agent_retries: int = 3
    seq_prompt_timeout: int | None = None
    seq_prompt_retries: int = 3
    message_timeout: int | None = 720
    message_retries: int = 5
    max_concurrent_agents: int = 5
    pod_timeout: int | None = None
    pod_start_retries: int = 3
    runpod_compute_profile: str | None = "cpu_light"
    disallowed_tools: list[str] = Field(default_factory=lambda: ["Task"])
    allowed_tools: list[str] | None = None
    model_config = ConfigDict(extra="allow")


class ClaudeAgentConfigNoConc(BaseModel):
    """Claude agent configuration without concurrency."""

    llm_backend: str = "claude_max"
    model: str = "claude-sonnet-4-6"
    effort: str = "high"
    max_turns: int | None = None
    agent_timeout: int | None = 7200
    agent_retries: int = 3
    seq_prompt_timeout: int | None = None
    seq_prompt_retries: int = 3
    message_timeout: int | None = 720
    message_retries: int = 5
    pod_timeout: int | None = None
    pod_start_retries: int = 3
    disallowed_tools: list[str] = Field(default_factory=lambda: ["Task"])
    allowed_tools: list[str] | None = None
    model_config = ConfigDict(extra="allow")


class GenHypoConfig(BaseModel):
    """Hypothesis generation module configuration."""

    seeded_hypos_per_llm: int = 0
    unseeded_hypos_per_llm: int = 1
    max_parallel: int | None = None
    llm_client: GenHypoLLMClientConfig = Field(default_factory=GenHypoLLMClientConfig)
    use_claude_agent: bool = True
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    model_config = ConfigDict(extra="allow")


class ReviewHypoConfig(BaseModel):
    """Single-pass adversarial review of the hypothesis.

    One call per outer ``gen_hypo_loop`` iteration when ``enabled``.
    """

    enabled: bool = True
    use_claude_agent: bool = True
    # ``llm_client`` / ``claude_agent`` is the canonical YAML key, matching
    # every other step (``gen_hypo``, ``gen_strat``, ``review_paper``, ...).
    # The legacy ``review_llm_client`` / ``review_claude_agent`` aliases are
    # kept for backward-compat with existing user pipeline.yaml snapshots.
    llm_client: MultiModelLLMClientConfig = Field(
        default_factory=MultiModelLLMClientConfig,
        validation_alias=AliasChoices("llm_client", "review_llm_client"),
    )
    claude_agent: ClaudeAgentConfigNoConc = Field(
        default_factory=ClaudeAgentConfigNoConc,
        validation_alias=AliasChoices("claude_agent", "review_claude_agent"),
    )
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class GenHypoLoopConfig(BaseModel):
    """Gen-hypo loop — gen_hypo → review_hypo × N iterations."""

    max_iterations: int = 2
    gen_hypo: GenHypoConfig = Field(default_factory=GenHypoConfig)
    review_hypo: ReviewHypoConfig = Field(default_factory=ReviewHypoConfig)
    model_config = ConfigDict(extra="allow")


class VerifyCitationsConfig(BaseModel):
    """Citation verification configuration."""

    parallel_fetch: bool = False
    retry: int = 2
    min_valid_citations: int = 5
    model_config = ConfigDict(extra="allow")


class NoveltyConfig(BaseModel):
    """Novelty evaluation configuration."""

    num_positive_per_llm: int = 1
    num_negative_per_llm: int = 1
    cap_mode: str = "equal"
    llm_client: NoveltyLLMClientConfig = Field(default_factory=NoveltyLLMClientConfig)
    use_claude_agent: bool = False
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    model_config = ConfigDict(extra="allow")


class FeasibilityConfig(BaseModel):
    """Feasibility evaluation configuration."""

    num_positive_per_llm: int = 1
    num_negative_per_llm: int = 1
    cap_mode: str = "equal"
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    use_claude_agent: bool = False
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    model_config = ConfigDict(extra="allow")


class AuditHypoConfig(BaseModel):
    """Hypothesis audit configuration."""

    max_concurrent_audits: int = 200
    verify_citations: VerifyCitationsConfig = Field(default_factory=VerifyCitationsConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    model_config = ConfigDict(extra="allow")


class VerifyArtifactsConfig(BaseModel):
    """Artifact verification configuration."""

    retry: int = 5
    min_valid_artifacts: int = 1
    model_config = ConfigDict(extra="allow")


class GenStratConfig(BaseModel):
    """Strategy generation configuration."""

    strats_per_call: int = 1
    calls_per_llm: int = 1
    art_limit: int | None = 5
    artifact_context_per_type: int = 10
    use_claude_agent: bool = True
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    verify_artifacts: VerifyArtifactsConfig = Field(default_factory=VerifyArtifactsConfig)
    model_config = ConfigDict(extra="allow")


class GenPlanConfig(BaseModel):
    """Plan generation configuration."""

    plans_per_strat: int = 1
    use_claude_agent: bool = True
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    model_config = ConfigDict(extra="allow")


class ResearchExecuteConfig(BaseModel):
    """Research execution configuration."""

    use_claude_agent: bool = True
    model: str = "gpt-5-mini"
    reasoning_effort: str = "medium"
    suffix: str | None = None
    max_tool_iterations: int = 10
    llm_timeout: int = 300
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    verify_retries: int = 2
    schema_retries: int = 3
    model_config = ConfigDict(extra="allow")


class AgentExecuteConfig(BaseModel):
    """Agent execution configuration."""

    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    verify_retries: int = 2
    schema_retries: int = 3
    min_examples: int = 50
    max_informal_loops: int = 3
    max_formal_loops: int = 5
    dataset_max_size: str = "300MB"
    dataset_search_tool_cap: int = 50
    dataset_chosen_for_preview_cap: int = 25
    dataset_chosen_for_download_cap: int = 15
    dataset_chosen_final_cap: int = 10
    model_config = ConfigDict(extra="allow")


class ExecuteConfig(BaseModel):
    """Execution configuration for artifact generation."""

    # Default 5 by design — execute fans out artifact-running Claude agents
    # (each spawns a worker pod / consumes RAM); keep parallelism low to
    # stay under RunPod quotas. Sibling ClaudeAgentConfig defaults to 20
    # because it's used for cheaper LLM-only steps.
    max_concurrent_agents: int = 5
    research: ResearchExecuteConfig = Field(default_factory=ResearchExecuteConfig)
    experiment: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    dataset: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    evaluation: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    proof: AgentExecuteConfig = Field(default_factory=AgentExecuteConfig)
    model_config = ConfigDict(extra="allow")


class GenPaperTextConfig(BaseModel):
    """Paper text generation in the invention loop."""

    start_at_iteration: int = 1
    verify_retries: int = 2
    claude_agent: ClaudeAgentConfig = Field(default_factory=ClaudeAgentConfig)
    model_config = ConfigDict(extra="allow")


class UpdHypoConfig(BaseModel):
    """Hypothesis revision — internal reflection using same LLM."""

    start_at_iteration: int = 1
    use_claude_agent: bool = True
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    model_config = ConfigDict(extra="allow")


class ReviewPaperConfig(BaseModel):
    """Adversarial paper review — different LLM for unbiased critique."""

    start_at_iteration: int = 1
    use_claude_agent: bool = True
    llm_client: MultiModelLLMClientConfig = Field(default_factory=MultiModelLLMClientConfig)
    claude_agent: ClaudeAgentConfigNoConc = Field(default_factory=ClaudeAgentConfigNoConc)
    model_config = ConfigDict(extra="allow")


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    patience: int = 2
    model_config = ConfigDict(extra="allow")


class InventionLoopConfig(BaseModel):
    """Invention loop configuration."""

    max_iterations: int = 3
    test_all_artifacts: bool = False
    allowed_artifacts: list[str] = Field(
        default_factory=lambda: ["research", "experiment", "dataset", "evaluation", "proof"]
    )
    gen_strat: GenStratConfig = Field(default_factory=GenStratConfig)
    gen_plan: GenPlanConfig = Field(default_factory=GenPlanConfig)
    execute: ExecuteConfig = Field(default_factory=ExecuteConfig)
    gen_paper_text: GenPaperTextConfig = Field(default_factory=GenPaperTextConfig)
    upd_hypo: UpdHypoConfig = Field(default_factory=UpdHypoConfig)
    review_paper: ReviewPaperConfig = Field(default_factory=ReviewPaperConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    model_config = ConfigDict(extra="allow")
