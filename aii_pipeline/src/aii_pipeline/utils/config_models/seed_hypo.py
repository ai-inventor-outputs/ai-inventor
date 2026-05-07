"""Seed hypothesis config models — invention KG pipeline settings."""

from pydantic import BaseModel, ConfigDict, Field


class InventionKGSelTopicsConfig(BaseModel):
    """Topic selection configuration for invention KG."""

    topics: list[str] = Field(
        default_factory=lambda: [
            "Data Mining Algorithms and Applications",
            "Opinion Dynamics and Social Influence",
            "Multi-Agent Systems and Negotiation",
            "Formal Methods in Verification",
            "Distributed Control Multi-Agent Systems",
            "Reinforcement Learning in Robotics",
            "Evolutionary Game Theory and Cooperation",
            "Speech and dialogue systems",
        ]
    )
    model_config = ConfigDict(extra="allow")


class InventionKGGetPapersConfig(BaseModel):
    """Paper retrieval configuration for invention KG."""

    email: str = ""
    papers_per_year: int = 10
    year_range: dict = Field(default_factory=lambda: {"start": 2020, "end": 2025})
    sort_by: str = "cited_by_count"
    model_config = ConfigDict(extra="allow")


class InventionKGGetTriplesClaudeAgentConfig(BaseModel):
    """Claude agent configuration for triple extraction."""

    model: str = "claude-sonnet-4-6"
    effort: str = "high"
    max_turns: int | None = None
    agent_timeout: int | None = 2400
    agent_retries: int = 3
    seq_prompt_timeout: int | None = None
    seq_prompt_retries: int = 3
    message_timeout: int | None = 720
    message_retries: int = 5
    disallowed_tools: list[str] = Field(default_factory=lambda: ["Task", "WebSearch", "WebFetch"])
    allowed_tools: list[str] | None = None
    model_config = ConfigDict(extra="allow")


class InventionKGGetTriplesConfig(BaseModel):
    """Triple extraction configuration for invention KG."""

    max_papers: int = 3
    # Default 10 by design — KG triple extraction runs Haiku-class agents per
    # paper; mid-tier parallelism balances throughput vs API rate limits.
    # Sibling steps differ (ExecuteConfig=5, ClaudeAgentConfig=20).
    max_concurrent_agents: int = 10
    stagger_delay: float = 2.0
    url_verification_retries: int = 2
    min_valid_urls: int = 0
    claude_agent: InventionKGGetTriplesClaudeAgentConfig = Field(
        default_factory=InventionKGGetTriplesClaudeAgentConfig
    )
    model_config = ConfigDict(extra="allow")


class InventionKGBlindSpotsConfig(BaseModel):
    """Blind spots detection configuration."""

    min_shared_concepts: int = 1
    max_similarity: float = 1.0
    entity_types: list[str] = Field(default_factory=lambda: ["method", "concept"])
    model_config = ConfigDict(extra="allow")


class InventionKGGenHypoSeedsConfig(BaseModel):
    """Hypothesis seed generation configuration."""

    blind_spots: InventionKGBlindSpotsConfig = Field(default_factory=InventionKGBlindSpotsConfig)
    model_config = ConfigDict(extra="allow")


class InventionKGGenGraphConfig(BaseModel):
    """Graph generation configuration for invention KG."""

    temporal_windows: list[list[int]] = Field(
        default_factory=lambda: [[2018, 2020], [2021, 2023], [2024, 2025]]
    )
    model_config = ConfigDict(extra="allow")


class InventionKGConfig(BaseModel):
    """Invention KG pipeline configuration."""

    first_step: str = "add_wikidata"
    last_step: str = "gen_graphs"
    sel_topics_out_dir: str = ""
    get_papers_out_dir: str = ""
    clean_papers_out_dir: str = ""
    get_triples_out_dir: str = ""
    add_wikidata_out_dir: str = ""
    link_to_papers_out_dir: str = ""
    gen_hypo_seeds_out_dir: str = ""
    gen_hypo_prompt_out_dir: str = ""
    gen_graphs_out_dir: str = ""
    sel_topics: InventionKGSelTopicsConfig = Field(default_factory=InventionKGSelTopicsConfig)
    get_papers: InventionKGGetPapersConfig = Field(default_factory=InventionKGGetPapersConfig)
    get_triples: InventionKGGetTriplesConfig = Field(default_factory=InventionKGGetTriplesConfig)
    gen_hypo_seeds: InventionKGGenHypoSeedsConfig = Field(
        default_factory=InventionKGGenHypoSeedsConfig
    )
    gen_graph: InventionKGGenGraphConfig = Field(default_factory=InventionKGGenGraphConfig)
    model_config = ConfigDict(extra="allow")


class SeedHypoSamplingConfig(BaseModel):
    """Sampling configuration for seed hypotheses."""

    sel_topics: str | list[str] = "auto"
    aii_prompt_topic_match_k: int = 1
    seed_sampling_pool: int = 20
    topics_per_agent: int = 2
    seeds_per_topic: int = 1
    model_config = ConfigDict(extra="allow")


class SeedHypoConfig(BaseModel):
    """Seed hypo module configuration."""

    first_step: str = "gen_seeds"
    last_step: str = "sample_seeds"
    invention_kg_seed_out_dir: str = ""
    sample_seeds_out_dir: str = ""
    invention_kg: InventionKGConfig = Field(default_factory=InventionKGConfig)
    sampling: SeedHypoSamplingConfig = Field(default_factory=SeedHypoSamplingConfig)
    model_config = ConfigDict(extra="allow")
