"""Per-type planning rules for artifact strategy and plan prompts.

Import and call get_artifact_planning() with the artifact types you want to include.
"""


# =============================================================================
# PLANNING RULES PER ARTIFACT TYPE
# =============================================================================


def get_dataset_planning() -> str:
    """Planning rule for DATASET artifacts."""
    return """DATASET:
- Plan for REAL third-party datasets (HuggingFace, Kaggle, direct-download URLs) — downloadable within time and size constraints
- Describe dataset criteria (domain, size, format) — executors find exact sources, but you can suggest candidates or search directions
- ALWAYS prefer real datasets over synthetic. Synthetic is a LAST RESORT only when no suitable real data exists"""


def get_experiment_planning() -> str:
    """Planning rule for EXPERIMENT artifacts."""
    return """EXPERIMENT: Must depend on at least one DATASET. Define clear metrics and baselines before running. Consider trying multiple method variations rather than a single approach."""


def get_evaluation_planning() -> str:
    """Planning rule for EVALUATION artifacts."""
    return """EVALUATION: Must depend on at least one EXPERIMENT. Focus on statistical rigor and validity checks."""


def get_research_planning() -> str:
    """Planning rule for RESEARCH artifacts."""
    return """RESEARCH: Plan early — findings guide dataset selection, experiment design, and methodology."""


def get_proof_planning() -> str:
    """Planning rule for PROOF artifacts."""
    return """PROOF: Use only when the hypothesis requires formal mathematical guarantees. Lean 4 + Mathlib."""


# =============================================================================
# REGISTRY
# =============================================================================

_ARTIFACT_PLANNING = {
    "dataset": get_dataset_planning,
    "experiment": get_experiment_planning,
    "evaluation": get_evaluation_planning,
    "research": get_research_planning,
    "proof": get_proof_planning,
}

# Default order for artifact types
DEFAULT_ARTIFACT_TYPES = ["dataset", "experiment", "evaluation", "research", "proof"]


# =============================================================================
# PUBLIC API
# =============================================================================


def get_artifact_planning(artifact_types: list[str] | None = None) -> str:
    """Get formatted artifact planning rules for prompts.

    Args:
        artifact_types: List of artifact types to include (e.g., ["experiment", "research"]).
                       If None, includes all types in default order.

    Returns:
        Formatted string with planning rules wrapped in <artifact_planning_rules> tag.
    """
    if artifact_types is None:
        artifact_types = DEFAULT_ARTIFACT_TYPES

    # Normalize to lowercase
    artifact_types = [t.lower() for t in artifact_types]

    sections = []
    for artifact_type in artifact_types:
        if artifact_type in _ARTIFACT_PLANNING:
            sections.append(_ARTIFACT_PLANNING[artifact_type]())

    content = "\n".join(sections)
    return f"<artifact_planning_rules>\n{content}\n</artifact_planning_rules>"
