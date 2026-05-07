"""Artifact type context - descriptions of available artifact types for prompts.

Import and call get_artifact_context() with the artifact types you want to include.
"""


# =============================================================================
# ARTIFACT TYPE DESCRIPTIONS
# =============================================================================


def get_experiment_description() -> str:
    """Description of EXPERIMENT artifact type."""
    return """EXPERIMENT
Run code to test hypotheses, implement methods, and collect empirical results.
Runtime: Python 3.10, UV (any pip package), isolated workspace, gradual scaling (mini → full data).
Tools: Full Bash/Python/filesystem access, WebSearch, WebFetch, aii_web_tools__fetch_grep (regex search over full document text), skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Implement and run any code-based experiment, compare method vs baselines.
Deps: REQUIRED at least one DATASET | OPTIONAL RESEARCH for methodology guidance"""


def get_evaluation_description() -> str:
    """Description of EVALUATION artifact type."""
    return """EVALUATION
Evaluate experiment results with metrics, statistical analysis, and validity checks.
Runtime: Python 3.10, UV (any evaluation library), isolated workspace, gradual scaling matching experiment.
Tools: Full Bash/Python/filesystem access, WebSearch, WebFetch, aii_web_tools__fetch_grep (regex search over full document text), skills.
Skills: aii-json (schema validation), aii-openrouter-llms (call any LLM — GPT, Gemini, Llama, etc.), domain-specific as needed.
Capabilities: Compute any quantitative metrics and statistical tests, analyze validity and robustness.
Deps: REQUIRED at least one EXPERIMENT | OPTIONAL DATASET if reference data needed"""


def get_dataset_description() -> str:
    """Description of DATASET artifact type."""
    return """DATASET
Collect, prepare, and merge datasets for experiments and analysis.
Runtime: Python 3.10, UV, isolated workspace.
Tools: Full Bash/Python/filesystem access, WebSearch, WebFetch, aii_web_tools__fetch_grep (regex search over full document text), skills.
Skills: aii-hf-datasets (HuggingFace Hub — ML datasets, many UCI/OpenML/Kaggle mirrors), aii-owid-datasets (Our World in Data — global statistics), aii-json (schema validation). Also any Python source (sklearn.datasets, openml, direct URLs, APIs) — must verify within 300MB limit.
Capabilities: Search, acquire, transform, combine, and standardize data from any available source.
Deps: REQUIRED none | OPTIONAL RESEARCH for guidance on what data to collect"""


def get_research_description() -> str:
    """Description of RESEARCH artifact type."""
    return """RESEARCH
Web research to answer key questions — like a researcher making decisions.
Runtime: LLM Agent, no code execution.
Tools: WebSearch, WebFetch, aii_web_tools__fetch_grep (regex search over full document text).
Capabilities: Find, synthesize, and compare information across sources; survey SOTA and best practices.
Deps: REQUIRED none | OPTIONAL other RESEARCH to build on prior findings"""


def get_proof_description() -> str:
    """Description of PROOF artifact type."""
    return """PROOF
Formally prove mathematical statements in Lean 4 with automated iteration.
Runtime: Claude Agent with Lean 4 compiler feedback loop.
Tools: Full Bash/Python/filesystem access, WebSearch, WebFetch, aii_web_tools__fetch_grep (regex search over full document text), skills.
Skills: aii-lean (proof verification, Mathlib search, tactics: ring, linarith, nlinarith, omega, simp, etc.)
Capabilities: Formally verify properties and inequalities, iterative proof development, lemma decomposition.
Deps: REQUIRED none | OPTIONAL RESEARCH for mathematical background"""


# =============================================================================
# REGISTRY
# =============================================================================

_ARTIFACT_DESCRIPTIONS = {
    "experiment": get_experiment_description,
    "evaluation": get_evaluation_description,
    "dataset": get_dataset_description,
    "research": get_research_description,
    "proof": get_proof_description,
}

# Default order for artifact types
DEFAULT_ARTIFACT_TYPES = ["dataset", "experiment", "evaluation", "research", "proof"]


# =============================================================================
# PUBLIC API
# =============================================================================


def get_artifact_context(artifact_types: list[str] | None = None) -> str:
    """Get formatted artifact type descriptions for prompts.

    Args:
        artifact_types: List of artifact types to include (e.g., ["experiment", "research"]).
                       If None, includes all types in default order.

    Returns:
        Formatted string with artifact type descriptions wrapped in <artifact_types> tag.
    """
    if artifact_types is None:
        artifact_types = DEFAULT_ARTIFACT_TYPES

    # Normalize to lowercase
    artifact_types = [t.lower() for t in artifact_types]

    sections = []
    for artifact_type in artifact_types:
        if artifact_type in _ARTIFACT_DESCRIPTIONS:
            sections.append(_ARTIFACT_DESCRIPTIONS[artifact_type]())

    content = "\n\n".join(sections)
    return f"<artifact_types>\n{content}\n</artifact_types>"
