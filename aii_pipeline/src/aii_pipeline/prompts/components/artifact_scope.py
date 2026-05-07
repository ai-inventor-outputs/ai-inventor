"""Artifact executor scope constraints for strategy and plan prompts.

Each artifact executor has a highly optimized prompt specifically for its task.
The executor will follow its prompt, not ad-hoc instructions from the plan.
This component tells the strategy/plan LLM what each executor actually does
so it assigns the right work to the right artifact type.

Import and call get_artifact_scope() with the artifact types you want to include.
"""


# =============================================================================
# SCOPE CONSTRAINTS PER ARTIFACT TYPE
# =============================================================================


def get_dataset_scope() -> str:
    return """DATASET executor scope:
  Output: data_out.json with rows of {input, output, metadata_fold, ...} — raw data only, no derived computations
  DOES: Download/generate datasets, analyze candidates to pick the best ones, standardize to JSON schema (features, labels, folds, metadata), validate schema, split into full/mini/preview
  DOES NOT: Run experiments, train models, compute derived statistics (PID/MI/correlations/synergy matrices) as final output
  If you need to COMPUTE something from data (synergy matrices, MI scores, timing benchmarks), use an EXPERIMENT artifact instead"""


def get_experiment_scope() -> str:
    return """EXPERIMENT executor scope:
  Output: method_out.json with results (metrics, predictions, analysis) — the core computational work
  DOES: Implement and run methods/algorithms, compute metrics, compare approaches, produce quantitative results
  DOES NOT: Collect new datasets (depends on DATASET artifacts for input data), write formal proofs
  This is the right artifact for any code that processes data and produces results"""


def get_evaluation_scope() -> str:
    return """EVALUATION executor scope:
  Output: eval_out.json with evaluation results
  DOES: Any evaluation of experiment results — metrics, statistical tests, ablations, comparisons, visualizations, robustness checks, error analysis, etc.
  DOES NOT: Implement new methods (use EXPERIMENT), collect data (use DATASET)
  This is for analyzing experiment outputs from any angle"""


def get_research_scope() -> str:
    return """RESEARCH executor scope:
  Output: research_out.json with {answer, sources, follow_up_questions} + research_report.md
  DOES: Web research — search, read, synthesize information from papers/docs/APIs into a structured report
  DOES NOT: Run code, download files, execute scripts, compute anything — no Bash/Python access
  Use for literature surveys, API documentation, technical specifications — pure information gathering"""


def get_proof_scope() -> str:
    return """PROOF executor scope:
  Output: Lean 4 proof files (.lean) with verified theorems
  DOES: Write and verify Lean 4 formal proofs with Mathlib, iterative compilation
  DOES NOT: Run Python experiments, collect data, do empirical analysis
  Use only when formal mathematical guarantees are needed"""


# =============================================================================
# REGISTRY
# =============================================================================

_ARTIFACT_SCOPE = {
    "dataset": get_dataset_scope,
    "experiment": get_experiment_scope,
    "evaluation": get_evaluation_scope,
    "research": get_research_scope,
    "proof": get_proof_scope,
}

DEFAULT_ARTIFACT_TYPES = ["dataset", "experiment", "evaluation", "research", "proof"]


# =============================================================================
# PUBLIC API
# =============================================================================


def get_artifact_scope(artifact_types: list[str] | None = None) -> str:
    """Get artifact executor scope constraints for prompts.

    Args:
        artifact_types: List of artifact types to include.
                       If None, includes all types in default order.

    Returns:
        Formatted string with scope constraints wrapped in XML tag.
    """
    if artifact_types is None:
        artifact_types = DEFAULT_ARTIFACT_TYPES

    artifact_types = [t.lower() for t in artifact_types]

    sections = []
    for artifact_type in artifact_types:
        if artifact_type in _ARTIFACT_SCOPE:
            sections.append(_ARTIFACT_SCOPE[artifact_type]())

    content = "\n\n".join(sections)
    return f"""<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

{content}
</artifact_executor_scope>"""
