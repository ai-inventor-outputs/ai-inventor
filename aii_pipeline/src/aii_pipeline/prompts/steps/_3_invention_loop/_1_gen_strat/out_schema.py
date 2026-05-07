"""Schemas for strategy generation — pool objects and LLM output.

Schemas:
- ArtifactDirection: High-level direction (id is code-assigned, excluded from LLM)
- Strategy: Research strategy (id is code-assigned, excluded from LLM)
- Strategies: Top-level wrapper for structured output

Utility:
- assign_artifact_direction_ids: Assigns unique IDs to artifact directions after LLM output
- validate_artifact_dependencies: Validates dependencies exist and follow type rules
- DEPENDENCY_RULES: Mapping of artifact type -> (required_dep_types, optional_dep_types)
"""

from typing import Annotated, Literal

from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from pydantic import Field

# =============================================================================
# SCHEMAS
# =============================================================================


class ArtifactDep(LLMPromptModel, LLMStructOutModel):
    """A single dependency on an existing artifact, with a short type label.

    ``id`` and ``label`` are LLM-generated at strategy time. ``label`` is free-text but
    short — a word or two naming the type of dependency, not a sentence.

    ``relation_type`` and ``relation_rationale`` are populated later, in upd_hypo,
    using the MultiCite citation-function typology (Lauscher et al., NAACL 2022).
    They are absent at strategy time and may stay absent for legacy runs.
    """

    id: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="ID of an existing artifact this artifact depends on"
    )
    label: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Short free-text label naming the type of this dependency (a word or two, not a sentence)"
    )
    relation_type: str | None = Field(
        default=None,
        description="MultiCite type set in upd_hypo: background/motivation/uses/extends/similarities/differences",
    )
    relation_rationale: str | None = Field(
        default=None,
        description="Brief rationale (≤50 chars) set in upd_hypo",
    )


class ArtifactDirection(LLMPromptModel, LLMStructOutModel):
    """High-level direction for an artifact to execute this iteration.

    ID is code-assigned (LLMPrompt only — visible in prompts, not LLM-generated).
    """

    id: Annotated[str, LLMPrompt] = Field(
        default="",
        description="Unique artifact ID assigned by code (e.g., 'experiment_iter1_dir1')",
    )
    type: Annotated[
        Literal["experiment", "research", "proof", "evaluation", "dataset"],
        LLMPrompt,
        LLMStructOut,
    ] = Field(description="Type of artifact to create")
    objective: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What we want to achieve with this artifact"
    )
    approach: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="High-level direction/method"
    )
    depends_on: Annotated[list[ArtifactDep], LLMPrompt, LLMStructOut] = Field(
        default_factory=list,
        description="Existing artifacts this depends on, each with a short type label",
    )


class Strategy(LLMPromptModel, LLMStructOutModel):
    """A research strategy.

    Content fields have LLMPrompt + LLMStructOut markers.
    ``id`` is code-assigned (LLMPrompt only — visible in prompts, not LLM-generated).

    ID format: strat_it{iteration}__{model}_idx{N}
    """

    kind: Literal["strategy"] = "strategy"
    id: Annotated[str, LLMPrompt] = Field(
        default="", description="Unique strategy ID (e.g., strat_it1__opus_idx1)"
    )
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Short name for this strategy"
    )
    objective: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="The novel contribution we're building toward"
    )
    rationale: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Why this strategy is promising"
    )
    artifact_directions: Annotated[list[ArtifactDirection], LLMPrompt, LLMStructOut] = Field(
        description="Artifacts to execute THIS iteration"
    )
    expected_outcome: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="What we'll have after this iteration's artifacts complete"
    )
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        default="",
        description="Brief summary of the strategy and its expected contribution",
    )


class Strategies(LLMPromptModel, LLMStructOutModel):
    """Top-level wrapper for LLM strategy generation output."""

    kind: Literal["strategies"] = "strategies"
    """Discriminator for ``AnyOutput`` — see ``aii_pipeline/run/typed_union.py``.
    No LLMStructOut annotation: invisible to the LLM's structured-output
    schema, populated via the default at deserialize time. Required so
    instances can land on ``task.output: AnyOutput`` (a discriminated
    union by ``kind``) — used per-task by gen_strat_tasks to mirror the
    raw LLM response so replay synthesis can re-validate."""

    strategies: Annotated[list[Strategy], LLMPrompt, LLMStructOut] = Field(
        description="List of generated strategies"
    )


class GenStratOut(LLMPromptModel):
    """Aggregate output of gen_strat module with all strategies.

    Every Strategy produced across the parallel tasks in one iteration.
    Used as the typed payload for ``module_output(output=...)``; readers
    walk ``module.output.strategies``.
    """

    kind: Literal["gen_strat_out"] = "gen_strat_out"
    strategies: list[Strategy] = Field(default_factory=list)


# =============================================================================
# UTILITY: Assign artifact IDs
# =============================================================================


def assign_artifact_direction_ids(
    strategy: dict,
    seen_ids: set[str],
    iteration: int,
) -> dict:
    """Assign unique IDs to artifact directions.

    The LLM does not generate IDs - we assign them sequentially based on type.
    IDs are like 'experiment_iter1_dir1'.

    IMPORTANT: Mutates seen_ids — new IDs are added so subsequent calls
    (e.g., for other strategies in the same batch) won't generate duplicates.
    Callers should pass a working copy if they need the original set unchanged.

    Args:
        strategy: Single strategy dict from LLM output
        seen_ids: Mutable set of all known IDs (pool + previously assigned). Gets updated in place.
        iteration: Current iteration number for ID generation

    Returns:
        Updated strategy with assigned IDs
    """
    artifact_directions = strategy.get("artifact_directions", [])

    def _make_unique(base_id: str) -> str:
        """Make an ID unique by appending suffix if needed."""
        if base_id not in seen_ids:
            seen_ids.add(base_id)
            return base_id

        # Find unique suffix
        suffix_num = 1
        while True:
            new_id = f"{base_id}_{suffix_num}"
            if new_id not in seen_ids:
                seen_ids.add(new_id)
                return new_id
            suffix_num += 1

    # Assign IDs
    for i, artifact in enumerate(artifact_directions):
        artifact_type = artifact.get("type", "unknown")
        # Create base ID: type_iter{N}_dir{M}
        base_id = f"{artifact_type}_iter{iteration}_dir{i + 1}"
        artifact["id"] = _make_unique(base_id)

    return strategy


# =============================================================================
# UTILITY: Validate artifact dependencies
# =============================================================================

# Dependency rules: artifact_type -> (required_dep_types, optional_dep_types)
# required_dep_types: Must have at least one dependency of these types
# optional_dep_types: Can have dependencies of these types (in addition to required)
DEPENDENCY_RULES: dict[str, tuple[set[str], set[str]]] = {
    "experiment": ({"dataset"}, {"research"}),  # REQUIRED: dataset, OPTIONAL: research
    "evaluation": (
        {"experiment"},
        {"dataset"},
    ),  # REQUIRED: experiment, OPTIONAL: dataset
    "dataset": (set(), {"research"}),  # REQUIRED: none, OPTIONAL: research
    "research": (set(), {"research"}),  # REQUIRED: none, OPTIONAL: research
    "proof": (set(), {"research"}),  # REQUIRED: none, OPTIONAL: research
}


def validate_artifact_dependencies(
    strategy: dict,
    artifact_pool_map: dict[str, str],
) -> tuple[bool, list[str]]:
    """Validate that artifact dependencies are valid and follow the rules.

    Checks:
    1. All depends_on IDs exist in the artifact pool
    2. Dependency types match the rules for each artifact type

    Args:
        strategy: Single strategy dict from LLM output
        artifact_pool_map: Dict mapping artifact ID -> artifact type for all existing artifacts

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []
    artifact_directions = strategy.get("artifact_directions", [])

    for artifact in artifact_directions:
        artifact_id = artifact.get("id", "unknown")
        artifact_type = artifact.get("type", "unknown")
        depends_on = artifact.get("depends_on", [])

        # Get rules for this artifact type
        required_types, optional_types = DEPENDENCY_RULES.get(artifact_type, (set(), set()))
        allowed_types = required_types | optional_types

        # Track which required types have been satisfied
        found_required_types: set[str] = set()

        for dep in depends_on:
            dep_id = dep.get("id", "") if isinstance(dep, dict) else getattr(dep, "id", "")
            dep_label = dep.get("label", "") if isinstance(dep, dict) else getattr(dep, "label", "")

            # Check 0: Label must be a non-empty short string.
            if not dep_label or not dep_label.strip():
                errors.append(
                    f"Artifact '{artifact_id}': dependency '{dep_id}' has empty label — every dep needs a short type label"
                )

            # Check 1: Does the dependency ID exist?
            if dep_id not in artifact_pool_map:
                errors.append(
                    f"Artifact '{artifact_id}': dependency '{dep_id}' does not exist in artifact pool"
                )
                continue

            # Check 2: Is the dependency type allowed?
            dep_type = artifact_pool_map[dep_id]
            if dep_type not in allowed_types:
                errors.append(
                    f"Artifact '{artifact_id}' ({artifact_type}): dependency '{dep_id}' has type '{dep_type}' "
                    f"which is not allowed (allowed: {allowed_types})"
                )
            elif dep_type in required_types:
                found_required_types.add(dep_type)

        # Check 3: Are required dependencies satisfied?
        # Skip if the pool has no artifacts of the required type (e.g., iteration 1: no datasets yet)
        pool_has_required = any(atype in required_types for atype in artifact_pool_map.values())
        if required_types and not found_required_types and pool_has_required:
            errors.append(
                f"Artifact '{artifact_id}' ({artifact_type}): missing required dependency. "
                f"Must have at least one dependency of type: {required_types}"
            )

    is_valid = not errors
    return is_valid, errors


# =============================================================================
# VERIFICATION: Verify strategies output
# =============================================================================


def verify_strategies(
    strategies: list[dict],
    num_expected: int,
    existing_artifact_ids: set[str],
    artifact_pool_map: dict[str, str],
    min_valid_artifacts: int = 1,
    allowed_artifacts: list[str] | None = None,
    art_limit: int | None = None,
) -> dict:
    """Verify strategies against rules.

    Returns dict with:
    - valid: bool - True if all checks pass (including min_valid_artifacts)
    - count_errors: list - Count mismatch errors
    - id_errors: list - Duplicate ID errors
    - dep_errors: list - Dependency errors
    - type_errors: list - Invalid artifact type errors
    - limit_errors: list - Artifact count exceeds art_limit
    - strategies_received: int - Number of strategies received
    - valid_artifact_count: int - Number of valid artifacts across all strategies
    - total_artifact_count: int - Total artifacts across all strategies
    - invalid_artifact_ids: set - IDs of invalid artifacts
    """
    # Default allowed types if not specified
    valid_types = (
        set(allowed_artifacts)
        if allowed_artifacts
        else {"research", "dataset", "proof", "experiment", "evaluation"}
    )

    count_errors = []
    id_errors = []
    dep_errors = []
    type_errors = []
    limit_errors = []

    # Track which artifact IDs are invalid (for counting valid artifacts)
    invalid_artifact_ids: set[str] = set()
    total_artifact_count = 0

    # Check strategy count
    if len(strategies) != num_expected:
        count_errors.append(f"Expected {num_expected} strategies, got {len(strategies)}")

    # Check each strategy
    for s_idx, s_dict in enumerate(strategies):
        artifact_directions = s_dict.get("artifact_directions", [])

        # Check artifact count limit per strategy
        if art_limit is not None and len(artifact_directions) > art_limit:
            limit_errors.append(
                f"Strategy {s_idx + 1} ('{s_dict.get('title', 'Untitled')}'): "
                f"has {len(artifact_directions)} artifact directions but limit is {art_limit}"
            )

        # First pass: collect ALL artifact IDs in this strategy
        strategy_ids = set()
        seen_ids = set()  # For duplicate detection within strategy
        for a_dict in artifact_directions:
            artifact_id = a_dict.get("id", "")
            artifact_type = a_dict.get("type", "")
            total_artifact_count += 1

            # Check artifact type is allowed
            if artifact_type not in valid_types:
                type_errors.append(
                    f"Strategy {s_idx + 1}: artifact '{artifact_id}' has type '{artifact_type}' "
                    f"which is not in allowed_artifacts: {sorted(valid_types)}"
                )
                invalid_artifact_ids.add(artifact_id)

            if artifact_id:
                strategy_ids.add(artifact_id)

                # Check for duplicate within same strategy
                if artifact_id in seen_ids:
                    id_errors.append(
                        f"Strategy {s_idx + 1}: duplicate artifact ID '{artifact_id}' within same strategy"
                    )
                    invalid_artifact_ids.add(artifact_id)
                seen_ids.add(artifact_id)

                # Check for duplicate against existing pool
                if artifact_id in existing_artifact_ids:
                    id_errors.append(
                        f"Strategy {s_idx + 1}: artifact ID '{artifact_id}' already exists in artifact pool"
                    )
                    invalid_artifact_ids.add(artifact_id)

        # Second pass: check dependencies
        for a_dict in artifact_directions:
            artifact_id = a_dict.get("id", "")
            depends_on = a_dict.get("depends_on", [])

            for dep in depends_on:
                dep_id = dep.get("id", "") if isinstance(dep, dict) else getattr(dep, "id", "")
                dep_label = (
                    dep.get("label", "") if isinstance(dep, dict) else getattr(dep, "label", "")
                )

                # Every dep must have a non-empty short label.
                if not dep_label or not dep_label.strip():
                    dep_errors.append(
                        f"Strategy {s_idx + 1}: artifact '{artifact_id}' has dependency on '{dep_id}' with empty label "
                        f"— every dep needs a short type label (a word or two)"
                    )
                    invalid_artifact_ids.add(artifact_id)

                # Check if depends_on references another artifact in SAME strategy (invalid - they run in parallel)
                if dep_id in strategy_ids:
                    dep_errors.append(
                        f"Strategy {s_idx + 1}: artifact '{artifact_id}' depends on '{dep_id}' "
                        f"which is in the SAME strategy (artifacts run in parallel, can't depend on each other)"
                    )
                    invalid_artifact_ids.add(artifact_id)
                # Check if depends_on references an ID that doesn't exist anywhere
                elif dep_id not in existing_artifact_ids:
                    dep_errors.append(
                        f"Strategy {s_idx + 1}: artifact '{artifact_id}' depends on '{dep_id}' "
                        f"which does not exist in artifact pool"
                    )
                    invalid_artifact_ids.add(artifact_id)

        # Validate dependency TYPES against artifact pool rules (e.g., experiments require datasets)
        is_valid, errors = validate_artifact_dependencies(s_dict, artifact_pool_map)
        if not is_valid:
            for err in errors:
                dep_errors.append(f"Strategy {s_idx + 1}: {err}")
                # Mark artifact as invalid based on error message (extract artifact ID if possible)
                # The error format is: "Artifact 'artifact_id': ..."
                if "Artifact '" in err:
                    try:
                        art_id = err.split("Artifact '")[1].split("'")[0]
                        invalid_artifact_ids.add(art_id)
                    except (IndexError, ValueError):
                        pass

    # Count valid artifacts — use a counter, not set length, because duplicate IDs
    # map to multiple artifacts but only one set entry
    invalid_count = sum(
        1
        for s_dict in strategies
        for a_dict in s_dict.get("artifact_directions", [])
        if a_dict.get("id", "") in invalid_artifact_ids
    )
    valid_artifact_count = total_artifact_count - invalid_count

    # Check if we have enough valid artifacts
    has_enough_valid = valid_artifact_count >= min_valid_artifacts

    # Overall valid: no errors AND enough valid artifacts
    valid = (
        not count_errors
        and not id_errors
        and not dep_errors
        and not type_errors
        and not limit_errors
        and has_enough_valid
    )

    return {
        "valid": valid,
        "count_errors": count_errors,
        "id_errors": id_errors,
        "dep_errors": dep_errors,
        "type_errors": type_errors,
        "limit_errors": limit_errors,
        "strategies_received": len(strategies),
        "valid_artifact_count": valid_artifact_count,
        "total_artifact_count": total_artifact_count,
        "invalid_artifact_ids": invalid_artifact_ids,
    }
