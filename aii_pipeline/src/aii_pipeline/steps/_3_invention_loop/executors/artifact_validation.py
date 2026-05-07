"""Artifact validation — schema validation for artifact output files.

Validates that agent-produced output files conform to the expected schema
for each artifact type (experiment, dataset, evaluation, proof, research).

Called by:
- ``pod_entrypoint.py`` (worker pod, runpod mode) — after agent completes
- ``exec_mode_router.py`` (local mode) — after agent completes

File-size checks are chained on via
:func:`aii_lib.agent_backend.utils.make_file_size_validator` (see
``make_post_validator`` below) and run inside the agent's own
``post_validate`` retry loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _get_validation_fns(artifact_type: str) -> tuple:
    """Import and return (verify_fn, retry_fn, get_expected_files) for artifact type.

    Returns:
        Tuple of (verify_fn, retry_fn, expected_files).
        For dataset: expected_files is None (uses file_paths from structured output).
    """
    if artifact_type == "experiment":
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.out_schema import (
            ExperimentArtifact,
            verify_experiment_output,
        )
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.u_prompt import (
            build_experiment_retry_prompt,
        )

        return (
            verify_experiment_output,
            build_experiment_retry_prompt,
            ExperimentArtifact.get_expected_out_files(),
        )

    if artifact_type == "dataset":
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.out_schema import (
            verify_dataset_output,
        )
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.u_prompt import (
            build_dataset_retry_prompt,
        )

        # Dataset uses file_paths from structured output, not expected_files
        return verify_dataset_output, build_dataset_retry_prompt, None

    if artifact_type == "evaluation":
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.out_schema import (
            EvaluationArtifact,
            verify_evaluation_output,
        )
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.u_prompt import (
            build_evaluation_retry_prompt,
        )

        return (
            verify_evaluation_output,
            build_evaluation_retry_prompt,
            EvaluationArtifact.get_expected_out_files(),
        )

    if artifact_type == "proof":
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.out_schema import (
            ProofArtifact,
            verify_proof_output,
        )
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.u_prompt import (
            build_proof_retry_prompt,
        )

        return (
            verify_proof_output,
            build_proof_retry_prompt,
            ProofArtifact.get_expected_out_files(),
        )

    if artifact_type == "research":
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.out_schema import (
            ResearchArtifact,
            verify_research_output,
        )
        from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.u_prompt import (
            build_research_retry_prompt,
        )

        return (
            verify_research_output,
            build_research_retry_prompt,
            ResearchArtifact.get_expected_out_files(),
        )

    raise ValueError(f"Unknown artifact type for validation: {artifact_type!r}")


def _extract_file_paths_from_structured_output(
    structured_output: dict | None,
) -> list[str]:
    """Extract file paths from structured output (dataset-specific).

    Same logic as dataset executor's ``_extract_structured_output``.
    """
    if not structured_output:
        return []
    from aii_lib.agent_backend import Agent

    out_expected_files = structured_output.get("out_expected_files", {})
    return Agent._collect_paths_recursive(out_expected_files)


def make_post_validator(
    artifact_type: str,
    workspace_dir: str | Path,
    min_examples: int = 3,
    max_file_size_mb: float = 100,
) -> Callable:
    """Create a post_validate closure for AgentOptions.

    Chains schema validation + file size validation.
    Returns a function compatible with ``AgentOptions.post_validate``:
    ``fn(structured_output) -> (valid, retry_prompt | None)``.

    The closure checks files on disk in workspace_dir using the
    artifact-type-specific verify/retry functions.
    """
    workspace_dir = Path(workspace_dir)
    verify_fn, retry_fn, expected_files = _get_validation_fns(artifact_type)
    _attempt = [0]  # mutable counter inside closure

    def validate(structured_output):
        _attempt[0] += 1

        if artifact_type == "dataset":
            file_paths = _extract_file_paths_from_structured_output(structured_output)
            verification = verify_fn(
                workspace_dir=workspace_dir,
                file_paths=file_paths,
                min_examples=min_examples,
            )
        elif artifact_type in ("proof", "research"):
            verification = verify_fn(
                workspace_dir=workspace_dir,
                expected_files=expected_files,
            )
        else:
            verification = verify_fn(
                workspace_dir=workspace_dir,
                expected_files=expected_files,
                min_examples=min_examples,
            )

        if verification.get("valid", False):
            return True, None

        retry_prompt = retry_fn(
            verification=verification,
            attempt=_attempt[0],
            max_attempts=10,
        )
        return False, retry_prompt

    # Chain with file size validation
    from aii_lib.agent_backend.utils import chain_validators, make_file_size_validator

    file_size_validator = make_file_size_validator(workspace_dir, max_file_size_mb)
    return chain_validators(validate, file_size_validator)
