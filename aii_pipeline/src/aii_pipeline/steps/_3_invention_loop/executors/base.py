"""Executor helpers — artifact-executor-specific utilities.

Generic agent utilities (build_options, end_task_*) live as
module-level helpers in aii_lib.agent_backend.utils. This module has
only the artifact-executor-specific helpers.

Usage::

    validation = build_validation("experiment", config)
    enrich_result(result_dict, result, plan, workspace_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_lib.agent_backend import AgentResponse

    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
    )
    from aii_pipeline.utils import PipelineConfig


VALIDATION_MODULE = "aii_pipeline.steps._3_invention_loop.executors.artifact_validation"


def build_validation(
    artifact_type: str,
    config: PipelineConfig,
    **overrides,
) -> dict:
    """Build validation config dict for create_and_run_agent.

    Args:
        artifact_type: One of experiment, dataset, evaluation, proof, research.
        config: Pipeline config.
        **overrides: Override any validation field.
    """
    exec_section = getattr(config.invention_loop.execute, artifact_type, None)
    val = {
        "module": VALIDATION_MODULE,
        "artifact_type": artifact_type,
        "schema_retries": getattr(exec_section, "schema_retries", 2) if exec_section else 2,
        "file_size_retries": 1,
        "max_file_size_mb": config.max_file_size_mb,
    }
    if exec_section and hasattr(exec_section, "min_examples"):
        val["min_examples"] = exec_section.min_examples
    val.update(overrides)
    return val


def enrich_result(
    result_dict: dict,
    result: AgentResponse,
    plan: BasePlan,
    workspace_dir: Path,
) -> dict:
    """Add common metadata + title/summary from structured output.

    Mutates and returns result_dict. Raises if no structured output.
    """
    result_dict["hypothesis"] = plan.title
    result_dict["workspace_path"] = str(workspace_dir)
    result_dict["example_count"] = 0

    title = ""
    summary = ""
    layman_summary = ""
    if result.structured_output and isinstance(result.structured_output, dict):
        title = result.structured_output.get("title", "")
        summary = result.structured_output.get("summary", "")
        layman_summary = result.structured_output.get("layman_summary", "")

    # File-based fallback: read structured_output.json from workspace if SDK output missing
    if not title and not summary:
        import json as _json

        _so_path = Path(workspace_dir) / "structured_output.json"
        if _so_path.exists():
            try:
                _so = _json.loads(_so_path.read_text(encoding="utf-8"))
                title = _so.get("title", "")
                summary = _so.get("summary", "")
                layman_summary = _so.get("layman_summary", "")
            except (OSError, ValueError) as _e:
                from loguru import logger as _log

                _log.debug(f"structured_output.json fallback read failed for {plan.id}: {_e}")

    if not title and not summary:
        raise RuntimeError(
            f"Executor produced no structured output for {plan.id} — cannot extract title/summary"
        )

    result_dict["title"] = title or plan.title.strip()
    result_dict["summary"] = summary
    result_dict["layman_summary"] = layman_summary
    return result_dict
