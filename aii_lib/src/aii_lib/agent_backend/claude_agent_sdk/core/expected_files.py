"""Expected files validation for Agent structured output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aii_lib.run import emit

from .prompts import DIAG_PREFIX


def validate_expected_files(
    expected_files_field: str | None,
    cwd: str | None,
    prompt_results: list | None = None,
) -> tuple[bool, list[str]]:
    """Validate that expected files exist (structured output mode).

    Extracts file paths from the agent's structured output and validates
    each one exists inside the workspace.

    Returns:
        (all_exist, missing_files_list)
    """
    if not expected_files_field:
        return True, []

    cwd_path = Path(cwd).resolve() if cwd else Path.cwd()

    # Extract file paths from structured output
    file_paths = _extract_file_paths(prompt_results or [], expected_files_field)
    if not file_paths:
        detail = _diagnose_missing_paths(prompt_results, expected_files_field)
        return False, [f"{DIAG_PREFIX}{detail}"]

    missing = []
    for rel_path in file_paths:
        file_path = (cwd_path / rel_path).resolve()

        # Security: ensure path is inside workspace
        if not str(file_path).startswith(str(cwd_path)):
            missing.append(f"`{rel_path}` (escapes workspace)")
            continue

        if not file_path.exists():
            missing.append(f"`{rel_path}`")

    return len(missing) == 0, missing


def _diagnose_missing_paths(
    prompt_results: list | None,
    field_name: str,
) -> str:
    """Build specific diagnostic message about why no file paths were found."""
    so_keys = None
    field_value = None
    for result in reversed(prompt_results or []):
        if result.structured_output and isinstance(result.structured_output, dict):
            so_keys = list(result.structured_output.keys())
            field_value = result.structured_output.get(field_name)
            break
    if so_keys is None:
        return f"no structured output returned (field `{field_name}` expected)"
    if field_name not in (so_keys or []):
        return f"field `{field_name}` missing from structured output (got keys: {so_keys})"
    return f"field `{field_name}` is empty or contains no paths (value: {field_value!r})"


def _extract_file_paths(
    prompt_results: list,
    field_name: str,
) -> list[str]:
    """Extract file paths from the latest structured output in prompt_results."""
    for result in reversed(prompt_results):
        if result.structured_output:
            data = result.structured_output if isinstance(result.structured_output, dict) else {}
            field_value = data.get(field_name)
            if field_value is not None:
                return collect_paths_recursive(field_value)
    return []


def collect_paths_recursive(obj: Any) -> list[str]:
    """Recursively collect all string values from a nested structure."""
    paths: list[str] = []
    if isinstance(obj, str):
        paths.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            paths.extend(collect_paths_recursive(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            paths.extend(collect_paths_recursive(value))
    return paths


def get_expected_file_fields(
    output_format: dict | None,
    expected_files_field: str | None,
) -> str:
    """Extract expected file field names from output_format schema.

    Resolves $ref to $defs and returns field names with descriptions.
    """
    schema = (output_format or {}).get("schema", {})
    if not schema or not expected_files_field:
        return "(unknown — no output schema available)"

    # Find the expected files property in the schema
    ef_prop = schema.get("properties", {}).get(expected_files_field, {})

    # Resolve $ref if present (e.g., "$ref": "#/$defs/ResearchExpectedFiles")
    ref = ef_prop.get("$ref", "")
    if ref.startswith("#/$defs/"):
        def_name = ref.split("/")[-1]
        ef_prop = schema.get("$defs", {}).get(def_name, {})

    # Extract properties with descriptions
    properties = ef_prop.get("properties", {})
    if not properties:
        return "(unknown — no properties found in schema)"

    lines = []
    for prop_name, prop_schema in properties.items():
        desc = prop_schema.get("description", "")
        lines.append(f"- `{prop_name}`: {desc}" if desc else f"- `{prop_name}`")
    return "\n".join(lines)


async def validate_and_retry_expected_files(
    options: Any,
    prompt_results: list,
    execute_prompt_fn: Any,
) -> bool:
    """Validate expected files and retry if missing.

    Args:
        options: AgentOptions with expected_files_struct_out_field, cwd, etc.
        prompt_results: List of PromptResult (mutated — retries appended).
        execute_prompt_fn: async callable(prompt, with_output_format)
            that returns (result, sdk_options_cache). Provided by Agent.

    Returns:
        True if all files valid, False otherwise.
    """
    from loguru import logger

    from .prompts import DIAG_PREFIX, build_expected_files_feedback

    all_exist, missing = validate_expected_files(
        options.expected_files_struct_out_field,
        options.cwd,
        prompt_results,
    )
    files_retry_count = 0
    max_retries = options.max_expected_files_retries
    while not all_exist and files_retry_count < max_retries:
        files_retry_count += 1
        display = [m.removeprefix(DIAG_PREFIX) for m in missing]
        emit.status_public_warning(
            f"Expected files missing (retry {files_retry_count}/{max_retries}): {display}"
        )
        feedback = build_expected_files_feedback(
            missing,
            options.expected_files_struct_out_field,
            lambda: get_expected_file_fields(
                options.output_format, options.expected_files_struct_out_field
            ),
        )
        try:
            result, _ = await execute_prompt_fn(
                feedback,
                with_output_format=True,
            )
        except Exception as e:
            logger.error(f"Expected files retry failed ({type(e).__name__})")
            emit.status_public_error(f"Expected files retry failed: {e}")
            break
        prompt_results.append(result)
        all_exist, missing = validate_expected_files(
            options.expected_files_struct_out_field,
            options.cwd,
            prompt_results,
        )
    if not all_exist:
        display = [m.removeprefix(DIAG_PREFIX) for m in missing]
        emit.status_public_warning(
            f"Expected files still missing after {files_retry_count} retries: {display}"
        )
        return False
    return True
