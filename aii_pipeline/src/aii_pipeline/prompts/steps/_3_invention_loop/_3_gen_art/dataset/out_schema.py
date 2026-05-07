"""Schema for dataset artifact.

Dataset artifacts find and prepare datasets from HuggingFace and Our World in Data.
Uses Claude agent with aii-hf-datasets, aii-owid-datasets, and aii-json skills.

Includes verification logic for post-execution validation.
"""

import json
from pathlib import Path
from typing import Annotated, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.prompts import LLMPrompt, LLMStructOut
from pydantic import Field

from ..out_schema import ArtifactType, BaseArtifact, BaseExpectedFiles

# =============================================================================
# SCHEMAS
# =============================================================================


class DatasetFileSet(BaseExpectedFiles):
    """One dataset's three required output variants."""

    full: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="Full dataset JSON file(s). Single file or split files. Example: ['full_data_out.json'] or ['full_data_out/full_data_out_1.json', 'full_data_out/full_data_out_2.json']"
    )
    mini: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Mini dataset JSON file path (3 examples). Example: 'mini_data_out.json'"
    )
    preview: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Preview dataset JSON file path (10 examples). Example: 'preview_data_out.json'"
    )


class DatasetExpectedFiles(BaseExpectedFiles):
    """All expected output files from dataset artifact."""

    script: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to data.py script. Example: 'data.py'"
    )
    datasets: Annotated[list[DatasetFileSet], LLMPrompt, LLMStructOut] = Field(
        description="Dataset file groups — one per dataset, each with full/mini/preview variants"
    )


class DatasetArtifact(BaseArtifact):
    """Dataset artifact — structured output + file metadata.

    Finds, evaluates, and prepares datasets for research experiments.
    Produces data.py and full_data_out.json files.
    """

    kind: Literal["dataset_artifact"] = "dataset_artifact"
    type: Annotated[Literal[ArtifactType.DATASET], LLMPrompt] = ArtifactType.DATASET
    out_expected_files: Annotated[DatasetExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include data.py script plus dataset file groups (full/mini/preview variants)."
    )
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default=[ExpectedFile("data.py", "Dataset fetching/generation script")],
        description="Primary file(s) to convert to demo formats",
    )

    @staticmethod
    def get_expected_out_files() -> list[ExpectedFile]:
        """All expected output files with descriptions. Used for dependency copying and verification."""
        return [
            ExpectedFile("data.py", "Python script to fetch/generate the dataset"),
            ExpectedFile("full_data_out.json", "Complete dataset as JSON with 'examples' array"),
            ExpectedFile("preview_data_out.json", "First 10 examples for preview"),
            ExpectedFile("mini_data_out.json", "First 3 examples for quick inspection"),
        ]


# =============================================================================
# VERIFICATION
# =============================================================================

# Expected schema structure for dataset files
DATASET_SCHEMA = {
    "required_keys": ["datasets"],
    "dataset_entry_required_keys": ["dataset", "examples"],
    "example_required_keys": ["input", "output"],
    "metadata_prefix": "metadata_",
    "min_examples": 50,  # Minimum expected examples (total across all datasets)
}


def verify_dataset_output(
    workspace_dir: Path,
    file_paths: list[str],
    min_examples: int = 50,
) -> dict:
    """Verify dataset output files against schema and content requirements.

    Uses file paths reported by the agent (from structured output). All paths
    are relative to workspace_dir and must resolve inside it.

    Args:
        workspace_dir: Path to workspace directory (agent CWD)
        file_paths: List of file paths reported by agent (relative to workspace_dir)
        min_examples: Minimum expected examples in full_data_out.json

    Returns dict with:
    - valid: bool - True if all checks pass
    - file_errors: list - Missing/out-of-bounds/unreadable files
    - schema_errors: list - Schema validation errors
    - content_warnings: list - Content quality warnings (empty fields, etc.)
    - files_found: dict - Info about each file found
    - example_count: int - Number of examples in full_data_out.json
    """
    workspace = Path(workspace_dir).resolve()

    file_errors: list[str] = []
    schema_errors: list[str] = []
    content_warnings: list[str] = []
    files_found: dict[str, dict] = {}
    example_count = 0

    for rel_path in file_paths:
        file_path = (workspace / rel_path).resolve()

        # Security: ensure path is inside workspace
        if not str(file_path).startswith(str(workspace)):
            file_errors.append(f"Path escapes workspace: {rel_path}")
            continue

        if not file_path.exists():
            file_errors.append(f"Missing file: {rel_path}")
            continue

        files_found[rel_path] = {"exists": True, "path": str(file_path)}

        # For JSON files, validate structure
        if rel_path.endswith(".json"):
            json_result = _validate_json_file(
                file_path=file_path,
                filename=rel_path,
                min_examples=min_examples,
            )
            schema_errors.extend(json_result.get("schema_errors", []))
            content_warnings.extend(json_result.get("content_warnings", []))
            files_found[rel_path].update(json_result.get("file_info", {}))

            # Track example count from full_data_out files (total across all datasets)
            if "full_data_out" in rel_path:
                example_count += json_result.get("example_count", 0)

        # For Python files, just check they're non-empty
        elif rel_path.endswith(".py"):
            try:
                content = file_path.read_text(encoding="utf-8")
                if len(content.strip()) < 50:
                    content_warnings.append(f"{rel_path} is very short ({len(content)} chars)")
                files_found[rel_path]["size"] = len(content)
            except Exception as e:
                file_errors.append(f"Cannot read {rel_path}: {e}")

    # Overall validity
    valid = not file_errors and not schema_errors

    return {
        "valid": valid,
        "file_errors": file_errors,
        "schema_errors": schema_errors,
        "content_warnings": content_warnings,
        "files_found": files_found,
        "example_count": example_count,
    }


def _validate_json_file(
    file_path: Path,
    filename: str,
    min_examples: int = 50,
) -> dict:
    """Validate a single JSON file against datasets-grouped schema.

    Expected structure:
    {
      "datasets": [
        {
          "dataset": "name",
          "examples": [
            {"input": "...", "output": "...", "metadata_fold": 2, ...}
          ]
        }
      ]
    }

    Returns dict with schema_errors, content_warnings, file_info, and example_count.
    """
    result = {
        "schema_errors": [],
        "content_warnings": [],
        "file_info": {},
        "example_count": 0,
    }

    # Try to parse JSON
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        result["file_info"]["size"] = len(content)
    except json.JSONDecodeError as e:
        result["schema_errors"].append(f"{filename}: Invalid JSON - {e}")
        return result
    except Exception as e:
        result["schema_errors"].append(f"{filename}: Cannot read - {e}")
        return result

    # Check root is object
    if not isinstance(data, dict):
        result["schema_errors"].append(
            f"{filename}: Root must be an object, got {type(data).__name__}"
        )
        return result

    # Check for 'datasets' key
    if "datasets" not in data:
        result["schema_errors"].append(f"{filename}: Missing required 'datasets' key")
        return result

    datasets = data["datasets"]
    if not isinstance(datasets, list):
        result["schema_errors"].append(f"{filename}: 'datasets' must be an array")
        return result

    if not datasets:
        result["schema_errors"].append(f"{filename}: 'datasets' array is empty")
        return result

    # Validate each dataset entry and count total examples
    total_examples = 0
    for ds_idx, ds_entry in enumerate(datasets):
        if not isinstance(ds_entry, dict):
            result["schema_errors"].append(f"{filename}: datasets[{ds_idx}] must be an object")
            continue

        # Check required dataset-level keys
        for key in DATASET_SCHEMA["dataset_entry_required_keys"]:
            if key not in ds_entry:
                result["schema_errors"].append(
                    f"{filename}: datasets[{ds_idx}] missing required '{key}' field"
                )

        ds_name = ds_entry.get("dataset", f"dataset_{ds_idx}")
        examples = ds_entry.get("examples", [])

        if not isinstance(examples, list):
            result["schema_errors"].append(
                f"{filename}: datasets[{ds_idx}] ('{ds_name}') 'examples' must be an array"
            )
            continue

        total_examples += len(examples)

        # Validate sample of examples from this dataset (first 3 per dataset)
        sample_size = min(3, len(examples))
        for i, example in enumerate(examples[:sample_size]):
            if not isinstance(example, dict):
                result["schema_errors"].append(
                    f"{filename}: '{ds_name}' example {i} must be an object"
                )
                continue

            # Check required example keys
            for key in DATASET_SCHEMA["example_required_keys"]:
                if key not in example:
                    result["schema_errors"].append(
                        f"{filename}: '{ds_name}' example {i} missing required '{key}' field"
                    )

            # Check for empty input/output (content warning, not error)
            if not str(example.get("input", "")).strip():
                result["content_warnings"].append(
                    f"{filename}: '{ds_name}' example {i} has empty 'input'"
                )
            if not str(example.get("output", "")).strip():
                result["content_warnings"].append(
                    f"{filename}: '{ds_name}' example {i} has empty 'output'"
                )

    result["example_count"] = total_examples
    result["file_info"]["example_count"] = total_examples
    result["file_info"]["dataset_count"] = len(datasets)

    # Check total example count (only for full_data_out.json)
    if filename == "full_data_out.json" and total_examples < min_examples:
        result["content_warnings"].append(
            f"{filename}: Only {total_examples} total examples across {len(datasets)} datasets (expected at least {min_examples})"
        )

    return result
