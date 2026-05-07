"""Schema for experiment artifact.

Experiment artifacts implement research methodology.
Uses Claude agent with aii-handbook-multi-llm-agents and aii-json skills.

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


class ExperimentExpectedFiles(BaseExpectedFiles):
    """All expected output files from experiment artifact."""

    script: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to method.py script. Example: 'method.py'"
    )
    full_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Full method output JSON file. Example: 'full_method_out.json'"
    )
    mini_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Mini method output JSON file. Example: 'mini_method_out.json'"
    )
    preview_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Preview method output JSON file. Example: 'preview_method_out.json'"
    )


class ExperimentArtifact(BaseArtifact):
    """Experiment artifact — structured output + file metadata.

    Implements research methodology with baseline comparison.
    Produces method.py and method_out.json files.
    """

    kind: Literal["experiment_artifact"] = "experiment_artifact"
    type: Annotated[Literal[ArtifactType.EXPERIMENT], LLMPrompt] = ArtifactType.EXPERIMENT
    out_expected_files: Annotated[ExperimentExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include method.py script plus full/mini/preview method output JSON files."
    )
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default=[ExpectedFile("method.py", "Research methodology implementation")],
        description="Primary file(s) to convert to demo formats",
    )

    @staticmethod
    def get_expected_out_files() -> list[ExpectedFile]:
        """All expected output files with descriptions. Used for dependency copying and verification."""
        return [
            ExpectedFile("method.py", "Python implementation of the research methodology"),
            ExpectedFile(
                "full_method_out.json",
                "Complete method output on full dataset (50+ examples)",
            ),
            ExpectedFile("mini_method_out.json", "Method output on mini dataset (3 examples)"),
            ExpectedFile("preview_method_out.json", "Method output preview (truncated)"),
        ]


# =============================================================================
# VERIFICATION
# =============================================================================

# Expected schema structure for experiment output files
EXPERIMENT_SCHEMA = {
    "required_keys": ["datasets"],
    "dataset_entry_required_keys": ["dataset", "examples"],
    "example_required_keys": ["input", "output"],
    "example_predict_prefix": "predict_",
    "metadata_prefix": "metadata_",
    "min_examples": 50,  # Minimum expected examples in full output (total across all datasets)
}


def verify_experiment_output(
    workspace_dir: Path,
    expected_files: list[str] | list[ExpectedFile] | None = None,
    min_examples: int = 50,
) -> dict:
    """Verify experiment output files against schema and content requirements.

    Args:
        workspace_dir: Path to workspace directory
        expected_files: List of expected files (strings or ExpectedFile objects)
        min_examples: Minimum expected examples in full_method_out.json

    Returns dict with:
    - valid: bool - True if all checks pass
    - file_errors: list - Missing/unreadable files
    - schema_errors: list - Schema validation errors
    - content_warnings: list - Content quality warnings
    - files_found: dict - Info about each file found
    - example_count: int - Number of examples in full_method_out.json

    Similar to verify_dataset_output for consistent retry patterns.
    """
    workspace = Path(workspace_dir)

    if expected_files is None:
        expected_files = ExperimentArtifact.get_expected_out_files()

    # Extract paths from ExpectedFile objects if needed
    file_paths = [f.path if isinstance(f, ExpectedFile) else f for f in expected_files]

    file_errors: list[str] = []
    schema_errors: list[str] = []
    content_warnings: list[str] = []
    files_found: dict[str, dict] = {}
    example_count = 0

    # Check each expected file (use extracted paths)
    for filename in file_paths:
        file_path = workspace / filename

        if not file_path.exists():
            file_errors.append(f"Missing file: {filename}")
            continue

        files_found[filename] = {"exists": True, "path": str(file_path)}

        # For JSON files, validate structure
        if filename.endswith(".json"):
            json_result = _validate_experiment_json(
                file_path=file_path,
                filename=filename,
                min_examples=min_examples,
            )
            schema_errors.extend(json_result.get("schema_errors", []))
            content_warnings.extend(json_result.get("content_warnings", []))
            files_found[filename].update(json_result.get("file_info", {}))

            # Track example count from full_method_out.json
            if filename == "full_method_out.json":
                example_count = max(example_count, json_result.get("example_count", 0))

        # For Python files, check they're non-empty and valid
        elif filename.endswith(".py"):
            try:
                content = file_path.read_text(encoding="utf-8")
                if len(content.strip()) < 100:
                    content_warnings.append(f"{filename} is very short ({len(content)} chars)")
                files_found[filename]["size"] = len(content)
                # Basic syntax check
                try:
                    compile(content, filename, "exec")
                except SyntaxError as e:
                    schema_errors.append(
                        f"{filename}: Python syntax error at line {e.lineno}: {e.msg}"
                    )
            except Exception as e:
                file_errors.append(f"Cannot read {filename}: {e}")

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


def _validate_experiment_json(
    file_path: Path,
    filename: str,
    min_examples: int = 50,
) -> dict:
    """Validate a single experiment JSON file against datasets-grouped schema.

    Expected structure:
    {
      "datasets": [
        {
          "dataset": "name",
          "examples": [
            {"input": "...", "output": "...", "metadata_fold": 2, "predict_baseline": "...", ...}
          ]
        }
      ]
    }
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

    # Check root
    if not isinstance(data, dict):
        result["schema_errors"].append(
            f"{filename}: Root must be an object, got {type(data).__name__}"
        )
        return result

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

    # Validate each dataset entry
    total_examples = 0
    any_predict = False

    for ds_idx, ds_entry in enumerate(datasets):
        if not isinstance(ds_entry, dict):
            result["schema_errors"].append(f"{filename}: datasets[{ds_idx}] must be an object")
            continue

        for key in EXPERIMENT_SCHEMA["dataset_entry_required_keys"]:
            if key not in ds_entry:
                result["schema_errors"].append(
                    f"{filename}: datasets[{ds_idx}] missing required '{key}' field"
                )

        ds_name = ds_entry.get("dataset", f"dataset_{ds_idx}")
        examples = ds_entry.get("examples", [])

        if not isinstance(examples, list):
            result["schema_errors"].append(f"{filename}: '{ds_name}' 'examples' must be an array")
            continue

        total_examples += len(examples)

        # Validate sample of examples (first 3 per dataset)
        sample_size = min(3, len(examples))
        for i, example in enumerate(examples[:sample_size]):
            if not isinstance(example, dict):
                result["schema_errors"].append(
                    f"{filename}: '{ds_name}' example {i} must be an object"
                )
                continue

            for key in EXPERIMENT_SCHEMA["example_required_keys"]:
                if key not in example:
                    result["schema_errors"].append(
                        f"{filename}: '{ds_name}' example {i} missing required '{key}' field"
                    )

            # Track predict_* fields
            predict_keys = [
                k for k in example if k.startswith(EXPERIMENT_SCHEMA["example_predict_prefix"])
            ]
            if predict_keys:
                any_predict = True
            for pk in predict_keys:
                if not str(example.get(pk, "")).strip():
                    result["content_warnings"].append(
                        f"{filename}: '{ds_name}' example {i} has empty '{pk}'"
                    )

    result["example_count"] = total_examples
    result["file_info"]["example_count"] = total_examples
    result["file_info"]["dataset_count"] = len(datasets)

    # Check total example count (only for full output file)
    if filename == "full_method_out.json" and total_examples < min_examples:
        result["content_warnings"].append(
            f"{filename}: Only {total_examples} total examples (expected at least {min_examples})"
        )

    if not any_predict:
        result["schema_errors"].append(
            f"{filename}: No predict_* fields found in any of the sampled examples (at least one required)"
        )

    return result
