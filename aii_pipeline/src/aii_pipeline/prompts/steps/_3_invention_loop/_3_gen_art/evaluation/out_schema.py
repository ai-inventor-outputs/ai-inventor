"""Schema for evaluation artifact.

Evaluation artifacts assess experimental results with metrics.
Uses Claude agent with aii-handbook-multi-llm-agents and aii-json skills.

Includes verification logic for post-execution validation.
"""

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.prompts import LLMPrompt, LLMStructOut
from pydantic import Field

from ..out_schema import ArtifactType, BaseArtifact, BaseExpectedFiles

# =============================================================================
# SCHEMAS
# =============================================================================


class EvaluationExpectedFiles(BaseExpectedFiles):
    """All expected output files from evaluation artifact."""

    script: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to eval.py script. Example: 'eval.py'"
    )
    full_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Full evaluation JSON file. Example: 'full_eval_out.json'"
    )
    mini_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Mini evaluation JSON file. Example: 'mini_eval_out.json'"
    )
    preview_output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Preview evaluation JSON file. Example: 'preview_eval_out.json'"
    )


class EvaluationArtifact(BaseArtifact):
    """Evaluation artifact — structured output + file metadata.

    Evaluates both proposed and baseline methods with appropriate metrics.
    Produces eval.py and eval_out.json files.
    """

    kind: Literal["evaluation_artifact"] = "evaluation_artifact"
    type: Annotated[Literal[ArtifactType.EVALUATION], LLMPrompt] = ArtifactType.EVALUATION
    out_expected_files: Annotated[EvaluationExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include eval.py script plus full/mini/preview evaluation JSON files."
    )
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default=[ExpectedFile("eval.py", "Evaluation script with metrics computation")],
        description="Primary file(s) to convert to demo formats",
    )

    @staticmethod
    def get_expected_out_files() -> list[ExpectedFile]:
        """All expected output files with descriptions. Used for dependency copying and verification."""
        return [
            ExpectedFile("eval.py", "Python evaluation script with metrics computation"),
            ExpectedFile(
                "full_eval_out.json",
                "Complete evaluation on full dataset (50+ examples)",
            ),
            ExpectedFile("mini_eval_out.json", "Evaluation on mini dataset (3 examples)"),
            ExpectedFile("preview_eval_out.json", "Evaluation preview (truncated)"),
        ]


# =============================================================================
# VERIFICATION
# =============================================================================

# Expected schema structure for evaluation output files
EVALUATION_SCHEMA = {
    "required_keys": ["metrics_agg", "datasets"],
    "dataset_entry_required_keys": ["dataset", "examples"],
    "example_required_keys": ["input", "output"],
    "metadata_prefix": "metadata_",
    "min_examples": 50,
}


def verify_evaluation_output(
    workspace_dir: Path,
    expected_files: list[str] | list[ExpectedFile] | None = None,
    min_examples: int = 50,
) -> dict:
    """Verify evaluation output files against schema and content requirements.

    Args:
        workspace_dir: Path to workspace directory
        expected_files: List of expected files (strings or ExpectedFile objects)
        min_examples: Minimum expected examples in full_eval_out.json

    Returns dict with:
    - valid: bool - True if all checks pass
    - file_errors: list - Missing/unreadable files
    - schema_errors: list - Schema validation errors
    - content_warnings: list - Content quality warnings
    - files_found: dict - Info about each file found
    - example_count: int - Number of examples in full_eval_out.json
    - metrics_agg: dict - Aggregated metrics (if found)
    """
    workspace = Path(workspace_dir)

    if expected_files is None:
        expected_files = EvaluationArtifact.get_expected_out_files()

    # Extract paths from ExpectedFile objects if needed
    file_paths = [f.path if isinstance(f, ExpectedFile) else f for f in expected_files]

    file_errors: list[str] = []
    schema_errors: list[str] = []
    content_warnings: list[str] = []
    files_found: dict[str, dict] = {}
    example_count = 0
    metrics_agg = {}

    for filename in file_paths:
        file_path = workspace / filename

        if not file_path.exists():
            file_errors.append(f"Missing file: {filename}")
            continue

        files_found[filename] = {"exists": True, "path": str(file_path)}

        if filename.endswith(".json"):
            json_result = _validate_evaluation_json(
                file_path=file_path,
                filename=filename,
                min_examples=min_examples,
            )
            schema_errors.extend(json_result.get("schema_errors", []))
            content_warnings.extend(json_result.get("content_warnings", []))
            files_found[filename].update(json_result.get("file_info", {}))

            if filename == "full_eval_out.json":
                example_count = max(example_count, json_result.get("example_count", 0))
                if json_result.get("metrics_agg"):
                    metrics_agg = json_result["metrics_agg"]

        elif filename.endswith(".py"):
            try:
                content = file_path.read_text(encoding="utf-8")
                if len(content.strip()) < 100:
                    content_warnings.append(f"{filename} is very short ({len(content)} chars)")
                files_found[filename]["size"] = len(content)
                try:
                    compile(content, filename, "exec")
                except SyntaxError as e:
                    schema_errors.append(
                        f"{filename}: Python syntax error at line {e.lineno}: {e.msg}"
                    )
            except Exception as e:
                file_errors.append(f"Cannot read {filename}: {e}")

    valid = not file_errors and not schema_errors

    return {
        "valid": valid,
        "file_errors": file_errors,
        "schema_errors": schema_errors,
        "content_warnings": content_warnings,
        "files_found": files_found,
        "example_count": example_count,
        "metrics_agg": metrics_agg,
    }


def _validate_evaluation_json(
    file_path: Path,
    filename: str,
    min_examples: int = 50,
) -> dict:
    """Validate a single evaluation JSON file against datasets-grouped schema.

    Expected structure:
    {
      "metrics_agg": {"accuracy": 0.85, ...},
      "datasets": [
        {
          "dataset": "name",
          "examples": [
            {"input": "...", "output": "...", "metadata_fold": 2, "predict_baseline": "...", "eval_accuracy": 0.9, ...}
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
        "metrics_agg": {},
    }

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

    if not isinstance(data, dict):
        result["schema_errors"].append(f"{filename}: Root must be an object")
        return result

    # Check metrics_agg
    if "metrics_agg" not in data:
        result["schema_errors"].append(f"{filename}: Missing required 'metrics_agg' key")
    else:
        metrics_agg = data["metrics_agg"]
        if not isinstance(metrics_agg, dict):
            result["schema_errors"].append(f"{filename}: 'metrics_agg' must be an object")
        elif not metrics_agg:
            result["content_warnings"].append(
                f"{filename}: 'metrics_agg' is empty (no aggregate metrics)"
            )
        else:
            result["metrics_agg"] = metrics_agg

    # Check datasets
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
    eval_metric_pattern = re.compile(r"^eval_[a-zA-Z_][a-zA-Z0-9_]*$")
    any_eval_metric = False

    for ds_idx, ds_entry in enumerate(datasets):
        if not isinstance(ds_entry, dict):
            result["schema_errors"].append(f"{filename}: datasets[{ds_idx}] must be an object")
            continue

        for key in EVALUATION_SCHEMA["dataset_entry_required_keys"]:
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

            for key in EVALUATION_SCHEMA["example_required_keys"]:
                if key not in example:
                    result["schema_errors"].append(
                        f"{filename}: '{ds_name}' example {i} missing required '{key}' field"
                    )

            # Track eval_* metrics
            eval_metrics = [k for k in example if eval_metric_pattern.match(k)]
            if eval_metrics:
                any_eval_metric = True

    result["example_count"] = total_examples
    result["file_info"]["example_count"] = total_examples
    result["file_info"]["dataset_count"] = len(datasets)

    if filename == "full_eval_out.json" and total_examples < min_examples:
        result["content_warnings"].append(
            f"{filename}: Only {total_examples} total examples (expected at least {min_examples})"
        )

    if not any_eval_metric:
        result["schema_errors"].append(
            f"{filename}: No eval_* metrics found in any of the sampled examples (at least one required)"
        )

    return result
