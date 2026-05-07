#!/usr/bin/env python
"""
JSON Schema Validator for Multi-Agent Systems Pipeline

Validates JSON files against predefined schemas for data/method/eval outputs.

Usage:
    python aii_json_validate_schema.py --format exp_eval_sol_out --file /path/to/eval_out.json
"""

import argparse
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_json__validate"
DEFAULT_TIMEOUT = 60.0

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
AVAILABLE_FORMATS = {
    "exp_sel_data_out": "exp_sel_data_out.json",
    "exp_gen_sol_out": "exp_gen_sol_out.json",
    "exp_eval_sol_out": "exp_eval_sol_out.json",
    "exp_proof_out": "exp_proof_out.json",
}


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_json_validate():
    """Initialize JSON validation environment with warmup."""
    import json

    from jsonschema import validate

    # Warmup: load actual schema and validate a minimal instance
    try:
        schema_path = SCHEMAS_DIR / "exp_gen_sol_out.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            validate(
                instance={
                    "datasets": [{"dataset": "d", "examples": [{"input": "x", "output": "x"}]}]
                },
                schema=schema,
            )
    except Exception:
        pass


@aii_ability(
    name="aii_json__validate",
    description="Validate a JSON file against a predefined schema for pipeline outputs.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_json_validate",
)
def core_json_validate(
    format_type: str = "",
    file_path: str = "",
    strict: bool = False,
    workspace_dir: str = "",
) -> dict:
    """
    Validate a JSON file against a schema.

    Args:
        format_type: Schema format type (e.g., "exp_eval_sol_out")
        file_path: Path to JSON file to validate
        strict: Treat warnings as errors

    Returns:
        Dict with success, errors, and warnings
    """
    import json

    from jsonschema import SchemaError, ValidationError, validate

    def load_schema(format_type: str) -> dict | None:
        schema_file = SCHEMAS_DIR / AVAILABLE_FORMATS[format_type]
        try:
            with open(schema_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def load_json_file(file_path: str) -> dict | None:
        try:
            with open(file_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def validate_format(data: dict, schema: dict) -> tuple:
        errors = []
        try:
            validate(instance=data, schema=schema)
            return True, []
        except ValidationError as e:
            error_path = (
                " -> ".join([str(p) for p in e.absolute_path]) if e.absolute_path else "root"
            )
            errors.append(f"Path: {error_path}")
            errors.append(f"Error: {e.message}")
            if e.validator:
                errors.append(f"Validator: {e.validator}")
            return False, errors
        except SchemaError as e:
            errors.append(f"Schema error: {e.message}")
            return False, errors

    def check_additional_requirements(data: dict, format_type: str) -> tuple:
        warnings = []

        if format_type == "sel_hypo_out":
            ideas = data.get("ideas", [])
            if not isinstance(ideas, list) or len(ideas) == 0:
                warnings.append("Warning: No ideas found")
                return len(warnings) == 0, warnings

            selected_count = sum(
                1 for idea in ideas if isinstance(idea, dict) and idea.get("selected", False)
            )
            if selected_count == 0:
                warnings.append("Warning: No ideas were selected (all rejected)")

            for i, idea in enumerate(ideas):
                if not isinstance(idea, dict):
                    continue
                if not idea.get("title", "").strip():
                    warnings.append(f"Warning: Idea {i} has empty 'title' field")
                if not idea.get("hypothesis", "").strip():
                    warnings.append(f"Warning: Idea {i} has empty 'hypothesis' field")

        elif format_type == "exp_sel_data_out":
            datasets = data.get("datasets", [])
            if not isinstance(datasets, list) or len(datasets) == 0:
                warnings.append("Warning: No datasets found")
                return len(warnings) == 0, warnings

            for ds_entry in datasets:
                if not isinstance(ds_entry, dict):
                    continue
                ds_name = ds_entry.get("dataset", "unknown")
                examples = ds_entry.get("examples", [])
                if not isinstance(examples, list):
                    continue
                for i, example in enumerate(examples[:5]):
                    if not isinstance(example, dict):
                        continue
                    if not example.get("input", "").strip():
                        warnings.append(f"Warning: '{ds_name}' example {i} has empty 'input' field")
                    if not example.get("output", "").strip():
                        warnings.append(
                            f"Warning: '{ds_name}' example {i} has empty 'output' field"
                        )

        elif format_type == "exp_gen_sol_out":
            datasets = data.get("datasets", [])
            if not isinstance(datasets, list):
                return len(warnings) == 0, warnings

            for ds_entry in datasets:
                if not isinstance(ds_entry, dict):
                    continue
                ds_name = ds_entry.get("dataset", "unknown")
                examples = ds_entry.get("examples", [])
                if not isinstance(examples, list):
                    continue
                for i, example in enumerate(examples[:5]):
                    if not isinstance(example, dict):
                        continue
                    predict_fields = [k for k in example if k.startswith("predict_")]
                    if not predict_fields:
                        warnings.append(
                            f"Warning: '{ds_name}' example {i} has no prediction fields (predict_* fields)"
                        )
                    else:
                        for field in predict_fields:
                            if not str(example.get(field, "")).strip():
                                warnings.append(
                                    f"Warning: '{ds_name}' example {i} has empty '{field}'"
                                )

        elif format_type == "exp_eval_sol_out":
            if not data.get("metrics_agg"):
                warnings.append("Warning: 'metrics_agg' is empty")

            datasets = data.get("datasets", [])
            if not isinstance(datasets, list):
                return len(warnings) == 0, warnings

            for ds_entry in datasets:
                if not isinstance(ds_entry, dict):
                    continue
                ds_name = ds_entry.get("dataset", "unknown")
                examples = ds_entry.get("examples", [])
                if not isinstance(examples, list):
                    continue
                for i, example in enumerate(examples[:5]):
                    if not isinstance(example, dict):
                        continue
                    predict_fields = [k for k in example if k.startswith("predict_")]
                    if not predict_fields:
                        warnings.append(
                            f"Warning: '{ds_name}' example {i} has no prediction fields (predict_* fields)"
                        )
                    eval_metrics = [k for k in example if k.startswith("eval_")]
                    if not eval_metrics:
                        warnings.append(
                            f"Warning: '{ds_name}' example {i} has no evaluation metrics (eval_* fields)"
                        )

        elif format_type == "exp_proof_out":
            if not data.get("lean_code", "").strip():
                warnings.append("Warning: 'lean_code' is empty")
            elif "sorry" in data.get("lean_code", "").lower():
                warnings.append("Warning: 'lean_code' contains 'sorry' (incomplete proof)")

            if not data.get("proof_explanation", "").strip():
                warnings.append("Warning: 'proof_explanation' is empty")

            lemmas = data.get("lemmas", [])
            if isinstance(lemmas, list):
                for i, lemma in enumerate(lemmas):
                    if not isinstance(lemma, dict):
                        continue
                    if not lemma.get("name", "").strip():
                        warnings.append(f"Warning: Lemma {i} has empty 'name'")
                    if not lemma.get("statement", "").strip():
                        warnings.append(f"Warning: Lemma {i} has empty 'statement'")

        return len(warnings) == 0, warnings

    # Validate file_path
    if not file_path:
        return {"success": False, "error": "file_path is required"}
    _project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    # If the caller passed a relative path AND a workspace_dir, resolve
    # the path against the workspace. Without this, ``Path(...).resolve()``
    # uses the *server*'s CWD (typically ``/ai-inventor/aii_server``) and
    # silently looks for the file in the wrong place — yielding the
    # confusing "Could not load JSON file" error agents kept hitting.
    _path = Path(file_path)
    if not _path.is_absolute() and workspace_dir:
        _path = Path(workspace_dir) / _path
    _resolved = _path.resolve()
    if not any(_resolved == d or d in _resolved.parents for d in [_project_root, Path("/tmp")]):
        return {
            "success": False,
            "error": "file_path must be under the project directory or /tmp",
        }

    # Validate format type
    if format_type not in AVAILABLE_FORMATS:
        return {"success": False, "error": f"Unknown format: {format_type}"}

    # Load schema
    schema = load_schema(format_type)
    if schema is None:
        return {"success": False, "error": f"Could not load schema for {format_type}"}

    # Load JSON file. Pass the resolved absolute path so a relative
    # ``file_path`` from the agent (whose CWD differs from the server's)
    # still finds the file — the security check above already confirmed
    # the resolved location is inside the project / /tmp.
    data = load_json_file(str(_resolved))
    if data is None:
        return {"success": False, "error": f"Could not load JSON file: {_resolved}"}

    # Validate against schema
    is_valid, errors = validate_format(data, schema)

    # Check additional requirements
    _has_no_warnings, warnings = check_additional_requirements(data, format_type)

    # Determine overall success
    if not is_valid or (warnings and strict):
        success = False
    else:
        success = True

    return {
        "success": success,
        "is_valid": is_valid,
        "format": format_type,
        "file": file_path,
        "errors": errors,
        "warnings": warnings,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSON files against Multi-Agent Systems pipeline schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aii_json_validate_schema.py --format exp_sel_data_out --file /path/to/full_data_out.json
  python aii_json_validate_schema.py --format exp_eval_sol_out --file /path/to/eval_out.json --strict
        """,
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=list(AVAILABLE_FORMATS.keys()),
        help="Output format type",
    )
    parser.add_argument("--file", type=str, required=True, help="Path to JSON file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "format_type": args.format,
            "file_path": args.file,
            "strict": args.strict,
        },
        timeout=DEFAULT_TIMEOUT,
    )

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Format: {result.get('format', args.format)}")

    if result.get("is_valid", False):
        print("Validation PASSED")
    else:
        print("Validation FAILED")

    if result.get("errors"):
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  {error}")

    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  {warning}")

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
