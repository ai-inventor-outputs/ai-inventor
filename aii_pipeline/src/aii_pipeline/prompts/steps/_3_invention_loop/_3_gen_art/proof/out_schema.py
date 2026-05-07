"""Schema for proof artifact.

Proof artifacts generate verified Lean 4 proofs.
Uses Claude agent with aii-lean skill for proof verification.

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


class ProofExpectedFiles(BaseExpectedFiles):
    """All expected output files from proof artifact."""

    proof_file: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to Lean 4 proof file. Example: 'proof.lean'"
    )
    output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to proof output JSON. Example: 'proof_out.json'"
    )


class ProofArtifact(BaseArtifact):
    """Proof artifact — structured output + file metadata.

    Generates formal mathematical proofs in Lean 4.
    Uses lemma-style proving with iterative refinement.
    Produces proof.lean and proof_out.json files.
    """

    kind: Literal["proof_artifact"] = "proof_artifact"
    type: Annotated[Literal[ArtifactType.PROOF], LLMPrompt] = ArtifactType.PROOF
    out_expected_files: Annotated[ProofExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include proof.lean and proof_out.json."
    )
    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default=[ExpectedFile("proof.lean", "Lean 4 formal proof")],
        description="Primary file(s) to convert to demo formats",
    )

    @staticmethod
    def get_expected_out_files() -> list[ExpectedFile]:
        """All expected output files with descriptions. Used for dependency copying and verification."""
        return [
            ExpectedFile("proof.lean", "Lean 4 proof code with theorem and proof"),
            ExpectedFile("proof_out.json", "Structured output with verification results"),
        ]


# =============================================================================
# VERIFICATION
# =============================================================================

# Expected schema structure for proof output files
PROOF_SCHEMA = {
    "required_keys": [
        "proof_successful",
        "verified",
        "lean_code",
        "proof_explanation",
        "lemmas",
    ],
    "lemma_required_keys": [
        "name",
        "statement",
        "compiler_out",
        "is_compiler_verified",
    ],
}


def verify_proof_output(
    workspace_dir: Path,
    expected_files: list[str] | list[ExpectedFile] | None = None,
) -> dict:
    """Verify proof output files against schema and content requirements.

    Args:
        workspace_dir: Path to workspace directory
        expected_files: List of expected files (strings or ExpectedFile objects)

    Returns dict with:
    - valid: bool - True if all checks pass
    - file_errors: list - Missing/unreadable files
    - schema_errors: list - Schema validation errors
    - content_warnings: list - Content quality warnings
    - files_found: dict - Info about each file found
    - proof_verified: bool - Whether proof was verified by Lean
    - lemma_count: int - Number of lemmas in proof
    """
    workspace = Path(workspace_dir)

    if expected_files is None:
        expected_files = ProofArtifact.get_expected_out_files()

    # Extract paths from ExpectedFile objects if needed
    file_paths = [f.path if isinstance(f, ExpectedFile) else f for f in expected_files]

    file_errors: list[str] = []
    schema_errors: list[str] = []
    content_warnings: list[str] = []
    files_found: dict[str, dict] = {}
    proof_verified = False
    lemma_count = 0

    for filename in file_paths:
        file_path = workspace / filename

        if not file_path.exists():
            file_errors.append(f"Missing file: {filename}")
            continue

        files_found[filename] = {"exists": True, "path": str(file_path)}

        if filename == "proof_out.json":
            json_result = _validate_proof_json(file_path, filename)
            schema_errors.extend(json_result.get("schema_errors", []))
            content_warnings.extend(json_result.get("content_warnings", []))
            files_found[filename].update(json_result.get("file_info", {}))
            proof_verified = json_result.get("proof_verified", False)
            lemma_count = json_result.get("lemma_count", 0)

        elif filename == "proof.lean":
            try:
                content = file_path.read_text(encoding="utf-8")
                if len(content.strip()) < 50:
                    content_warnings.append(f"{filename} is very short ({len(content)} chars)")
                if "sorry" in content.lower():
                    content_warnings.append(f"{filename} contains 'sorry' (incomplete proof)")
                files_found[filename]["size"] = len(content)
            except Exception as e:
                file_errors.append(f"Cannot read {filename}: {e}")

    valid = not file_errors and not schema_errors

    return {
        "valid": valid,
        "file_errors": file_errors,
        "schema_errors": schema_errors,
        "content_warnings": content_warnings,
        "files_found": files_found,
        "proof_verified": proof_verified,
        "lemma_count": lemma_count,
    }


def _validate_proof_json(file_path: Path, filename: str) -> dict:
    """Validate proof_out.json against schema requirements."""
    result = {
        "schema_errors": [],
        "content_warnings": [],
        "file_info": {},
        "proof_verified": False,
        "lemma_count": 0,
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

    # Check required keys
    for key in PROOF_SCHEMA["required_keys"]:
        if key not in data:
            result["schema_errors"].append(f"{filename}: Missing required '{key}' key")

    # Check proof status
    if "verified" in data:
        result["proof_verified"] = bool(data["verified"])
        result["file_info"]["verified"] = data["verified"]

    if "proof_successful" in data:
        result["file_info"]["proof_successful"] = data["proof_successful"]
        if not data["proof_successful"]:
            result["content_warnings"].append(f"{filename}: proof_successful is false")

    # Check lean_code — guard against agent writing a non-string (dict/list)
    # so the validator surfaces a schema error instead of crashing the gen_art
    # task with AttributeError on .strip()/.lower().
    lean_code = data.get("lean_code", "")
    if not isinstance(lean_code, str):
        result["schema_errors"].append(
            f"{filename}: 'lean_code' must be a string, got {type(lean_code).__name__}"
        )
    elif not lean_code or len(lean_code.strip()) < 50:
        result["schema_errors"].append(f"{filename}: 'lean_code' is empty or too short")
    elif "sorry" in lean_code.lower():
        result["content_warnings"].append(f"{filename}: lean_code contains 'sorry'")

    # Check lemmas array
    lemmas = data.get("lemmas", [])
    if not isinstance(lemmas, list):
        result["schema_errors"].append(f"{filename}: 'lemmas' must be an array")
    else:
        result["lemma_count"] = len(lemmas)
        result["file_info"]["lemma_count"] = len(lemmas)

        for i, lemma in enumerate(lemmas[:5]):
            if not isinstance(lemma, dict):
                result["schema_errors"].append(f"{filename}: Lemma {i} must be an object")
                continue
            for key in PROOF_SCHEMA["lemma_required_keys"]:
                if key not in lemma:
                    result["schema_errors"].append(f"{filename}: Lemma {i} missing '{key}'")

    return result
