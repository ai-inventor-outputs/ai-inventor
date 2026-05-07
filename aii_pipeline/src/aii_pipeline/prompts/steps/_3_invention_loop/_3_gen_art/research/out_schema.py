"""Schema for research artifact.

Research artifacts conduct thorough web research.
Uses structured LLM output with JSON schema (not agent-based).

Includes verification logic for post-execution validation.
"""

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.prompts import LLMPrompt, LLMPromptModel, LLMStructOut, LLMStructOutModel
from pydantic import Field

from ..out_schema import ArtifactType, BaseArtifact, BaseExpectedFiles

# =============================================================================
# SCHEMAS
# =============================================================================


class Source(LLMPromptModel, LLMStructOutModel):
    """A source used in the research."""

    index: Annotated[int, LLMPrompt, LLMStructOut] = Field(
        description="Citation number (1, 2, 3, ...)"
    )
    url: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Full URL of the source")
    title: Annotated[str, LLMPrompt, LLMStructOut] = Field(description="Title of the article/page")
    summary: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Brief summary of what this source contributed"
    )


class ResearchExpectedFiles(BaseExpectedFiles):
    """All expected output files from research artifact."""

    output: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Path to research output JSON. Example: 'research_out.json'"
    )


class ResearchArtifact(BaseArtifact):
    """Research artifact — structured output + file metadata.

    Conducts thorough web research using WebSearch and WebFetch.
    Returns structured JSON output with citations.
    """

    kind: Literal["research_artifact"] = "research_artifact"
    type: Annotated[Literal[ArtifactType.RESEARCH], LLMPrompt] = ArtifactType.RESEARCH
    out_expected_files: Annotated[ResearchExpectedFiles, LLMPrompt, LLMStructOut] = Field(
        description="All output files you created. Must include research_out.json with your research findings."
    )
    answer: Annotated[str, LLMPrompt, LLMStructOut] = Field(
        description="Comprehensive answer with NUMBERED CITATIONS. Cite sources by number: 'Claim [1].' or 'According to [2, 3]...'"
    )
    sources: Annotated[list[Source], LLMPrompt, LLMStructOut] = Field(
        description="All sources used, with index matching citation numbers in answer"
    )
    follow_up_questions: Annotated[list[str], LLMPrompt, LLMStructOut] = Field(
        description="2-3 follow-up questions that emerged from the investigation"
    )

    @staticmethod
    def get_expected_out_files() -> list[ExpectedFile]:
        """All expected output files with descriptions. Used for dependency copying and verification."""
        return [
            ExpectedFile(
                "research_out.json",
                "Structured research findings with answer, sources, and title",
            ),
        ]

    out_demo_files: Annotated[list[ExpectedFile], LLMPrompt] = Field(
        default=[
            ExpectedFile(
                "research_report.md",
                "Research report markdown (auto-generated from artifact)",
            )
        ],
        description="Primary file(s) to convert to demo formats",
    )


# =============================================================================
# VERIFICATION
# =============================================================================


def verify_research_output(
    workspace_dir: Path,
    expected_files: list[str] | list[ExpectedFile] | None = None,
) -> dict:
    """Verify research output files against schema and content requirements.

    Args:
        workspace_dir: Path to workspace directory
        expected_files: List of expected files (strings or ExpectedFile objects)

    Returns dict with:
    - valid: bool - True if all checks pass
    - file_errors: list - Missing/unreadable files
    - schema_errors: list - Schema validation errors
    - content_warnings: list - Content quality warnings
    - files_found: dict - Info about each file found
    - source_count: int - Number of sources cited
    """
    workspace = Path(workspace_dir)

    if expected_files is None:
        expected_files = ["research_out.json"]

    # Extract paths from ExpectedFile objects if needed
    file_paths = [f.path if isinstance(f, ExpectedFile) else f for f in expected_files]

    file_errors: list[str] = []
    schema_errors: list[str] = []
    content_warnings: list[str] = []
    files_found: dict[str, dict] = {}
    source_count = 0

    for filename in file_paths:
        file_path = workspace / filename

        if not file_path.exists():
            file_errors.append(f"Missing file: {filename}")
            continue

        files_found[filename] = {"exists": True, "path": str(file_path)}

        if filename == "research_out.json":
            json_result = _validate_research_json(file_path, filename)
            schema_errors.extend(json_result.get("schema_errors", []))
            content_warnings.extend(json_result.get("content_warnings", []))
            files_found[filename].update(json_result.get("file_info", {}))
            source_count = json_result.get("source_count", 0)

    valid = not file_errors and not schema_errors

    return {
        "valid": valid,
        "file_errors": file_errors,
        "schema_errors": schema_errors,
        "content_warnings": content_warnings,
        "files_found": files_found,
        "source_count": source_count,
    }


def _validate_research_json(file_path: Path, filename: str) -> dict:
    """Validate research_out.json against schema requirements."""
    result = {
        "schema_errors": [],
        "content_warnings": [],
        "file_info": {},
        "source_count": 0,
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

    # Check required fields
    required_fields = ["title", "answer", "sources", "follow_up_questions", "summary"]
    for field in required_fields:
        if field not in data:
            result["schema_errors"].append(f"{filename}: Missing required '{field}' field")

    # Check title — agents occasionally hand us a dict/list instead of a
    # string for these fields; guard so validation surfaces a useful schema
    # error instead of crashing the whole gen_art task with an AttributeError.
    title = data.get("title", "")
    if not isinstance(title, str):
        result["schema_errors"].append(
            f"{filename}: 'title' must be a string, got {type(title).__name__}"
        )
    elif not title or len(title.strip()) < 5:
        result["content_warnings"].append(f"{filename}: 'title' is too short")

    # Check answer
    answer = data.get("answer", "")
    if not isinstance(answer, str):
        result["schema_errors"].append(
            f"{filename}: 'answer' must be a string, got {type(answer).__name__}"
        )
    elif not answer or len(answer.strip()) < 100:
        result["content_warnings"].append(f"{filename}: 'answer' is too short")

    # Check sources
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        result["schema_errors"].append(f"{filename}: 'sources' must be an array")
    else:
        result["source_count"] = len(sources)
        result["file_info"]["source_count"] = len(sources)

        if len(sources) < 3:
            result["content_warnings"].append(
                f"{filename}: Only {len(sources)} sources (recommend at least 3)"
            )

        # Check source structure
        for i, source in enumerate(sources[:5]):
            if not isinstance(source, dict):
                result["schema_errors"].append(f"{filename}: Source {i} must be an object")
                continue
            for key in ["index", "url", "title", "summary"]:
                if key not in source:
                    result["schema_errors"].append(f"{filename}: Source {i} missing '{key}'")

        # Check citation consistency (are all source indices cited?). Skip
        # when ``answer`` isn't a string — schema_errors above already flagged
        # that and finditer would crash on a non-string.
        if isinstance(answer, str) and answer and sources:
            source_indices = {s.get("index") for s in sources if isinstance(s, dict)}
            citation_pattern = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
            cited_indices = set()
            for match in citation_pattern.finditer(answer):
                for idx in match.group(1).replace(" ", "").split(","):
                    try:
                        cited_indices.add(int(idx))
                    except ValueError:
                        pass

            uncited = source_indices - cited_indices
            if uncited:
                result["content_warnings"].append(
                    f"{filename}: Sources with uncited indices: {uncited}"
                )

    # Check follow_up_questions
    follow_up = data.get("follow_up_questions", [])
    if not isinstance(follow_up, list):
        result["schema_errors"].append(f"{filename}: 'follow_up_questions' must be an array")
    elif len(follow_up) < 2:
        result["content_warnings"].append(
            f"{filename}: Only {len(follow_up)} follow-up questions (recommend 2-3)"
        )

    return result
