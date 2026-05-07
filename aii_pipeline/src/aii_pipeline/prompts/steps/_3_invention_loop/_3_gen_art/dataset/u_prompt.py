"""User prompt for dataset artifact.

Read top-to-bottom to understand the full prompt structure.
Each prompt group is delivered as a separate sequential prompt, each with header + TODOs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )

from aii_pipeline.prompts.components.read_skills import get_read_skills
from aii_pipeline.prompts.components.resources import get_resources_prompt
from aii_pipeline.prompts.components.todo import get_todo_header
from aii_pipeline.prompts.components.tool_calling import get_tool_calling_guidance
from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt
from aii_pipeline.prompts.components.workspace import get_workspace_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dependencies import (
    build_dependencies_prompt,
)

# =============================================================================
# PROMPT SECTIONS (edit these directly)
# =============================================================================

HEADER = """{workspace}
{user_data}
<artifact_plan>
{plan_text}
</artifact_plan>

{dependencies_section}
{resources}

<available_data_sources>
Use the sources appropriate to your task. Read the relevant skill file BEFORE using each source.

- **HuggingFace Hub** (HF) — ML datasets (NLP, vision, tabular, benchmarks)
- **Our World in Data** (OWID) — Global statistics (energy, health, economics, environment, demographics)
- **Alternate methods** — Python/Bash (sklearn.datasets, openml, direct URL, APIs, etc.)

If the plan specifies a source or one fits better, use it.
You may combine sources. Use WebSearch to research candidates (background, papers, provenance) — NOT to find/download datasets.
</available_data_sources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — dataset selection, evaluation metrics, agent orchestration patterns
</available_domain_handbooks>

{tool_calling}

{todo_header}"""

# Prompt groups — {placeholders} filled by _fill_prompts()
# Each inner list = one sequential prompt. Each string = one TODO within that prompt.
# Each funnel stage is 2× the next, capped by configurable limits.
PROMPTS = [
    [  # Prompt 1: Search & download
        get_read_skills(
            "aii-python",
            "aii-long-running-tasks",
            "aii-json",
            "aii-file-size-limit",
            "aii-use-hardware",
            "aii-parallel-computing",
        ),
        """Read skill files for your data sources (see <available_data_sources>) and domain handbook if applicable (see <available_domain_handbooks>). Based on plan and context, decide which source(s) to use. Include everything specified in the artifact plan, but you may also collect additional relevant data beyond what's listed. Run {search_tool_n} diverse searches across chosen source(s) — BROAD, GENERAL terms, not very specific. Parallelize where supported.""",
        """Identify the {chosen_for_preview_n} most promising datasets. IMPORTANT: Only consider datasets under {max_dataset_size}. Preview/inspect sample rows for each candidate. Parallelize previews.""",
        """Research each candidate BEFORE choosing which to download. For each, WebSearch: dataset name, papers citing it, original source/task, popularity. Red flags: no search results, no papers, anonymized features (F1, F2...), <100 downloads, no documentation. Green flags: papers using it, clear documentation, meaningful features, established benchmark. Also consider: will features/structure allow meaningful evaluation of the planned method?""",
        """Decide which to KEEP vs DISCARD. Look for: clear structure, relevant fields, quality examples matching requirements, confirmed provenance. Determine which {chosen_for_download_n} datasets have the most suitable data. Download and save to `temp/datasets/`. Parallelize downloads.""",
    ],
    [  # Prompt 2: Implement data.py & validate
        """For the top {chosen_for_download_n} datasets, create data.py (uv inline script) that: loads from temp/datasets/, standardizes to exp_sel_data_out.json schema (aii-json skill), extracts all examples per dataset, handles domain requirements, saves to full_data_out.json.

Each data ROW must be a separate example — do NOT create one example per dataset or per fold. Each data point (row, sample, instance) = one example. 500 rows → 500 examples. The output is GROUPED BY DATASET:
```json
{{
  "datasets": [
    {{
      "dataset": "iris",
      "examples": [
        {{"input": "...", "output": "...", "metadata_fold": 2, "metadata_feature_names": [...]}},
        ...
      ]
    }},
    {{
      "dataset": "adult_census",
      "examples": [...]
    }}
  ]
}}
```
Per-example required fields:
- `input`: input features/text (tabular: JSON string of feature values)
- `output`: target/label (as string)
Per-example optional metadata via `metadata_<name>` fields (flat, not nested object):
- `metadata_fold`: fold assignment (int), `metadata_feature_names`: feature name list, `metadata_task_type`: "classification"/"regression", `metadata_n_classes`: number of classes, `metadata_row_index`: original row index, etc.
Do NOT use `split`, `dataset`, or `context` as per-example fields. Dataset name goes at the group level, metadata goes in `metadata_*` fields.""",
        """Run 'uv run data.py' and fix errors. Validate full_data_out.json against exp_sel_data_out.json schema (aii-json skill) — fix errors. Generate preview, mini, full versions with aii-json skill's format script.""",
        """Read preview to inspect examples. Choose THE BEST {chosen_final_n} DATASET{chosen_final_s} based on domain requirements and artifact objective. Be very attentive to meticulously and exhaustively fix any errors in your code.""",
    ],
    [  # Prompt 3: Finalize
        """Update data.py to only include the chosen {chosen_final_n} dataset{chosen_final_s_lower} and generate full_data_out.json. Re-run to generate full_data_out.json. Validate output format with aii-json skill and fix any errors. Generate full, mini, and preview versions with aii-json skill's format script using `--input full_data_out.json` (creates full_full_data_out.json, mini_full_data_out.json, preview_full_data_out.json — rename to full_data_out.json, mini_data_out.json, preview_data_out.json).""",
        """Verify full_data_out.json, preview_data_out.json, and mini_data_out.json exist in your workspace (see <workspace>) and contain correct data.""",
        """Apply aii-file-size-limit skill's file size check procedure ({file_max_size} limit) to full_data_out.json.""",
        """Ensure a `pyproject.toml` exists in your workspace with ALL dependencies pinned to the exact versions installed in your .venv (run `.venv/bin/pip freeze` to get them). This is required for reproducibility. The [project] section must include name, version, requires-python, and a dependencies list with pinned versions (e.g. `numpy==2.0.2`, not `numpy>=2.0`).""",
    ],
]


def _fill_prompts(
    target_n: int,
    max_dataset_size: str = "300MB",
    file_max_size: str = "100MB",
    search_tool_cap: int = 50,
    chosen_for_preview_cap: int = 25,
    chosen_for_download_cap: int = 15,
) -> list[list[str]]:
    """Compute funnel numbers and fill PROMPTS placeholders.

    Each stage is 2× the next, capped by the stage limit:
      chosen_for_download_n = min(target_n * 2, chosen_for_download_cap)
      chosen_for_preview_n  = min(chosen_for_download_n * 2, chosen_for_preview_cap)
      search_tool_n         = min(chosen_for_preview_n * 2, search_tool_cap)

    Args:
        target_n: Final number of datasets to deliver.
        max_dataset_size: Max dataset size string (e.g. "300MB").
        file_max_size: Max output file size string (e.g. "100MB").
        search_tool_cap: Max keyword searches (default 50, configurable in config.yaml).
        chosen_for_preview_cap: Max datasets chosen for preview (default 25).
        chosen_for_download_cap: Max datasets chosen for download (default 15).
    """
    chosen_for_download_n = min(target_n * 2, chosen_for_download_cap)
    chosen_for_preview_n = min(chosen_for_download_n * 2, chosen_for_preview_cap)
    search_tool_n = min(chosen_for_preview_n * 2, search_tool_cap)

    placeholders = {
        "search_tool_n": search_tool_n,
        "chosen_for_preview_n": chosen_for_preview_n,
        "chosen_for_download_n": chosen_for_download_n,
        "chosen_final_n": target_n,
        "chosen_final_s": "S" if target_n > 1 else "",
        "chosen_final_s_lower": "s" if target_n > 1 else "",
        "per_dataset": " per dataset" if target_n > 1 else "",
        "max_dataset_size": max_dataset_size,
        "file_max_size": file_max_size,
    }

    return [[item.format(**placeholders) for item in group] for group in PROMPTS]


# =============================================================================
# EXPORTS (main prompt functions)
# =============================================================================


def get_all_prompts(
    plan_text: str,
    artifacts: list[BaseArtifact] | None = None,
    dependency_ids: list[str] | None = None,
    target_num_datasets: int = 1,
    max_dataset_size: str = "300MB",
    file_max_size: str = "100MB",
    search_tool_cap: int = 50,
    chosen_for_preview_cap: int = 25,
    chosen_for_download_cap: int = 15,
    workspace_path: str = "",
    user_folder_path: str = "",
) -> list[str]:
    """Get sequential prompts — one per phase, each with header + TODOs."""
    prompt_groups = _fill_prompts(
        target_num_datasets,
        max_dataset_size,
        file_max_size,
        search_tool_cap,
        chosen_for_preview_cap,
        chosen_for_download_cap,
    )
    header = _build_header(
        plan_text,
        artifacts,
        dependency_ids,
        workspace_path,
        max_dataset_size,
        user_folder_path,
    )
    return [f"{header}\n{_format_todos(group)}" for group in prompt_groups]


# =============================================================================
# HELPERS (private functions)
# =============================================================================


def _get_resources() -> str:
    """Get combined resources section."""
    return get_resources_prompt(include=["software", "skills"])


def _format_todos(todos: list[str]) -> str:
    """Format TODO items into a single <todos> block."""
    lines = ["<todos>"]
    for i, item in enumerate(todos, start=1):
        lines.append(f"TODO {i}. {item}")
    lines.append("</todos>")
    return "\n".join(lines)


def _build_header(
    plan_text: str,
    artifacts: list[BaseArtifact] | None,
    dependency_ids: list[str] | None,
    workspace_path: str = "",
    max_dataset_size: str = "300MB",
    user_folder_path: str = "",
) -> str:
    """Build the header section with substitutions."""
    deps_section = (
        build_dependencies_prompt(artifacts, dependency_ids or [])
        if artifacts and dependency_ids
        else ""
    )
    return HEADER.format(
        workspace=get_workspace_prompt(workspace_path) if workspace_path else "",
        user_data=get_user_folder_prompt(user_folder_path) + get_user_request_prompt(),
        plan_text=plan_text,
        dependencies_section=deps_section,
        resources=_get_resources(),
        tool_calling=get_tool_calling_guidance(),
        todo_header=get_todo_header(),
        max_dataset_size=max_dataset_size,
    )


# =============================================================================
# RETRY PROMPT
# =============================================================================


def build_dataset_retry_prompt(
    verification: dict,
    attempt: int = 1,
    max_attempts: int = 2,
) -> str:
    """Build a retry prompt for dataset verification failures.

    Similar to build_artifact_retry_prompt() in gen_strat.

    Args:
        verification: Result from verify_dataset_output()
        attempt: Current attempt number (1-indexed)
        max_attempts: Total max attempts

    Returns:
        Retry prompt explaining what failed and how to fix it
    """
    file_errors = verification.get("file_errors", [])
    schema_errors = verification.get("schema_errors", [])
    content_warnings = verification.get("content_warnings", [])

    sections = []

    # Header
    sections.append(f"""<verification_failed>
Your dataset output failed verification (attempt {attempt}/{max_attempts}).
</verification_failed>""")

    # File errors (most critical)
    if file_errors:
        sections.append("""
<file_errors>
MISSING OR UNREADABLE FILES:""")
        for err in file_errors:
            sections.append(f"  - {err}")
        sections.append("""
Fix: Create the missing files directly in your workspace (see <workspace> above for the exact path).
     Use 'ls' to check what files exist, then create the missing ones.
</file_errors>""")

    # Schema errors
    if schema_errors:
        sections.append("""
<schema_errors>
JSON SCHEMA VALIDATION ERRORS:""")
        for err in schema_errors[:10]:  # Limit to 10
            sections.append(f"  - {err}")
        if len(schema_errors) > 10:
            sections.append(f"  ... and {len(schema_errors) - 10} more errors")
        sections.append("""
Fix: Your JSON files must follow this datasets-grouped structure:
     {
       "datasets": [
         {
           "dataset": "dataset_name",
           "examples": [
             {
               "input": "string (required)",
               "output": "string (required)",
               "metadata_fold": 2,
               "metadata_feature_names": [...]
             }
           ]
         }
       ]
     }

     NO 'split', 'dataset', or 'context' per-example. Dataset name at group level.
     Metadata via flat metadata_<name> fields (e.g. metadata_fold, metadata_task_type).
     Read exp_sel_data_out.json schema in aii-json skill.
     Then update data.py and regenerate the output files.
</schema_errors>""")

    # Content warnings
    if content_warnings:
        sections.append("""
<content_warnings>
CONTENT QUALITY ISSUES:""")
        for warn in content_warnings[:5]:
            sections.append(f"  - {warn}")
        if len(content_warnings) > 5:
            sections.append(f"  ... and {len(content_warnings) - 5} more warnings")
        sections.append("""
Fix: Ensure examples have non-empty input and output fields.
     Review data.py to ensure proper data extraction.
</content_warnings>""")

    # Task section
    tasks = []
    if file_errors:
        tasks.append(
            "1. Create all missing files (data.py, full_data_out.json, preview_data_out.json, mini_data_out.json)"
        )
    if schema_errors:
        tasks.append("2. Fix JSON schema errors by updating data.py")
        tasks.append("3. Re-run data.py to regenerate all output files")
        tasks.append(
            "4. Validate with aii-json skill: validate full_data_out.json against exp_sel_data_out schema"
        )
    if content_warnings and not schema_errors:
        tasks.append("5. Address content quality warnings if possible")

    if tasks:
        sections.append(f"""
<task>
FIX THESE ISSUES:
{chr(10).join(tasks)}

After making changes, verify:
- 'ls -la' shows all required files
- JSON files are valid (use aii-json skill validation)
- full_data_out.json has at least 50 examples

IMPORTANT: Your final response should be at most 300 characters long.
</task>""")

    return "\n".join(sections)
