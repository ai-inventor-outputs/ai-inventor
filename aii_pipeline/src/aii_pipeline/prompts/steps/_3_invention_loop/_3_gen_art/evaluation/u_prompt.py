"""User prompt for evaluation artifact.

Read top-to-bottom to understand the full prompt structure.
Each prompt group is delivered as a separate sequential prompt, each with header + TODOs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
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

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — evaluation metrics, agent orchestration patterns, benchmark design
</available_domain_handbooks>

{tool_calling}

{todo_header}"""

PROMPTS = [
    [  # Prompt 1: Implement
        get_read_skills(
            "aii-python",
            "aii-long-running-tasks",
            "aii-json",
            "aii-file-size-limit",
            "aii-use-hardware",
            "aii-parallel-computing",
        ),
        """Read preview files from dependencies to understand prediction format. Evaluate ALL experiments provided — do not skip or select a subset. Avoid re-training or re-executing the method unless absolutely necessary; prefer loading predictions from each dependency's method_out.json / predict_* fields. Read domain handbook if applicable (see <available_domain_handbooks>). Decide evaluation metrics based on artifact plan. Test basic functionality with 'uv run'.""",
        """Fully implement evaluation as described in artifact plan in './eval.py'. Use exp_eval_sol_out.json schema in aii-json skill for output format validation. Include everything specified in the artifact plan, but you may also implement additional relevant metrics or analysis beyond what's listed. Be very attentive to meticulously and exhaustively fix any errors in your code.""",
    ],
    [  # Prompt 2: Format & verify
        """Use aii-json skill's format script with `--input eval_out.json` to generate full, mini, and preview versions. If not in your workspace (see <workspace> above), copy them there. Run 'ls -lh' to verify these three files exist (DO NOT read them).""",
        """Apply aii-file-size-limit skill's file size check procedure ({file_max_size} limit) to eval_out.json and full_eval_out.json.""",
        """Ensure a `pyproject.toml` exists in your workspace with ALL dependencies pinned to the exact versions installed in your .venv (run `.venv/bin/pip freeze` to get them). This is required for reproducibility. The [project] section must include name, version, requires-python, and a dependencies list with pinned versions (e.g. `numpy==2.0.2`, not `numpy>=2.0`).""",
    ],
]


# =============================================================================
# EXPORTS (main prompt functions)
# =============================================================================


def get_all_prompts(
    plan_text: str,
    artifacts: list[BaseArtifact] | None = None,
    dependency_ids: list[str] | None = None,
    file_max_size: str = "100MB",
    workspace_path: str = "",
    user_folder_path: str = "",
) -> list[str]:
    """Get sequential prompts — one per phase, each with header + TODOs."""
    header = _build_header(plan_text, artifacts, dependency_ids, workspace_path, user_folder_path)
    result = []
    for group in PROMPTS:
        todos = [t.format(file_max_size=file_max_size) for t in group]
        result.append(f"{header}\n{_format_todos(todos)}")
    return result


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
    )


# =============================================================================
# RETRY PROMPT
# =============================================================================


def build_evaluation_retry_prompt(
    verification: dict,
    attempt: int = 1,
    max_attempts: int = 2,
) -> str:
    """Build a retry prompt for evaluation verification failures."""
    file_errors = verification.get("file_errors", [])
    schema_errors = verification.get("schema_errors", [])
    content_warnings = verification.get("content_warnings", [])

    sections = []

    sections.append(f"""<verification_failed>
Your evaluation output failed verification (attempt {attempt}/{max_attempts}).
</verification_failed>""")

    if file_errors:
        sections.append("""
<file_errors>
MISSING OR UNREADABLE FILES:""")
        for err in file_errors:
            sections.append(f"  - {err}")
        sections.append("""
Fix: Create missing files by running eval.py.
     Required: eval.py, eval_out.json, full_eval_out.json, mini_eval_out.json, preview_eval_out.json
</file_errors>""")

    if schema_errors:
        sections.append("""
<schema_errors>
JSON SCHEMA / CODE VALIDATION ERRORS:""")
        for err in schema_errors[:10]:
            sections.append(f"  - {err}")
        if len(schema_errors) > 10:
            sections.append(f"  ... and {len(schema_errors) - 10} more errors")
        sections.append("""
Fix: Your JSON must follow the datasets-grouped exp_eval_sol_out.json schema:
     {
       "metrics_agg": {"<metric_name>": 0.85, ...},  // REQUIRED, at least one metric
       "datasets": [
         {
           "dataset": "dataset_name",
           "examples": [
             {
               "input": "...", "output": "...",
               "metadata_fold": 2,
               "predict_<method>": "...",
               "eval_<metric>": 0.9
             }
           ]
         }
       ]
     }

     NO 'split', 'dataset', or 'context' per-example. Dataset name at group level.
     Metadata via flat metadata_<name> fields.
     Read exp_eval_sol_out.json schema in aii-json skill.
</schema_errors>""")

    if content_warnings:
        sections.append("""
<content_warnings>
CONTENT QUALITY ISSUES:""")
        for warn in content_warnings[:5]:
            sections.append(f"  - {warn}")
        sections.append("""
Fix: Ensure metrics_agg has values and each example has eval_* metrics.
</content_warnings>""")

    tasks = []
    if file_errors:
        tasks.append("1. Run eval.py to generate missing files")
    if schema_errors:
        tasks.append("2. Fix eval.py to produce correct JSON schema")
        tasks.append("3. Use aii-json skill validation to verify")
    if tasks:
        sections.append(f"""
<task>
FIX ISSUES:
{chr(10).join(tasks)}

IMPORTANT: Your final response should be at most 300 characters long.
</task>""")

    return "\n".join(sections)
