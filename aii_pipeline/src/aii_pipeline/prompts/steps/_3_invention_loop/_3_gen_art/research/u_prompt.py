"""User prompt for research artifact.

Read top-to-bottom to understand the full prompt structure.
Research uses LLM with web search (not agent-based with sequential todos).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.prompts.components.read_skills import get_read_skills
from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt
from aii_pipeline.prompts.components.workspace import get_workspace_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dependencies import (
    DEPENDENCY_FIELDS,
    build_dependencies_prompt,
)

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT(
    context_content: str,
    plan_text: str,
    workspace: str = "",
    read_skills: str = "",
    user_data: str = "",
) -> str:
    return f"""{f"{read_skills}" if read_skills else ""}{workspace}
{user_data}

{
        f'''<context>
{context_content}
</context>

'''
        if context_content
        else ""
    }<artifact_plan>
{plan_text}
</artifact_plan>

<investigation_process>
1. DIVERGE: Brainstorm multiple angles/framings of the question before searching. Think across fields — what adjacent domains might have relevant insights?
2. SEARCH: Multiple queries per angle with different phrasings to discover the landscape
3. FETCH: Read promising URLs at high level. Snippets are NOT enough — fetch full pages
4. DETAIL: aii_web_tools__fetch_grep for specifics from key pages/PDFs
5. CONTRAST: Actively try to disprove your emerging conclusions. Search with different phrasings, "[topic] criticism", "[topic] limitations". Check across fields — the same finding may exist under different names
6. SYNTHESIZE: Integrate into balanced conclusion
7. ITERATE: Expect to repeat steps 2-6 if findings are incomplete or one-sided. Don't settle on first results
8. SUMMARIZE: Output JSON must include 'title' and 'summary' fields
</investigation_process>

<output_requirements>
- Write research_out.json to your workspace with all findings
- Provide your finding as clear prose WITH NUMBERED CITATIONS
- EVERY factual claim must have a citation number in brackets: [1], [2], [1, 3], etc.
- Include BOTH supporting AND contradicting evidence
- Be explicit about confidence level and what would change it
- End with follow-up questions for further investigation
</output_requirements>

Research everything specified in the artifact plan, but you may also investigate additional relevant aspects beyond what's listed. Investigate this question thoroughly."""


FORCE_OUTPUT_PROMPT = """STOP SEARCHING. You have gathered enough research.

Write your final JSON output NOW using ONLY the sources you have already found.

Do not search for more sources. Output your complete JSON response immediately."""


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    plan_text: str,
    artifacts: list[BaseArtifact] | None = None,
    dependency_ids: list[str] | None = None,
    agent_mode: bool = False,
    workspace_path: str = "",
    user_folder_path: str = "",
) -> str:
    """Build the user prompt for research execution.

    Args:
        plan_text: Plan fields serialized as YAML (from plan.to_prompt_yaml()).
        artifacts: List of artifacts to resolve dependency IDs from.
        dependency_ids: Artifact IDs to include as dependencies.
        agent_mode: If True, uses agent-style deps prompt (absolute workspace paths, read-only).
                   If False, uses LLM-style deps prompt (text summaries only).
        workspace_path: Absolute path to agent workspace (agent_mode only).
        user_folder_path: Absolute path to user data folder.
    """
    if not artifacts or not dependency_ids:
        deps_prompt = ""
    elif agent_mode:
        # Agent mode: full <dependencies> block with workspace paths and file info
        deps_prompt = build_dependencies_prompt(artifacts, dependency_ids)
    else:
        # LLM mode: just text summaries with enriched fields (no workspace instructions)
        from aii_lib.prompts import LLMPromptModel

        dep_set = set(dependency_ids)
        deps = [a for a in artifacts if a.id in dep_set]
        deps_prompt = (
            LLMPromptModel.list_to_prompt_yaml(
                deps,
                label="Dependency",
                include=DEPENDENCY_FIELDS,
                strip_nulls=True,
            )
            if deps
            else ""
        )
    return PROMPT(
        context_content=deps_prompt,
        plan_text=plan_text,
        workspace=get_workspace_prompt(workspace_path) if workspace_path else "",
        read_skills=get_read_skills("aii-web-research-tools") + "\n\n" if agent_mode else "",
        user_data=get_user_folder_prompt(user_folder_path) + get_user_request_prompt(),
    )


def get_force_output_prompt() -> str:
    """Prompt to force output when tool iterations are exhausted."""
    return FORCE_OUTPUT_PROMPT


# =============================================================================
# RETRY PROMPT
# =============================================================================


def build_research_retry_prompt(
    verification: dict,
    attempt: int = 1,
    max_attempts: int = 2,
) -> str:
    """Build a retry prompt for research verification failures."""
    file_errors = verification.get("file_errors", [])
    schema_errors = verification.get("schema_errors", [])
    content_warnings = verification.get("content_warnings", [])

    sections = []

    sections.append(f"""<verification_failed>
Your research output failed verification (attempt {attempt}/{max_attempts}).
</verification_failed>""")

    if file_errors:
        sections.append("""
<file_errors>
MISSING FILES:""")
        for err in file_errors:
            sections.append(f"  - {err}")
        sections.append("""
Fix: Create research_out.json with your research results.
</file_errors>""")

    if schema_errors:
        sections.append("""
<schema_errors>
JSON SCHEMA ERRORS:""")
        for err in schema_errors[:10]:
            sections.append(f"  - {err}")
        sections.append("""
Fix: research_out.json must have:
     {
       "answer": "comprehensive answer with [1], [2] citations",
       "sources": [{"index": 1, "url": "...", "title": "...", "summary": "..."}],
       "follow_up_questions": ["Question 1?", "Question 2?"],
       "summary": "what was found"
     }

     Each citation [N] in answer MUST match a source with that index.
</schema_errors>""")

    if content_warnings:
        sections.append("""
<content_warnings>
CONTENT ISSUES:""")
        for warn in content_warnings[:5]:
            sections.append(f"  - {warn}")
        sections.append("""
Fix: Ensure answer is comprehensive, has proper citations, and all sources are cited.
</content_warnings>""")

    sections.append("""
<task>
FIX ISSUES:
1. Output valid research_out.json with all required fields
2. Ensure every factual claim has a numbered citation [1], [2], etc.
3. Ensure every source has a matching citation in the answer
</task>""")

    return "\n".join(sections)
