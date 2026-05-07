"""User prompt for proof artifact - Seed-Prover inspired structure.

Read top-to-bottom to understand the full prompt structure.
Each prompt group is delivered as a separate sequential prompt, each with header + TODOs.

Based on Seed-Prover approach (ByteDance Seed AI4Math):
- Lemma-style proving: Generate intermediate lemmas before main proof
- Search Mathlib: Use semantic and pattern search before writing proofs
- Iterative refinement: Test with compiler, fix errors, try alternative approaches
- Self-summarization: Learn from failed attempts, consider trajectory changes
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

{context_section}
{resources}

{tool_calling}

{todo_header}"""

PROMPTS = [
    [  # Prompt 1: Prove
        get_read_skills("aii-lean", "aii-json"),
        """Read the exp_proof_out schema from the aii-json skill for output format. Include everything in artifact plan; you may also prove additional lemmas/properties. Analyze the theorem: proof type (definitional equality, induction, algebraic, case analysis), mathematical domain (number theory, algebra, combinatorics, analysis), required imports (Mathlib.Tactic, BigOperators, etc.). Note if division should be avoided (use multiplication form).""",
        """SEARCH: Search Mathlib using aii-lean skill's semantic and pattern search. Run multiple searches in parallel — note useful lemmas, theorems, and tactics.""",
        """DECOMPOSE: Identify useful intermediate lemmas before tackling the main theorem.""",
        """SKETCH: Write the full proof structure with `sorry` placeholders for all lemmas and the main theorem. Verify it compiles — this confirms the overall logic is sound.""",
        """PROVE LEMMAS: Tackle `sorry`s one by one. Be meticulous and exhaustive — spend significant effort on each lemma. For each: search Mathlib for related proofs, try multiple tactics (ring, simp, omega, linarith, nlinarith), explore alternative formulations. Use `calc` blocks for equality proofs. Break into smaller sub-lemmas if needed. Prove independently using `lemma` keyword. Keep proved lemmas — they can be reused across attempts. If a lemma fails 3+ times, consider if it's actually true or needs a different approach.""",
        """PROVE THEOREM: Replace the main theorem's `sorry` using `theorem` keyword and apply proved lemmas. Search Mathlib for related theorems that could help. Be thorough — try every combination of proved lemmas, tactics, and alternative approaches before giving up. If you can see how it would work with different lemmas, go back and re-sketch and prove the new lemmas.""",
        """VERIFY: Test the complete proof with aii-lean skill. If errors, fix syntax/type errors, add missing imports, re-verify after each fix. If verified=true and no `sorry` remains, consider your task completed.""",
        """SELF-SUMMARIZE: What worked? What failed? What to try next?""",
        """RETRY OR PIVOT: Fix failed lemmas and retry. If still failing, try completely different proof strategy (definitional equality, induction, algebraic, case analysis, direct), different type representation, stronger/weaker intermediate lemmas. Search Mathlib again. If multiple lemmas keep failing, pivot — go back to the SEARCH step with a completely different proof approach. If theorem appears unprovable after exhaustive attempts, document specific reasons why and note which sub-lemmas ARE provable (partial progress). IMPORTANT: Keep proved lemmas in your "lemma pool" — don't discard working code. Hard-to-prove lemmas are often crucial to the final proof.""",
    ],
    [  # Prompt 2: Finalize
        """**FINAL TESTING PHASE**: Re-verify the complete proof one more time with aii_run_lean.py. Check that verified=true and has_sorries=false. If any errors remain, fix them. Ensure the proof is complete without any 'sorry' placeholders.""",
        """Save the complete Lean 4 code to './proof.lean'. Create './proof_out.json' following the exp_proof_out schema from the aii-json skill exactly:
- proof_successful: true/false
- verified: true/false (from aii_run_lean.py result)
- lean_code: complete Lean 4 code as string
- proof_explanation: explanation of proof strategy
- lemmas: array of {name, statement, compiler_out, is_compiler_verified, tactic} for each lemma
- approaches_tried: array of {approach, reason_failed} if proof failed
- error_messages: array of final error messages if proof failed""",
        """Use 'ls' to verify ./proof.lean and ./proof_out.json exist in your workspace.""",
    ],
]


# =============================================================================
# EXPORTS (main prompt functions)
# =============================================================================


def get_all_prompts(
    plan_text: str,
    artifacts: list[BaseArtifact] | None = None,
    dependency_ids: list[str] | None = None,
    workspace_path: str = "",
    user_folder_path: str = "",
) -> list[str]:
    """Get sequential prompts — one per phase, each with header + TODOs."""
    header = _build_header(plan_text, artifacts, dependency_ids, workspace_path, user_folder_path)
    return [f"{header}\n{_format_todos(group)}" for group in PROMPTS]


# =============================================================================
# HELPERS (private functions)
# =============================================================================


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
        context_section=deps_section,
        resources=get_resources_prompt(include=["software"]),
        tool_calling=get_tool_calling_guidance(),
        todo_header=get_todo_header(),
    )


# =============================================================================
# RETRY PROMPT
# =============================================================================


def build_proof_retry_prompt(
    verification: dict,
    attempt: int = 1,
    max_attempts: int = 2,
) -> str:
    """Build a retry prompt for proof verification failures."""
    file_errors = verification.get("file_errors", [])
    schema_errors = verification.get("schema_errors", [])
    content_warnings = verification.get("content_warnings", [])
    proof_verified = verification.get("proof_verified", False)

    sections = []

    sections.append(f"""<verification_failed>
Your proof output failed verification (attempt {attempt}/{max_attempts}).
</verification_failed>""")

    if file_errors:
        sections.append("""
<file_errors>
MISSING FILES:""")
        for err in file_errors:
            sections.append(f"  - {err}")
        sections.append("""
Fix: Create the missing files:
     - proof.lean: Complete Lean 4 proof code
     - proof_out.json: Proof metadata following the exp_proof_out schema from the aii-json skill
</file_errors>""")

    if schema_errors:
        sections.append("""
<schema_errors>
JSON SCHEMA ERRORS:""")
        for err in schema_errors[:10]:
            sections.append(f"  - {err}")
        sections.append("""
Fix: proof_out.json must have:
     {
       "proof_successful": true/false,
       "verified": true/false (from aii-lean skill),
       "lean_code": "complete Lean 4 code",
       "proof_explanation": "explanation",
       "lemmas": [{"name": "...", "statement": "...", "compiler_out": "...", "is_compiler_verified": true/false}]
     }

     Read the exp_proof_out schema from the aii-json skill for exact format.
</schema_errors>""")

    if content_warnings:
        sections.append("""
<content_warnings>
CONTENT ISSUES:""")
        for warn in content_warnings[:5]:
            sections.append(f"  - {warn}")
        sections.append("""
Fix: Remove 'sorry' from proofs, ensure proof compiles with aii-lean skill.
</content_warnings>""")

    if not proof_verified:
        sections.append("""
<proof_not_verified>
The proof has not been verified by Lean. Use the aii-lean skill's verify command on proof.lean.
</proof_not_verified>""")

    tasks = ["1. Fix any missing files or schema errors"]
    if not proof_verified:
        tasks.append("2. Run aii-lean skill to verify proof.lean compiles")
        tasks.append("3. Update proof_out.json with verified: true if successful")

    sections.append(f"""
<task>
FIX ISSUES:
{chr(10).join(tasks)}

IMPORTANT: Your final response should be at most 300 characters long.
</task>""")

    return "\n".join(sections)
