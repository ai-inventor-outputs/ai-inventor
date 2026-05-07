"""User prompt for strategy generation (Step 3.1: GEN_STRAT).

Read top-to-bottom to understand the full prompt structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.utils import LLMPromptModel, to_prompt_yaml, to_prompt_yaml_list

from ....components.artifact_planning import get_artifact_planning
from ....components.artifact_scope import get_artifact_scope
from ....components.artifact_summaries import get_artifact_context
from ....components.user_folder import get_user_folder_prompt
from ....components.user_request import get_user_request_prompt

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT(
    hypothesis_text: str,
    current_iteration: int,
    max_iterations: int,
    remaining: int,
    previous_strategy_section: str,
    artifact_planning_guidance: str,
    artifact_types_text: str,
    artifact_scope_text: str,
    existing_artifacts_text: str,
    narratives_text: str,
    num_strategies: int,
    art_limit: int | None = None,
    reviewer_feedback_text: str = "",
) -> str:
    return f"""<hypothesis>
Your strategy should advance this hypothesis.

{hypothesis_text}
</hypothesis>

<iteration_status>
Current iteration: {current_iteration} of {max_iterations}
Remaining (including this one): {remaining}
</iteration_status>

{previous_strategy_section}

<dependency_rules>
- depends_on is a list of objects {{id, label}} — each entry references an existing artifact and tags how it is being used
- "id" can ONLY reference IDs from <existing_artifacts> — never IDs you are proposing (all new artifacts run in parallel)
- "label" is a SHORT free-text type label (a word or two, NOT a sentence) describing what role the dep plays — e.g. "dataset", "validates", "extends", "supersedes". Required on every dep.
- Setting depends_on provides the dependency's out_dependency_files to your artifact at execution time
- If no suitable existing artifacts exist, use empty depends_on
- New artifact IDs are assigned by the system after submission — do not invent IDs for your proposed artifacts
</dependency_rules>

<available_artifact_types>
Artifact types you can plan. Use this to choose the right types for your strategy objectives.

{artifact_types_text}
</available_artifact_types>

{artifact_scope_text}

{artifact_planning_guidance}

<existing_artifacts>
{
        existing_artifacts_text.strip()
        if existing_artifacts_text.strip()
        else "None yet (first iteration)."
    }
</existing_artifacts>

{
        f'''<current_paper>
The current paper draft — represents the research story so far.

Use this to understand what's working, what's not, and what gaps remain.
Gaps and weak results signal what to try differently — not what to conclude.

{narratives_text}
</current_paper>'''
        if narratives_text
        else ""
    }

{
        f'''<reviewer_feedback>
Paper reviewer feedback from the previous iteration. Your strategy MUST address these critiques.
Prioritize major issues — these are the most impactful improvements to make.

{reviewer_feedback_text}
</reviewer_feedback>'''
        if reviewer_feedback_text
        else ""
    }

<task>
Generate {num_strategies} research {
        "strategy" if num_strategies == 1 else "strategies"
    } for THIS iteration.

{
        f"**ARTIFACT LIMIT: Each strategy may contain AT MOST {art_limit} artifact directions.** Focus on the highest-impact artifacts. Quality over quantity."
        if art_limit
        else ""
    }

Each strategy should:
1. Define a clear OBJECTIVE - what novel contribution we're building toward
2. Plan artifacts to execute NOW - specify type, objective, approach, and depends_on for each
3. Account for parallel execution - all strategies and all planned artifacts run simultaneously, their artifacts are combined into one shared pool

{
        f'''Generate {num_strategies} COMPLEMENTARY strategies that will ALL run simultaneously.

**CRITICAL — COMPLEMENTARY, NOT CONFLICTING:**
- Each strategy should approach the hypothesis from a DIFFERENT angle or explore a different facet
- Strategies must NOT duplicate effort — no two strategies should plan the same artifact for the same purpose
- Together they should cover more ground than a single strategy could alone
- Think of them as {num_strategies} research teams working simultaneously — how would you split the work so each team's output makes the others' more valuable?'''
        if num_strategies > 1
        else ""
    }
</task>"""


# =============================================================================
# HELPERS
# =============================================================================


def _build_previous_strategy_section(previous_strategies: list[dict]) -> str:
    """Build previous strategy section from all strategies of the previous iteration."""
    if not previous_strategies:
        return """<previous_strategies>
No previous strategies exist. This is the FIRST iteration.
</previous_strategies>"""

    strategies_text = to_prompt_yaml_list(previous_strategies, label="Strategy")

    return f"""<previous_strategies>
Strategies from the PREVIOUS iteration. You can CONTINUE these directions,
ADAPT based on what worked and what didn't in the artifacts produced, or PIVOT if results suggest a better path.

{strategies_text}
</previous_strategies>"""


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    hypothesis: dict,
    artifacts: list[BaseArtifact],
    current_iteration: int,
    max_iterations: int,
    previous_strategies: list[dict] | None = None,
    allowed_artifacts: list[str] | None = None,
    num_strategies: int = 1,
    art_limit: int | None = None,
    artifact_context_per_type: int = 10,
    reviewer_feedback_text: str | None = None,
    paper_text: str | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for strategy generation."""
    hypo_filtered = {
        k: v
        for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    remaining = max_iterations - current_iteration + 1

    prompt = PROMPT(
        hypothesis_text=to_prompt_yaml(hypo_filtered),
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        remaining=remaining,
        previous_strategy_section=_build_previous_strategy_section(previous_strategies or []),
        artifact_planning_guidance=get_artifact_planning(allowed_artifacts),
        artifact_types_text=get_artifact_context(allowed_artifacts),
        artifact_scope_text=get_artifact_scope(allowed_artifacts),
        existing_artifacts_text=LLMPromptModel.list_to_prompt_yaml(
            artifacts,
            label="Item",
            include={
                "id",
                "type",
                "title",
                "summary",
                "out_dependency_files",
                "workspace_path",
                "out_expected_files",
            },
            strip_nulls=True,
        ),
        narratives_text=paper_text or "",
        num_strategies=num_strategies,
        art_limit=art_limit,
        reviewer_feedback_text=reviewer_feedback_text or "",
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()


# =============================================================================
# RETRY PROMPT BUILDERS (for artifact verification failures)
# =============================================================================


def build_artifact_retry_prompt(
    verification: dict,
    num_strategies_requested: int,
    min_valid_artifacts: int | None = None,
    art_limit: int | None = None,
) -> str:
    """Build retry prompt for artifact verification failures.

    Reports ALL problems (limit, deps, types, etc.) in a single prompt so the LLM
    can fix everything at once.

    Note: ID errors no longer occur since IDs are code-assigned after LLM output.

    Args:
        verification: Dict from verify_strategies()
        num_strategies_requested: How many strategies were requested
        min_valid_artifacts: Minimum valid artifacts required (optional, for error message)
        art_limit: Max artifact directions per strategy (optional, for error message)

    Returns:
        Retry prompt string explaining ALL issues and requesting fixes
    """
    lines = [
        "<verification_results>",
        "Your previous response had issues that need fixing:",
        "",
    ]

    count_errors = verification.get("count_errors", [])
    dep_errors = verification.get("dep_errors", [])
    type_errors = verification.get("type_errors", [])
    limit_errors = verification.get("limit_errors", [])
    strategies_received = verification.get("strategies_received", 0)
    valid_artifact_count = verification.get("valid_artifact_count", 0)
    total_artifact_count = verification.get("total_artifact_count", 0)

    # Count mismatch
    if count_errors:
        lines.append("STRATEGY COUNT ERROR:")
        lines.append(f"  Requested: {num_strategies_requested} strategies")
        lines.append(f"  Received: {strategies_received} strategies")
        lines.append("")

    # Artifact limit exceeded
    if limit_errors:
        lines.append(f"ARTIFACT LIMIT EXCEEDED (max {art_limit} artifact directions per strategy):")
        for err in limit_errors:
            lines.append(f"  - {err}")
        lines.append("")

    # Type errors (artifact type not in allowed_artifacts)
    if type_errors:
        lines.append("ARTIFACT TYPE ERRORS (only certain artifact types are allowed):")
        for err in type_errors:
            lines.append(f"  - {err}")
        lines.append("")

    # Dependency errors
    if dep_errors:
        lines.append(
            "DEPENDENCY ERRORS (depends_on can ONLY reference IDs from <existing_artifacts>):"
        )
        for err in dep_errors:
            lines.append(f"  - {err}")
        lines.append("")

    # Valid artifact count (if min_valid_artifacts check failed)
    if min_valid_artifacts is not None and valid_artifact_count < min_valid_artifacts:
        lines.append("INSUFFICIENT VALID ARTIFACTS:")
        lines.append(f"  Required: at least {min_valid_artifacts} valid artifacts")
        lines.append(f"  Found: {valid_artifact_count} valid out of {total_artifact_count} total")
        lines.append(
            "  Artifacts with invalid types, duplicate IDs, or invalid dependencies don't count as valid."
        )
        lines.append("")

    lines.append("</verification_results>")
    lines.append("")
    lines.append("<task>")
    lines.append("Fix ALL issues above and regenerate your strategies:")
    lines.append("")

    step_num = 1
    if count_errors:
        lines.append(
            f"{step_num}. Generate EXACTLY {num_strategies_requested} strategies (you provided {strategies_received})"
        )
        step_num += 1

    if limit_errors:
        lines.append(f"{step_num}. Reduce artifact directions to AT MOST {art_limit} per strategy:")
        lines.append(f"   - Keep only the {art_limit} highest-impact artifacts per strategy")
        lines.append("   - Remove or merge lower-priority artifacts")
        step_num += 1

    if type_errors:
        lines.append(f"{step_num}. Fix artifact type errors:")
        lines.append("   - Only use artifact types listed in <available_artifact_types>")
        lines.append("   - Check the allowed types and change your artifacts to use valid types")
        step_num += 1

    if dep_errors:
        lines.append(f"{step_num}. Fix dependency errors:")
        lines.append(
            "   - depends_on is a list of {id, label} objects — every entry MUST have a non-empty short label"
        )
        lines.append("   - id can ONLY reference IDs from <existing_artifacts>")
        lines.append(
            "   - You CANNOT reference artifacts you are proposing in this strategy as dependencies (they all run in parallel)"
        )
        lines.append("   - Follow the dependency type rules (e.g., experiments require datasets)")
        lines.append("   - If no suitable existing artifacts exist, use depends_on: []")
        step_num += 1

    if min_valid_artifacts is not None and valid_artifact_count < min_valid_artifacts:
        lines.append(
            f"{step_num}. Ensure at least {min_valid_artifacts} artifacts are fully valid (correct types, no ID conflicts, valid dependencies)"
        )

    lines.append("")
    lines.append("Output the corrected JSON with the fixed strategies.")
    lines.append("</task>")

    return "\n".join(lines)
