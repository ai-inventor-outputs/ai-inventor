"""System prompt for strategy generation (Step 3.1: GEN_STRAT).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.research_practices import get_research_practices
from ....components.resources import ARTIFACT_RESOURCES, get_resources_prompt
from ....components.time_budgets import get_time_budgets_overview
from ....components.tool_calling import get_tool_calling_guidance, get_web_tool_guidance
from ....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT(context: str, resources_text: str) -> str:
    return f"""{context}

{resources_text}

{get_time_budgets_overview()}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

{get_research_practices("gen_strat")}

<principles>
1. FOCUS ON NOVELTY - every strategy must lead to a genuinely novel contribution
2. MAXIMIZE PARALLELIZATION - all artifacts in your strategy run in parallel
3. BUILD ON EXISTING WORK - use completed artifacts from previous iterations, learn from failures
4. ITERATE ON THE METHOD - a negative result is about the approach, not the hypothesis. Try different methods, parameters, data, or formulations within the hypothesis bounds.
5. DIAGNOSE BEFORE DECIDING - before each iteration, review what worked, what didn't, and why. Use that to choose what to try next. Gaps are action items, not conclusions.
6. SET DEPENDENCIES WISELY - depends_on is a list of {{id, label}} objects referencing existing artifacts; each label is a short free-text type (a word or two, e.g. "dataset", "validates", "extends") that tags how the dep is used
7. PLAN FOR DEPENDENCIES - if an artifact depends on another (e.g. experiments need datasets), ensure prerequisites exist first or plan them this iteration for the next
</principles>

{get_work_solo_reminder()}"""


# =============================================================================
# HELPERS
# =============================================================================


def _get_combined_resource_keys(
    allowed_artifacts: list[str] | None = None,
) -> list[str]:
    """Get union of all resource keys needed for the allowed artifact types."""
    if allowed_artifacts is None:
        allowed_artifacts = list(ARTIFACT_RESOURCES.keys())
    combined: set[str] = set()
    for art_type in allowed_artifacts:
        combined |= ARTIFACT_RESOURCES.get(art_type, set())
    return list(combined)


# =============================================================================
# EXPORTS
# =============================================================================


def get(allowed_artifacts: list[str] | None = None) -> str:
    """Build system prompt for strategy generation."""
    resource_keys = _get_combined_resource_keys(allowed_artifacts)
    return PROMPT(
        context=get_aii_context(focus="gen_strat"),
        resources_text=get_resources_prompt(include=resource_keys)
        if resource_keys
        else get_resources_prompt(),
    )
