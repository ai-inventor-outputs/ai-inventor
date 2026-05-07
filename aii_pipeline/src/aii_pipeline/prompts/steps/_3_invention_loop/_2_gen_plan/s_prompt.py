"""System prompt for plan generation (Step 3.2: GEN_PLAN).

Expands artifact_directions from the strategy into detailed plans.
Each artifact type has its own plan schema with type-specific fields.

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.aii_context import get_aii_context
from ....components.resources import ARTIFACT_RESOURCES, get_resources_prompt
from ....components.time_budgets import get_time_budget_for_type
from ....components.tool_calling import get_tool_calling_guidance, get_web_tool_guidance
from ....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT(
    context: str,
    artifact_summaries: str,
    resources_text: str,
    artifact_type: str,
) -> str:
    return f"""{context}

<artifact_type_info>
You are expanding an artifact direction of type: {artifact_type.upper()}

{artifact_summaries}
</artifact_type_info>

{resources_text}

{get_time_budget_for_type(artifact_type)}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

<plan_guidelines>
You are expanding an artifact direction from the strategy into a detailed plan.
The artifact direction specifies what to do at a high level (type, objective, approach, dependencies).
Your job is to make it concrete and actionable as a detailed plan.
Use web research to look up technical details, verify feasibility, and find reference materials
that will make your plan more concrete and actionable for the executor.

GOOD PLANS:
- Make each component SPECIFIC and actionable (not vague platitudes)
- Consider both success AND failure scenarios
- Build on the approach in the artifact direction
- Add concrete details the executor needs

BAD PLANS:
- Vague hand-waving ("do research on X")
- Ignoring the approach in the artifact direction
- Missing critical details the executor needs
</plan_guidelines>

{get_work_solo_reminder()}"""


# =============================================================================
# HELPERS
# =============================================================================


def _get_artifact_description(artifact_type: str) -> str:
    """Get single artifact type description."""
    from ....components.artifact_summaries import (
        get_dataset_description,
        get_evaluation_description,
        get_experiment_description,
        get_proof_description,
        get_research_description,
    )

    DESCRIPTIONS = {
        "research": get_research_description,
        "proof": get_proof_description,
        "dataset": get_dataset_description,
        "experiment": get_experiment_description,
        "evaluation": get_evaluation_description,
    }

    if artifact_type in DESCRIPTIONS:
        return DESCRIPTIONS[artifact_type]()
    return ""


# =============================================================================
# EXPORTS
# =============================================================================


def get(artifact_type: str) -> str:
    """Build system prompt for expanding a specific artifact type."""
    artifact_description = _get_artifact_description(artifact_type)

    # Get resource keys for this artifact type
    resource_keys = list(ARTIFACT_RESOURCES.get(artifact_type, set()))

    return PROMPT(
        context=get_aii_context(focus="gen_plan"),
        artifact_summaries=artifact_description,
        resources_text=get_resources_prompt(include=resource_keys)
        if resource_keys
        else get_resources_prompt(),
        artifact_type=artifact_type,
    )
