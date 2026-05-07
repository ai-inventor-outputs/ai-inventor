"""System prompt for visualization image generation.

Read top-to-bottom to understand the full prompt structure.
"""


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    from ....components.research_practices import get_research_practices
    from ....components.work_solo_reminder import get_work_solo_reminder

    return f"""{get_research_practices("viz_gen")}

{get_work_solo_reminder()}"""


# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """System prompt for generating visualization images."""
    return PROMPT()
