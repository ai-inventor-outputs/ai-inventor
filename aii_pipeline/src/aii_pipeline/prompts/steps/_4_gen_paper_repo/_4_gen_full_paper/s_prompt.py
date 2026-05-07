"""System prompt for LaTeX compilation.

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.research_practices import get_research_practices
from ....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """System prompt for Claude Agent LaTeX compilation."""
    return get_research_practices("write_paper") + "\n\n" + get_work_solo_reminder()
