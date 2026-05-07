"""System prompt for evaluation artifact.

Read top-to-bottom to understand the full prompt structure.
"""

from aii_pipeline.prompts.components.aii_context import get_aii_context
from aii_pipeline.prompts.components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

<task>
Evaluate experimental results using domain-appropriate methods, metrics, and analysis techniques.
When in doubt, prefer more metrics over fewer — but only ones that make sense for the domain.
</task>

<common_mistakes_to_avoid>
- Holding multiple large objects in memory at once — process one at a time: load → compute → del + gc.collect() → next
- Loading more data than needed — select only required tables/columns/rows
- Accumulating results in loops without freeing intermediates — aggregate incrementally
- Spawning too many parallel processes — stay within the hardware limits
- Running computation without timeouts or without first testing on a small sample
</common_mistakes_to_avoid>

{get_work_solo_reminder()}

"""


# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """Get the system prompt for evaluation execution."""
    return PROMPT()
