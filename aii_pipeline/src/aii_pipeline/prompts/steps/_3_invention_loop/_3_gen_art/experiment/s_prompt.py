"""System prompt for experiment artifact.

Read top-to-bottom to understand the full prompt structure.
"""

from aii_pipeline.prompts.components.aii_context import get_aii_context
from aii_pipeline.prompts.components.research_practices import get_research_practices
from aii_pipeline.prompts.components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

{get_research_practices("experiment")}

<task>
Implement the research methodology as a production-ready experimental system.
Adapt your implementation approach based on the hypothesis and domain requirements.
</task>

<critical_requirements>
- Fully implement the methodology described in hypothesis
- Use appropriate frameworks based on research domain
- Load and process data from the specified data_filepath
- Complete working systems
- Handle all edge cases, errors, and exceptions properly
- Always implement baseline comparison method
- Keep final response under 300 characters
</critical_requirements>

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
    """Get the system prompt for experiment execution."""
    return PROMPT()
