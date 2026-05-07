"""System prompt for research artifact.

Read top-to-bottom to understand the full prompt structure.
Research uses LLM with web search (not agent-based).
"""

from .....components.aii_context import get_aii_context
from .....components.tool_calling import (
    get_tool_calling_guidance,
    get_web_tool_guidance,
)
from .....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

<task>
Conduct thorough, unbiased research on the given topic.
Adapt your investigation approach based on the research question and domain.
</task>

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

<critical_requirements>
1. SOURCE DIVERSITY - Consult MANY sources (10+), not just the first few results
2. AVOID SELECTION BIAS - Actively seek contradicting viewpoints, not just confirming ones
3. TRIANGULATE - Cross-reference claims across multiple independent sources
4. ACKNOWLEDGE UNCERTAINTY - Be honest about confidence levels and limitations
5. SYNTHESIZE - Produce a coherent answer that accounts for conflicting evidence
</critical_requirements>

<common_mistakes_to_avoid>
</common_mistakes_to_avoid>

{get_work_solo_reminder()}

"""


# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """Get the system prompt for research execution."""
    return PROMPT()
