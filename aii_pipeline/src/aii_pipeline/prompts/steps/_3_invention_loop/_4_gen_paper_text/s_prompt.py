"""System prompt for gen_paper_text (Step 3.4: GEN_PAPER_TEXT).

Paper text generation in the invention loop —
iterative paper drafting, not final publication.
"""

from ....components.aii_context import get_aii_context
from ....components.research_practices import get_research_practices
from ....components.tool_calling import get_tool_calling_guidance, get_web_tool_guidance
from ....components.work_solo_reminder import get_work_solo_reminder


def PROMPT(context: str) -> str:
    return f"""{context}

{get_research_practices("gen_paper_text")}

{get_web_tool_guidance()}

{get_tool_calling_guidance()}

{get_work_solo_reminder()}

"""


def get() -> str:
    """Get system prompt for paper text generation in the invention loop."""
    return PROMPT(context=get_aii_context(focus="gen_paper_text"))
