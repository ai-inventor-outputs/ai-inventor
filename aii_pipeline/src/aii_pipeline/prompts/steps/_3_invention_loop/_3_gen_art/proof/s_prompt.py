"""System prompt for proof artifact - Seed-Prover inspired approach.

Read top-to-bottom to understand the full prompt structure.
"""

from .....components.aii_context import get_aii_context
from .....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    context = get_aii_context(focus="gen_art")
    return f"""{context}

<task>
Generate verified Lean 4 proofs using lemma-style proving.
Iterate based on compiler feedback, learning from failed attempts through self-summarization.
</task>

<tactics_reference>
See aii-lean skill "Mathlib Tactics Reference" section for the full list of automation and discovery tactics with examples.
</tactics_reference>

<critical_requirements>
- Use Lean 4 syntax (not Lean 3)
- No 'sorry' in final code — all proofs must be complete
</critical_requirements>

<common_mistakes_to_avoid>
- Check Nat vs Int vs Real types — Nat subtraction truncates at 0, Nat division is floor division, type coercions cause compiler errors
</common_mistakes_to_avoid>

{get_work_solo_reminder()}

"""


# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """Get the system prompt for proof execution."""
    return PROMPT()
