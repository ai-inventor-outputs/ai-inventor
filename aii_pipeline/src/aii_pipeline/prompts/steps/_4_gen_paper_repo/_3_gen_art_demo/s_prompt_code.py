"""System prompt for artifact demo generation (notebook conversion).

Read top-to-bottom to understand the full prompt structure.
"""

from ....components.work_solo_reminder import get_work_solo_reminder

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT() -> str:
    return """<conversion_philosophy>
**MINIMAL CHANGES — PRESERVE THE ORIGINAL CODE**

The goal is to make the artifact's code READABLE, UNDERSTANDABLE, and RUNNABLE in a short time
to someone reviewing the research, with the option to easily scale parameters back to original
values for a full run (which can take much longer). Think of this as annotating and reformatting,
not refactoring.

**DO:**
- Split the original script into logical notebook cells (imports, setup, processing, results)
- Add markdown cells BETWEEN code cells explaining what each section does and why
- Add inline comments where the logic is non-obvious
- Add a visualization/summary cell at the end showing key outputs
- Fix hardcoded file paths to use the GitHub data loading pattern

**DO NOT:**
- Rewrite functions or change algorithms
- Rename variables or restructure logic
- Add error handling, type hints, or "improvements" that weren't in the original
- Simplify or "clean up" the original code
- Remove any original comments or logic
- Change the computational approach

The reader should recognize the original script when looking at the notebook — it's the
same code, just split into cells with explanatory markdown between sections.
</conversion_philosophy>"""


# =============================================================================
# EXPORTS
# =============================================================================


def get() -> str:
    """System prompt for notebook conversion agent."""
    return PROMPT() + "\n\n" + get_work_solo_reminder()
