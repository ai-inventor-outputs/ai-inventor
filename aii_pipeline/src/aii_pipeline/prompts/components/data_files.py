"""Data file size variants guidance for prompts."""


def get_reading_mini_preview_full() -> str:
    """Guidance about data file size variants (mini/preview/full)."""
    return """Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER read directly with the Read tool (too large). Instead, extract values programmatically using grep, Bash, or a Python script (use aii-long-running-tasks skill for scripts)."""
