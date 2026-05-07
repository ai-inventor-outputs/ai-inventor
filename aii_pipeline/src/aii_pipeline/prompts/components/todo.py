"""Standard TODO list header for prompts."""


def get_todo_header():
    """Return the standard TODO list header used across all prompts."""
    return """FIRST, add ALL of these to your todo list with "TodoWrite" tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.
"""
