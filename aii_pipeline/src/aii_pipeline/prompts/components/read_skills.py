"""Prompt component for instructing agents to read and follow skills."""


def get_read_skills(*skill_names: str) -> str:
    """Generate a TODO instruction to read and strictly follow the given skills.

    Args:
        *skill_names: Skill names (e.g. "aii-python", "aii-use-hardware").

    Returns:
        Formatted instruction string.
    """
    joined = ", ".join(skill_names)
    return f"Read and STRICTLY follow these skills: {joined}."
