"""Model utilities for aii_lib."""

# Map full model IDs to short display names for task IDs and logs
_SHORT_NAMES = {
    "claude-opus-4-7": "opus",
    "claude-sonnet-4-5": "sonnet",
    "claude-haiku-4-5": "haiku",
}


def get_model_short(model: str) -> str:
    """Get short model name for task IDs and display.

    Examples:
        openai/gpt-5-mini -> gpt-5-mini
        claude-opus-4-7 -> opus
        claude-sonnet-4-5 -> sonnet
        anthropic/claude-opus-4.5 -> claude-opus-4.5

    Args:
        model: Full model name (e.g., "openai/gpt-5-mini" or "claude-opus-4-7")

    Returns:
        Short model name
    """
    # Strip provider prefix first
    short = model.split("/")[-1]
    # Apply known short names
    return _SHORT_NAMES.get(short, short)
