"""Enum types and type definitions for configuration."""

from enum import Enum
from typing import Literal, NotRequired

from typing_extensions import TypedDict


class SystemPromptPreset(TypedDict):
    """
    System prompt preset configuration.

    Use this to configure Claude Code's system prompt with optional append.

    Example:
        {
            "type": "preset",
            "preset": "claude_code",
            "append": "Always include detailed docstrings in Python code."
        }
    """

    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]  # Optional: append custom instructions


class SessionType(Enum):
    """
    Session management type.

    - NEW: Start a new session (default)
    - RESUME: Resume from a previous session ID (continues original session)
    - FORK: Fork from a previous session ID (creates new branch)
    """

    NEW = "new"
    RESUME = "resume"
    FORK = "fork"
