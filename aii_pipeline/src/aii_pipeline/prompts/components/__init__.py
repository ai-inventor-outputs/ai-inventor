"""AI Inventor prompt components.

This module provides re-exports for convenient imports of prompt utilities.
Individual prompt files use simple static strings with .format() calls.

Usage:
    from aii_pipeline.prompts.components import (
        get_aii_context,
        get_resources_prompt,
        get_tool_calling_guidance,
    )
"""

# Context utilities
from .aii_context import (
    FocusArea,
    get_aii_context,
)

# Artifact planning
from .artifact_planning import get_artifact_planning

# Artifact summaries
from .artifact_summaries import get_artifact_context

# Data files utilities
from .data_files import get_reading_mini_preview_full

# Read skills instruction
from .read_skills import get_read_skills

# Resources utilities
from .resources import get_resources_prompt

# Todo utilities
from .todo import get_todo_header

# Tool calling utilities
from .tool_calling import get_tool_calling_guidance

# User folder utilities
from .user_folder import get_user_folder_prompt

__all__ = [
    # Context
    "get_aii_context",
    "FocusArea",
    # Artifact summaries
    "get_artifact_context",
    # Artifact planning
    "get_artifact_planning",
    # Resources
    "get_resources_prompt",
    # Tool calling
    "get_tool_calling_guidance",
    # Todo
    "get_todo_header",
    # Data files
    "get_reading_mini_preview_full",
    # Read skills
    "get_read_skills",
    # User folder
    "get_user_folder_prompt",
]
