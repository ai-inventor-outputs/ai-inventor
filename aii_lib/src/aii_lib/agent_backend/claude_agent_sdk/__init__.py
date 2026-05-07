"""
Claude Agent SDK implementation.

Sequential agent wrapper for Claude Agent SDK with streaming mode support.
"""

from .agent import Agent
from .models import (
    AgentOptions,
    AgentResponse,
    ExpectedFile,
    PromptResult,
    SessionType,
    SystemPromptPreset,
)
from .utils.execution.sdk_client import AgentProcessError, SubscriptionAccessError

# Re-export from utils.init_helpers for convenience
from .utils.init_helpers import (
    ALL_AGENTS,
    cleanup_agents,
    create_custom_tools_server,
    get_agent,
    list_agents,
    # MCP tool utilities
    load_tools_from_file,
    load_tools_from_files,
    # Agent management
    prepare_agents,
    setup_custom_tools,
)

__all__ = [
    # Core types
    "SessionType",
    "SystemPromptPreset",
    "AgentOptions",
    "ExpectedFile",
    "PromptResult",
    "AgentResponse",
    # Exceptions
    "AgentProcessError",
    "SubscriptionAccessError",
    # Main agent class
    "Agent",
    # Agent management
    "prepare_agents",
    "cleanup_agents",
    "get_agent",
    "list_agents",
    "ALL_AGENTS",
    # MCP tool utilities
    "load_tools_from_file",
    "load_tools_from_files",
    "create_custom_tools_server",
    "setup_custom_tools",
]
