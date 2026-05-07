"""
Agent Backend - Agent SDK wrappers.

Currently supports:
- claude_agent_sdk/: Claude Agent SDK (sequential agent with streaming)

Architecture mirrors llm_backend/ for consistency.
"""

from typing import TYPE_CHECKING

# Re-export everything from claude_agent_sdk/ (default implementation)
from .claude_agent_sdk import (
    ALL_AGENTS,
    # Main agent class
    Agent,
    AgentOptions,
    AgentResponse,
    ExpectedFile,
    PromptResult,
    # Core types
    SessionType,
    SystemPromptPreset,
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

# Agent utilities (module-level helpers, formerly AgentInitializer/AgentFinalizer)
from .utils import (
    build_options,
    chain_validators,
    check_oversized_files,
    copy_dependencies,
    end_task,
    end_task_error,
    end_task_failure,
    end_task_success,
    end_task_timeout,
    ensure_servers,
    gen_dependency_prompt,
    generate_requirements,
    get_oversized_files_prompt,
    make_file_size_validator,
    read_metadata,
    setup_workspace,
    start_task,
)

# Type stubs for lazy imports
if TYPE_CHECKING:
    from .claude_agent_sdk.utils.cli import run_agent, run_agent_sync


# Lazy imports to avoid runpy warning
def __getattr__(name: str) -> object:
    """Lazy-load CLI functions on demand."""
    if name == "run_agent":
        from .claude_agent_sdk.utils.cli import run_agent

        return run_agent
    if name == "run_agent_sync":
        from .claude_agent_sdk.utils.cli import run_agent_sync

        return run_agent_sync
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core types
    "SessionType",
    "SystemPromptPreset",
    "AgentOptions",
    "ExpectedFile",
    "PromptResult",
    "AgentResponse",
    # Main agent class
    "Agent",
    # Run functions
    "run_agent",
    "run_agent_sync",
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
    # Agent utilities (module-level helpers, formerly AgentInitializer/AgentFinalizer)
    "setup_workspace",
    "copy_dependencies",
    "gen_dependency_prompt",
    "ensure_servers",
    "build_options",
    "start_task",
    "end_task",
    "end_task_success",
    "end_task_failure",
    "end_task_timeout",
    "end_task_error",
    "check_oversized_files",
    "get_oversized_files_prompt",
    "read_metadata",
    "generate_requirements",
    "chain_validators",
    "make_file_size_validator",
]

__version__ = "0.1.0"
