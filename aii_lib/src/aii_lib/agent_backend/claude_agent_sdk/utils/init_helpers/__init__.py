"""
Initialization module for aii_lib agent backend.

Provides registry and loader functionality for agents and MCPs.
All configuration is loaded from .claude/ directory and .mcp.json.
"""

from .agents import (
    ALL_AGENTS,
    AgentDefinition,
    cleanup_agents,
    get_agent,
    list_agents,
    # Individual agent exports
    math_solver,
    math_tutor,
    palindrome_checker,
    prepare_agents,
    quick_calc,
    text_analyzer,
    text_master,
    text_transformer,
)
from .mcp_registry import (
    ALL_MCPS,
    # Individual MCP exports (if available)
    McpDefinition,
    get_mcp,
    list_mcps,
)

# Import individual MCPs if they exist
try:
    from .mcp_registry import context7
except ImportError:
    context7 = None

try:
    from .mcp_registry import hf_mcp_server
except ImportError:
    hf_mcp_server = None

try:
    from .mcp_registry import chrome_devtools
except ImportError:
    chrome_devtools = None

try:
    from .mcp_registry import shadcn
except ImportError:
    shadcn = None

from .mcp_loader import (
    cleanup_mcps,
    prepare_mcps,
)
from .mcp_tools import (
    create_custom_tools_server,
    load_tools_from_file,
    load_tools_from_files,
    setup_custom_tools,
)

__all__ = [
    # Agent management
    "AgentDefinition",
    "prepare_agents",
    "cleanup_agents",
    "get_agent",
    "list_agents",
    "ALL_AGENTS",
    "math_solver",
    "quick_calc",
    "math_tutor",
    "text_analyzer",
    "text_transformer",
    "palindrome_checker",
    "text_master",
    # MCP management
    "McpDefinition",
    "prepare_mcps",
    "cleanup_mcps",
    "list_mcps",
    "get_mcp",
    "ALL_MCPS",
    "context7",
    "hf_mcp_server",
    "chrome_devtools",
    "shadcn",
    # MCP tool utilities
    "load_tools_from_file",
    "load_tools_from_files",
    "create_custom_tools_server",
    "setup_custom_tools",
]
