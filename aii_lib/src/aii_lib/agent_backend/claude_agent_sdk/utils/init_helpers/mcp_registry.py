"""
MCP Server Registry.

Discovers available MCP servers from project .mcp.json and provides
explicit selection for workspace preparation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class McpDefinition:
    """Definition of an MCP server."""

    name: str
    config: dict[str, Any]
    source_path: Path


def _find_mcp_json() -> Path | None:
    """Find .mcp.json file by searching upward from current directory."""
    current = Path.cwd()

    # Search upward for .mcp.json
    for parent in [current, *current.parents]:
        mcp_file = parent / ".mcp.json"
        if mcp_file.exists():
            return mcp_file

    return None


def _discover_mcps() -> dict[str, McpDefinition]:
    """
    Discover all available MCP servers from .mcp.json.

    Returns:
        Dict mapping MCP server names to McpDefinition objects
    """
    mcp_file = _find_mcp_json()

    if not mcp_file:
        return {}

    try:
        with open(mcp_file) as f:
            data = json.load(f)

        mcps = {}
        mcp_servers = data.get("mcpServers", {})

        for name, config in mcp_servers.items():
            mcps[name] = McpDefinition(name=name, config=config, source_path=mcp_file)

        return mcps

    except Exception as e:
        raise RuntimeError(f"Failed to parse MCP config from {mcp_file}: {e}") from e


# Discover all available MCPs at import time
_all_mcps = _discover_mcps()


def list_mcps() -> list[str]:
    """Get list of all available MCP server names."""
    return sorted(_all_mcps.keys())


def get_mcp(name: str) -> McpDefinition | None:
    """Get MCP definition by name."""
    return _all_mcps.get(name)


# Export individual MCP definitions as module attributes
# This allows: from aii_lib.agent_backend.claude_agent_sdk.utils.init_helpers import context7, hf_mcp_server
for mcp_name, mcp_def in _all_mcps.items():
    # Convert kebab-case to snake_case for Python variable names
    var_name = mcp_name.replace("-", "_")
    globals()[var_name] = mcp_def


# Export list of all MCP definitions
ALL_MCPS = list(_all_mcps.values())


__all__ = [  # noqa: PLE0604 — comprehension produces strings; ruff can't infer
    "McpDefinition",
    "list_mcps",
    "get_mcp",
    "ALL_MCPS",
    *[name.replace("-", "_") for name in _all_mcps],
]
