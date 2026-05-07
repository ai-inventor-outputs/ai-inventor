"""
MCP Server Loader.

Prepares MCP servers for agent execution by creating .mcp.json in workspace.
"""

import json
from pathlib import Path

from aii_lib.run import emit

from .mcp_registry import McpDefinition, get_mcp


def prepare_mcps(
    mcps: list[McpDefinition | str],
    cwd: Path,
    run_id: str | None = None,
) -> Path | None:
    """
    Prepare MCP servers for execution by creating .mcp.json in workspace.

    Creates {cwd}/.mcp.json with only the selected MCP servers.

    Args:
        mcps: List of McpDefinition objects or MCP server name strings
        cwd: Working directory (workspace) where .mcp.json will be created
        run_id: Run ID for sequenced logging

    Returns:
        Path to created .mcp.json file, or None if no MCPs were prepared

    Example:
        >>> from aii_lib.agent_backend.claude_agent_sdk.utils.init_helpers import prepare_mcps, context7, hf_mcp_server
        >>> workspace = Path("/path/to/workspace")
        >>> prepare_mcps([context7, hf_mcp_server], cwd=workspace)
        PosixPath('/path/to/workspace/.mcp.json')
    """
    if not mcps:
        emit.status_public_warning(
            "No MCP servers selected - skipping .mcp.json creation", run_id=run_id
        )
        return None

    # Ensure cwd exists
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    # Build MCP configuration
    mcp_config = {"mcpServers": {}}

    for mcp in mcps:
        # Convert string to McpDefinition
        if isinstance(mcp, str):
            mcp_def = get_mcp(mcp)
            if not mcp_def:
                emit.status_public_warning(
                    f"MCP server '{mcp}' not found - skipping", run_id=run_id
                )
                continue
        else:
            mcp_def = mcp

        # Add to config
        mcp_config["mcpServers"][mcp_def.name] = mcp_def.config

    # Write .mcp.json
    mcp_json_path = cwd / ".mcp.json"

    with open(mcp_json_path, "w", encoding="utf-8") as f:
        json.dump(mcp_config, f, indent=2)

    emit.status_private_info(
        f"Prepared {len(mcp_config['mcpServers'])} MCP server(s) in {mcp_json_path}",
        run_id=run_id,
    )

    return mcp_json_path


def cleanup_mcps(cwd: Path) -> None:
    """
    Clean up MCP configuration from workspace.

    Removes {cwd}/.mcp.json

    Args:
        cwd: Working directory (workspace) to clean up
    """
    mcp_json_path = Path(cwd) / ".mcp.json"

    if mcp_json_path.exists():
        mcp_json_path.unlink()


__all__ = ["cleanup_mcps", "prepare_mcps"]
