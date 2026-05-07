"""User and system message parsers."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from claude_agent_sdk import SystemMessage, ToolResultBlock, UserMessage

from .formatting import (
    TOOL_NAME_MAP_OUTPUT,
    format_mcp_data,
    get_tool_abbrev,
    serialize_message_for_debug,
)


def parse_user_message(
    message: UserMessage,
    prompt_index: int,
    on_message_logged: Callable | None = None,
    last_tool_id: str | None = None,
    last_tool_name: str | None = None,
    tool_id_to_agent_name: dict | None = None,
    tool_id_to_tool_name: dict | None = None,
    seen_tool_result_ids: set | None = None,
) -> tuple[str | None, str | None]:
    """
    Parse a UserMessage and extract tool results (especially Task tool results).

    Args:
        message: The UserMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        last_tool_id: Previous tool ID for tracking
        last_tool_name: Previous tool name for tracking
        tool_id_to_agent_name: Mapping of Task tool IDs to agent names
        tool_id_to_tool_name: Mapping of all tool IDs to tool names (for result matching)

    Returns:
        Tuple of (new_last_tool_id, new_last_tool_name)
    """
    current_tool_id = last_tool_id or ""
    current_tool_name = last_tool_name or ""

    # Initialize tool_id_to_agent_name if not provided
    if tool_id_to_agent_name is None:
        tool_id_to_agent_name = {}

    # Initialize tool_id_to_tool_name if not provided
    if tool_id_to_tool_name is None:
        tool_id_to_tool_name = {}

    # Initialize seen_tool_result_ids if not provided
    if seen_tool_result_ids is None:
        seen_tool_result_ids = set()

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Check if message content is a list and contains ToolResultBlocks
    if isinstance(message.content, list):
        for content_block in message.content:
            if isinstance(content_block, ToolResultBlock):
                # Extract tool result info
                tool_use_id = (
                    content_block.tool_use_id
                    if hasattr(content_block, "tool_use_id")
                    else current_tool_id
                )

                # Deduplication: skip if we've already logged this tool result
                if tool_use_id and tool_use_id in seen_tool_result_ids:
                    continue
                if tool_use_id:
                    seen_tool_result_ids.add(tool_use_id)
                is_error = content_block.is_error if hasattr(content_block, "is_error") else False

                # Find which tool this result belongs to
                # Try tool_id_to_tool_name first, fallback to last_tool_name
                tool_name = tool_id_to_tool_name.get(tool_use_id, current_tool_name)

                # Format the result based on tool type
                display_content = ""

                # Handle Task tool results specially - extract just the text
                if tool_name == "Task":
                    # content_block.content can be:
                    # 1. A list of dicts: [{'type': 'text', 'text': '...'}]
                    # 2. A string representation: "[{'type': 'text', 'text': '...'}]"
                    # 3. Just a string: "result text"

                    if isinstance(content_block.content, list):
                        # Case 1: Already a list
                        if len(content_block.content) > 0:
                            first_element = content_block.content[0]
                            if isinstance(first_element, dict) and "text" in first_element:
                                display_content = first_element["text"]
                            else:
                                display_content = str(first_element)
                    elif isinstance(content_block.content, str):
                        # Case 2 or 3: String (might be JSON or plain text)
                        if content_block.content.startswith("["):
                            try:
                                import json

                                parsed = json.loads(content_block.content)
                                if isinstance(parsed, list) and parsed:
                                    first_element = parsed[0]
                                    if isinstance(first_element, dict) and "text" in first_element:
                                        display_content = first_element["text"]
                                    else:
                                        display_content = str(first_element)
                            except (
                                json.JSONDecodeError,
                                ValueError,
                                TypeError,
                                IndexError,
                                KeyError,
                            ):
                                # If JSON parsing fails, use the whole string
                                display_content = content_block.content
                        else:
                            # Plain text
                            display_content = content_block.content
                    else:
                        # Unknown type, convert to string
                        display_content = (
                            str(content_block.content) if content_block.content else ""
                        )
                else:
                    # For other tools, use raw content as string
                    display_content = str(content_block.content) if content_block.content else ""

                # Format MCP tool outputs with each field on its own line
                if (
                    tool_name
                    and tool_name.startswith("mcp__")
                    and content_block.content is not None
                ):
                    display_content = format_mcp_data(content_block.content)

                # Get abbreviated tool name for output
                tool_name_abbrev = TOOL_NAME_MAP_OUTPUT.get(
                    tool_name or "", get_tool_abbrev(tool_name or "", "_OUT")
                )

                # Check if this message belongs to a subagent
                parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
                agent_context = ""
                subagent_id = None

                # For TASK_OUT, subagent_id is the Task tool's OWN ID (tool_use_id), not its parent
                if tool_name == "Task" and tool_use_id and tool_use_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[tool_use_id]
                    tool_id_short = tool_use_id[-2:] if len(tool_use_id) >= 2 else tool_use_id
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = tool_use_id  # Use Task tool's own ID
                # For other tools, use parent_tool_use_id as before
                elif parent_tool_use_id and parent_tool_use_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[parent_tool_use_id]
                    tool_id_short = (
                        parent_tool_use_id[-2:]
                        if len(parent_tool_use_id) >= 2
                        else parent_tool_use_id
                    )
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = parent_tool_use_id

                # Format message text with tool name prefix (consistent with OpenRouter format)
                display_text = f"Tool: {tool_name or 'unknown'}\nResult:\n{display_content if display_content else ''}"

                # Log tool result
                if on_message_logged:
                    on_message_logged(
                        {
                            "type": "tool_output",
                            "text": display_text,
                            "ts": datetime.now(UTC).isoformat(),
                            "tool": tool_name or "",
                            "tool_id": tool_use_id,
                            "task_name": agent_context,
                            "is_error": is_error,
                            "output": content_block.content,
                            "backend": "claude_agent",
                            "extras": {
                                "subagent_id": subagent_id,
                                "parent_tool_use_id": parent_tool_use_id,
                                "raw_api_message": raw_api_message,
                                "tool_name_abbrev": tool_name_abbrev,
                                "display_content": display_content,
                            },
                        }
                    )

    return current_tool_id, current_tool_name


def parse_system_message(
    message: SystemMessage,
    prompt_index: int,
    on_message_logged: Callable | None = None,
    system_prompt: str | None = None,
    effort: str | None = None,
) -> tuple[str | None, str | None]:
    """Parse a SystemMessage and extract early session ID and initialization data.

    Args:
        message: The SystemMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        system_prompt: Optional system prompt from AgentOptions to include in metadata

    Returns:
        Tuple of (session_id, model) if available, else (None, None)
    """
    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Extract data from SystemMessage
    subtype = getattr(message, "subtype", None)
    data = getattr(message, "data", {})

    # Extract key fields from data
    session_id = None
    cwd = None
    model = None

    if isinstance(data, dict):
        session_id = data.get("session_id")
        cwd = data.get("cwd")
        model = data.get("model")

    # Use full model name (no shortening)

    # Log system message (skip if no model - e.g., continued conversations)
    if on_message_logged and model:
        # Build details dict with all system info
        details = {}
        if model:
            details["model"] = model
        if session_id:
            details["Session ID"] = session_id
        if cwd:
            # Show last part of path
            cwd_short = Path(cwd).name if cwd else ""
            if cwd_short:
                details["Working Directory"] = cwd_short

        # Add counts for tools, skills, MCP servers
        if isinstance(data, dict):
            tools = data.get("tools", [])
            skills = data.get("skills", [])
            mcp_servers = data.get("mcp_servers", [])

            if tools:
                details["Tools"] = len(tools)
            if skills:
                details["Skills"] = len(skills)
            if mcp_servers:
                details["MCP Servers"] = len(mcp_servers)

            # Add permission mode if not default
            permission_mode = data.get("permissionMode")
            if permission_mode and permission_mode != "askForPermission":
                details["Permission"] = permission_mode

        # Prepare metadata with full data object for verbose logging
        metadata = {
            "subtype": subtype,
            "session_id": session_id,
            "cwd": cwd,
            "model": model,
        }

        # Include full data object if it's a dict (for verbose logging)
        if isinstance(data, dict):
            # Add system_prompt to data if provided (SDK doesn't include it)
            data_with_prompt = data.copy()
            if system_prompt is not None:
                data_with_prompt["system_prompt"] = system_prompt
            metadata["data"] = data_with_prompt
        else:
            # If data is not a dict, create one with system_prompt
            metadata["data"] = {"system_prompt": system_prompt} if system_prompt is not None else {}

        on_message_logged(
            {
                "type": "system",
                "text": "",  # Leave empty, will be built by preprocessor
                "ts": datetime.now(UTC).isoformat(),
                "model": model,
                "effort": effort,
                "details": details,
                "tool": "",
                "tool_id": "",
                "task_name": "",
                "is_error": False,
                "backend": "claude_agent",
                "extras": {
                    "subagent_id": None,
                    "parent_tool_use_id": None,
                    "raw_api_message": raw_api_message,
                    **metadata,
                },
            }
        )

    # Return full model name for consistent display across SYSTEM and SUMMARY
    return session_id, model
