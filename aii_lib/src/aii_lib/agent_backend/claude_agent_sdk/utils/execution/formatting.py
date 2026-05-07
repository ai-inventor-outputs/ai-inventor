"""Message formatting utilities — tool abbreviations, JSON formatting, summary lines."""

from __future__ import annotations

import re
from typing import Any


def _fmt_tokens(n: int) -> str:
    """Format token count compactly: 1,234 or 12.3K or 1.2M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _format_summary_line(
    num_turns: int,
    input_tokens: int,
    output_tokens: int,
    tool_calls: dict[str, int],
    tool_costs: dict[str, dict],
    total_cost: float,
) -> str:
    """Format a compact summary line for end-of-prompt display."""
    parts = [f"Summary: {num_turns} turns"]
    parts.append(f"Task In-Out: {_fmt_tokens(input_tokens)}-{_fmt_tokens(output_tokens)}")
    if tool_calls:
        tool_parts = []
        for name, count in tool_calls.items():
            entry = f"{name}: {count}"
            if name in tool_costs:
                entry += f" (${tool_costs[name]['total']:.2f})"
            tool_parts.append(entry)
        parts.append(f"Tools: {', '.join(tool_parts)}")
    tool_cost = sum(v["total"] for v in tool_costs.values()) if tool_costs else 0.0
    token_cost = total_cost - tool_cost
    parts.append(f"Cost: ${total_cost:.4f} (Token: ${token_cost:.4f}, Tools: ${tool_cost:.4f})")
    return " | ".join(parts)


def _clean_string_value(s: str) -> str:
    r"""Clean up string values for display (remove \\n\\t artifacts)."""
    s = re.sub(r"[\n\t]+", " ", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


def _clean_json_strings(obj: Any) -> Any:
    """Recursively clean string values in a JSON object."""
    if isinstance(obj, str):
        return _clean_string_value(obj)
    if isinstance(obj, dict):
        return {k: _clean_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_strings(item) for item in obj]
    return obj


def get_tool_abbrev(tool_name: str, suffix: str) -> str:
    """
    Get abbreviated tool name for display.

    Args:
        tool_name: Full tool name (e.g., "mcp__custom-tools__calculator_basic")
        suffix: "_IN" or "_OUT"

    Returns:
        Abbreviated name (e.g., "CALC_IN")
    """
    # For MCP tools, extract the actual tool name after the last "__"
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            # Last part is the actual tool name
            actual_tool_name = parts[-1]
        else:
            actual_tool_name = tool_name
    else:
        actual_tool_name = tool_name

    # Special case: openrouter tools all show as "OPER"
    if actual_tool_name.lower().startswith("openrouter"):
        return f"OPER{suffix}"

    # Special case: all HuggingFace MCP tools show as "HF"
    if "hf-mcp-server" in tool_name:
        return f"HF{suffix}"

    # Special case: chrome-devtools MCP tools show as "CHRM"
    if "chrome-devtools" in tool_name:
        return f"CHRM{suffix}"

    # Special case: context7 MCP tools show as "CTX7"
    if "context7" in tool_name:
        return f"CTX7{suffix}"

    # Special case: shadcn MCP tools show as "SHAD"
    if "shadcn" in tool_name:
        return f"SHAD{suffix}"

    # Special case: calculator MCP tools show as "CALC"
    if "calc" in tool_name or "calculator" in actual_tool_name.lower():
        return f"CALC{suffix}"

    # Special case: arxiv MCP tools - distinguish Search vs Fetch
    if "arxiv" in tool_name:
        if "search" in actual_tool_name.lower():
            return f"ARXS{suffix}"
        if "fetch" in actual_tool_name.lower():
            return f"ARXF{suffix}"
        return f"ARXV{suffix}"

    # Special case: wikipedia MCP tools show as "WIKS"
    if "wikipedia" in tool_name or "wikisearch" in actual_tool_name.lower():
        return f"WIKS{suffix}"

    # Try to create a smart 4-char abbreviation
    # Strategy 1: For camelCase names with multiple capitals (e.g., ArxivSearch, ArxivFetch)
    # Use first 3 chars + last capital letter
    capitals = [c for c in actual_tool_name if c.isupper()]
    if len(capitals) >= 2:
        # Get first 3 chars + last capital
        # ArxivSearch → "Arx" + "S" = "ARXS"
        # ArxivFetch → "Arx" + "F" = "ARXF"
        first_three = actual_tool_name[:3].upper()
        last_capital = capitals[-1]
        return f"{first_three}{last_capital}{suffix}"

    # Strategy 2: Default to first 4 chars
    return f"{actual_tool_name[:4].upper()}{suffix}"


# Map tool names to abbreviated format (aii_pipeline style)
TOOL_NAME_MAP_INPUT = {
    "Read": "READ_IN",
    "Write": "WRIT_IN",
    "Edit": "EDIT_IN",
    "Bash": "BASH_IN",
    "Grep": "GREP_IN",
    "Glob": "GLOB_IN",
    "Task": "TASK_IN",
    "TodoWrite": "TODO_IN",
    "ToolSearch": "TSRC_IN",
    "WebSearch": "SRCH_IN",
    "WebFetch": "FTCH_IN",
    "Skill": "SKIL_IN",
    "OpenRouter": "OPER_IN",
}

TOOL_NAME_MAP_OUTPUT = {
    "Read": "READ_OUT",
    "Write": "WRIT_OUT",
    "Edit": "EDIT_OUT",
    "Bash": "BASH_OUT",
    "Grep": "GREP_OUT",
    "Glob": "GLOB_OUT",
    "Task": "TASK_OUT",
    "TodoWrite": "TODO_OUT",
    "ToolSearch": "TSRC_OUT",
    "WebSearch": "SRCH_OUT",
    "WebFetch": "FTCH_OUT",
    "Skill": "SKIL_OUT",
    "OpenRouter": "OPER_OUT",
}


def _pretty_print_value(value: Any, indent: int = 2, max_chars: int | None = None) -> str:
    """Pretty-print a value, detecting and formatting JSON strings.

    Args:
        value: Any value to format
        indent: Indentation for JSON formatting
        max_chars: Optional max chars for truncation (from telemetry config)

    Returns:
        Formatted string
    """
    import json

    value_str = str(value) if not isinstance(value, str) else value

    # Try to parse as JSON if it looks like JSON
    if value_str.strip().startswith("{") or value_str.strip().startswith("["):
        try:
            parsed = json.loads(value_str)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            # Apply truncation if configured
            if max_chars and len(formatted) > max_chars:
                formatted = (
                    formatted[:max_chars] + f"\n... (+{len(value_str) - max_chars} chars truncated)"
                )
            return formatted
        except json.JSONDecodeError:
            pass

    # For non-JSON, just apply truncation if needed
    if max_chars and len(value_str) > max_chars:
        return value_str[:max_chars] + f"... (+{len(value_str) - max_chars} chars)"

    return value_str


def _get_telemetry_truncation() -> int | None:
    """Read the console sink's truncation setting off the live Run.

    Walks the live Run's :class:`RunSink` list looking for a
    :class:`ConsoleRunSink` so MCP tool-data formatting picks up the
    same truncation the operator configured. Returns ``None`` (no
    truncation override) if no Run is set up yet or no console sink
    is attached.
    """
    from aii_lib.run import get_current_run

    run = get_current_run()
    if run is None:
        return None
    try:
        from aii_lib.run.sinks.console import ConsoleRunSink
    except Exception:
        return None
    for sink in run.sinks:
        if isinstance(sink, ConsoleRunSink):
            return sink.truncation
    return None


def format_mcp_data(data: dict | list | str, max_chars: int | None = None) -> str:
    """
    Format MCP tool data (input or output) for display.

    For simple values: key: value on one line
    For complex/long values: pretty-printed JSON with optional truncation

    Args:
        data: Dictionary, list, or string data from MCP tool
        max_chars: Optional max chars for truncation. If None, uses telemetry config.

    Returns:
        Formatted string
    """
    import json

    # Get truncation from config if not provided
    if max_chars is None:
        max_chars = _get_telemetry_truncation()

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            # Pretty-print JSON values
            value_str = _pretty_print_value(value, indent=2, max_chars=max_chars)

            # For multiline values, format nicely
            if "\n" in value_str:
                lines.append(f"{key}:")
                # Indent each line of the value
                for line in value_str.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {value_str}")

        return "\n".join(lines) if lines else str(data)

    if isinstance(data, list):
        # For lists (like MCP output), format each item
        # First, try to unwrap MCP content blocks: [{"type": "text", "text": "..."}]
        unwrapped_items = []
        for item in data:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_content = item["text"]
                if isinstance(text_content, str):
                    # Try to parse nested JSON
                    try:
                        nested = json.loads(text_content)
                        # Clean string values (remove \n\t artifacts from HF descriptions)
                        nested = _clean_json_strings(nested)
                        unwrapped_items.append(nested)
                        continue
                    except json.JSONDecodeError:
                        # Not JSON, use text directly (cleaned)
                        unwrapped_items.append(_clean_string_value(text_content))
                        continue
            unwrapped_items.append(item)

        # If we unwrapped to a single item, just format that
        if len(unwrapped_items) == 1:
            item = unwrapped_items[0]
            if isinstance(item, (dict, list)):
                formatted = json.dumps(item, indent=2, ensure_ascii=False)
                if max_chars and len(formatted) > max_chars:
                    formatted = (
                        formatted[:max_chars]
                        + f"\n... (+{max(0, len(str(item)) - max_chars)} chars truncated)"
                    )
                return formatted
            return str(item)

        # Multiple items - format with indices
        items = []
        for i, item in enumerate(unwrapped_items):
            if isinstance(item, dict):
                # Pretty-print dict items
                try:
                    formatted = json.dumps(item, indent=2, ensure_ascii=False)
                    if max_chars and len(formatted) > max_chars:
                        formatted = (
                            formatted[:max_chars]
                            + f"\n... (+{max(0, len(str(item)) - max_chars)} chars)"
                        )
                    items.append(f"[{i}]:\n{formatted}")
                except (TypeError, ValueError):
                    items.append(f"[{i}]: {item!s}")
            else:
                items.append(f"[{i}]: {item!s}")
        return "\n\n".join(items) if items else str(data)

    # For raw strings, try to parse as JSON
    return _pretty_print_value(data, indent=2, max_chars=max_chars)


def format_tool_input(tool_name: str, tool_input: dict, agent_name: str = "") -> str:
    """Format tool input for display (aii_pipeline style)."""
    if tool_name == "Task":
        # Format: subagent_name:\nprompt (newline after colon)
        description = tool_input.get("description", "")
        prompt = tool_input.get("prompt", "")
        subagent_type = tool_input.get("subagent_type", agent_name or "task")

        # Put agent name and colon on first line, then newline, then prompt
        return f"{subagent_type}:\n{prompt}"

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        description = tool_input.get("description", "")
        if description:
            return f"{description}:\n{command}"
        return command

    if tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        todo_lines = []
        for i, todo in enumerate(todos, 1):
            status = todo.get("status", "pending")
            content_text = todo.get("content", "")
            # Add 1 newline before first todo, 2 newlines before others for better readability
            prefix = "\n" if i == 1 else "\n\n"
            todo_lines.append(f"{prefix}{i}. [{status}] {content_text}")
        return "".join(todo_lines) if todo_lines else "No todos"

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "unknown")
        content = tool_input.get("content", "")
        return f"File: {file_path}\n\n{content}"

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "unknown")
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")
        return f"File: {file_path}\nOLD: {old_string}\nNEW: {new_string}"

    if tool_name == "Read":
        # Just show file path
        file_path = tool_input.get("file_path", "unknown")
        return file_path

    if tool_name == "Glob":
        # Just show pattern in quotes
        pattern = tool_input.get("pattern", "")
        return f'Pattern: "{pattern}"'

    if tool_name == "Grep":
        # Show pattern in quotes
        pattern = tool_input.get("pattern", "")
        return f'Pattern: "{pattern}"'

    if tool_name == "WebSearch":
        # Show query and allowed_domains if present
        query = tool_input.get("query", "")
        allowed_domains = tool_input.get("allowed_domains", [])
        if allowed_domains:
            return f"{query} | allowed_domains: {allowed_domains}"
        return query

    if tool_name == "WebFetch":
        # Show URL and prompt
        url = tool_input.get("url", "")
        prompt = tool_input.get("prompt", "")
        if prompt:
            return f"URL: {url}\nPrompt: {prompt}"
        return f"URL: {url}"

    if tool_name == "Skill":
        # Show skill name and command
        skill = tool_input.get("skill", "unknown")
        command = tool_input.get("command", "")
        if command:
            return f"{skill}:\n{command}"
        return f"{skill}"

    if tool_name.startswith("mcp__"):
        # Format MCP tool inputs with each field on its own line
        return format_mcp_data(tool_input)

    return str(tool_input)


def serialize_message_for_debug(message: Any) -> dict:
    """Serialize a message object for full debug output.

    Captures all fields from the Anthropic API response.
    """
    try:
        # Try to get all attributes from the message object
        if hasattr(message, "__dict__"):
            debug_data = {}
            for key, value in message.__dict__.items():
                # Handle complex objects
                if hasattr(value, "__dict__"):
                    debug_data[key] = serialize_message_for_debug(value)
                elif isinstance(value, list):
                    debug_data[key] = [
                        serialize_message_for_debug(item)
                        if hasattr(item, "__dict__")
                        else str(item)
                        for item in value
                    ]
                else:
                    # Convert to string for non-serializable types
                    try:
                        import json

                        json.dumps(value)  # Test if serializable
                        debug_data[key] = value
                    except (TypeError, ValueError):
                        debug_data[key] = str(value)
            return debug_data
        return {"raw": str(message)}
    except Exception as e:
        return {"error": f"Failed to serialize message: {e}", "raw": str(message)}
