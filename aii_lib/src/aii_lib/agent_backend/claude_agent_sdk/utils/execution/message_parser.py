"""Message parsing utilities."""

from collections.abc import Callable
from datetime import UTC, datetime

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from .formatting import (
    TOOL_NAME_MAP_INPUT,
    TOOL_NAME_MAP_OUTPUT,
    _format_summary_line,
    format_mcp_data,
    format_tool_input,
    get_tool_abbrev,
    serialize_message_for_debug,
)


def parse_assistant_message(
    message: AssistantMessage,
    prompt_index: int,
    on_message_logged: Callable | None = None,
    last_tool_id: str | None = None,
    last_tool_name: str | None = None,
    tool_id_to_agent_name: dict | None = None,
    tool_id_to_tool_name: dict | None = None,
    model: str | None = None,
    tool_calls_count: dict | None = None,
    seen_tool_result_ids: set | None = None,
) -> tuple[str | None, str | None]:
    """
    Parse an AssistantMessage and extract text/tool blocks.

    Args:
        message: The AssistantMessage to parse
        prompt_index: Index of the current prompt
        on_message_logged: Optional callback for message events
        last_tool_id: Previous tool ID for tracking
        last_tool_name: Previous tool name for tracking
        tool_id_to_agent_name: Mapping of Task tool IDs to agent names (for subagent tracking)
        tool_id_to_tool_name: Mapping of all tool IDs to tool names (for result matching)
        model: Model name for accurate cost calculation
        tool_calls_count: Dict to track tool call counts {tool_name: count}

    Returns:
        Tuple of (new_last_tool_id, new_last_tool_name)
    """
    current_tool_id = last_tool_id
    current_tool_name = last_tool_name

    # Initialize tool_id_to_agent_name if not provided
    if tool_id_to_agent_name is None:
        tool_id_to_agent_name = {}

    # Initialize tool_id_to_tool_name if not provided
    if tool_id_to_tool_name is None:
        tool_id_to_tool_name = {}

    # Initialize tool_calls_count if not provided
    if tool_calls_count is None:
        tool_calls_count = {}

    # Initialize seen_tool_result_ids if not provided
    if seen_tool_result_ids is None:
        seen_tool_result_ids = set()

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Check if this message belongs to a subagent (Task tool)
    parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
    agent_context = ""
    subagent_id = None

    if parent_tool_use_id and parent_tool_use_id in tool_id_to_agent_name:
        # This message is from a subagent
        agent_name = tool_id_to_agent_name[parent_tool_use_id]
        # Format: agent_name + last 2 chars of tool ID
        tool_id_short = (
            parent_tool_use_id[-2:] if len(parent_tool_use_id) >= 2 else parent_tool_use_id
        )
        agent_context = f"{agent_name}:{tool_id_short}"
        subagent_id = parent_tool_use_id

    for block in message.content:
        if isinstance(block, TextBlock):
            # Log text blocks (Claude's responses)
            if on_message_logged:
                on_message_logged(
                    {
                        "type": "claude_msg",
                        "text": block.text,
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": "",
                        "tool_id": "",
                        "task_name": agent_context,
                        "is_error": False,
                        "backend": "claude_agent",
                        "extras": {
                            "subagent_id": subagent_id,
                            "parent_tool_use_id": parent_tool_use_id,
                            "raw_api_message": raw_api_message,
                            "text_block": block.text,
                        },
                    }
                )

        elif isinstance(block, ThinkingBlock):
            # Log thinking blocks (Claude's internal reasoning).
            # Skip zero-content blocks — the SDK occasionally emits a
            # ThinkingBlock whose .thinking is empty, which would surface
            # as a noisy ``agent_think|task_id|`` line in telemetry.
            if on_message_logged and block.thinking:
                on_message_logged(
                    {
                        "type": "thinking",
                        "text": block.thinking,
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": "",
                        "tool_id": "",
                        "task_name": agent_context,
                        "is_error": False,
                        "backend": "claude_agent",
                        "extras": {
                            "subagent_id": subagent_id,
                            "parent_tool_use_id": parent_tool_use_id,
                            "raw_api_message": raw_api_message,
                            "thinking": block.thinking,
                            "signature": getattr(block, "signature", None),
                        },
                    }
                )

        elif isinstance(block, ToolUseBlock):
            # Save tool info for matching with results
            current_tool_id = block.id
            current_tool_name = block.name

            # Track tool ID to name mapping for later result matching
            tool_id_to_tool_name[block.id] = block.name

            # Track tool call counts for summary metrics
            tool_calls_count[block.name] = tool_calls_count.get(block.name, 0) + 1

            # Check if this ToolUseBlock has parent_tool_use_id (for nested tool calls from subagents)
            block_parent_id = getattr(block, "parent_tool_use_id", None)
            if block_parent_id and not parent_tool_use_id:
                # Update agent_context for this specific tool use
                if block_parent_id in tool_id_to_agent_name:
                    agent_name = tool_id_to_agent_name[block_parent_id]
                    tool_id_short = (
                        block_parent_id[-2:] if len(block_parent_id) >= 2 else block_parent_id
                    )
                    agent_context = f"{agent_name}:{tool_id_short}"
                    subagent_id = block_parent_id

            # Track Task tool invocations for subagent mapping
            if block.name == "Task":
                # Get fields for agent name extraction
                task_description = block.input.get("description", "")
                subagent_type = block.input.get("subagent_type", "")
                task_prompt = block.input.get("prompt", "")

                # Extract agent name with correct priority
                agent_name = "task"

                # PRIORITY 1: Use subagent_type (this is the actual agent name!)
                if subagent_type:
                    agent_name = subagent_type.replace("-", "_")

                # PRIORITY 2: Extract from prompt (fallback if no subagent_type)
                elif task_prompt and ":" in task_prompt[:100]:
                    # Check if prompt starts with "agent_name:" pattern
                    potential_agent_name = task_prompt.split(":", 1)[0].strip()
                    # Validate it looks like an agent name (short, no spaces/special chars)
                    if len(potential_agent_name) < 30 and not any(
                        c in potential_agent_name for c in ["\n", "\t", "  "]
                    ):
                        agent_name = potential_agent_name

                # PRIORITY 3: Use description (last resort)
                elif task_description and len(task_description) < 30:
                    agent_name = task_description.replace(" ", "_")

                tool_id_to_agent_name[block.id] = agent_name

            # Format tool input
            message_text = format_tool_input(
                block.name, block.input, tool_id_to_agent_name.get(block.id, "")
            )
            tool_name_abbrev = TOOL_NAME_MAP_INPUT.get(
                block.name, get_tool_abbrev(block.name, "_IN")
            )

            # Skip internal SDK stash reads. When a tool returns oversized
            # output, the SDK saves it to ``.claude/projects/<run>/<sid>/
            # tool-results/<id>.txt`` and the agent later reads it back.
            # That reread is plumbing — surfacing it as a "Read" pill in
            # the activity feed is just noise. Mark the tool_use_id seen
            # so the matching tool_result also gets skipped via the
            # existing dedup gate.
            file_path = (
                (block.input or {}).get("file_path") if isinstance(block.input, dict) else None
            )
            is_stash_read = (
                block.name == "Read"
                and isinstance(file_path, str)
                and "/.claude/projects/" in file_path
                and "/tool-results/" in file_path
            )
            if is_stash_read:
                if block.id:
                    seen_tool_result_ids.add(block.id)

            # Log tool use
            if on_message_logged and not is_stash_read:
                on_message_logged(
                    {
                        "type": "tool_input",
                        "text": message_text,
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": block.name,
                        "tool_id": block.id,
                        "task_name": agent_context,
                        "is_error": False,
                        "input": block.input,
                        "backend": "claude_agent",
                        "extras": {
                            "subagent_id": subagent_id,
                            "parent_tool_use_id": parent_tool_use_id,
                            "raw_api_message": raw_api_message,
                            "tool_name_abbrev": tool_name_abbrev,
                        },
                    }
                )

        elif isinstance(block, ToolResultBlock):
            # Get tool name from last tracked tool
            tool_use_id_for_result = (
                block.tool_use_id if hasattr(block, "tool_use_id") else current_tool_id
            )

            # Deduplication: skip if we've already logged this tool result
            if tool_use_id_for_result and tool_use_id_for_result in seen_tool_result_ids:
                continue
            if tool_use_id_for_result:
                seen_tool_result_ids.add(tool_use_id_for_result)

            tool_name_abbrev = TOOL_NAME_MAP_OUTPUT.get(
                current_tool_name or "",
                get_tool_abbrev(current_tool_name or "", "_OUT"),
            )

            # Format tool output based on tool type
            output_text = str(block.content) if block.content else ""
            if (
                current_tool_name
                and current_tool_name.startswith("mcp__")
                and block.content is not None
            ):
                # Format MCP tool outputs with each field on its own line
                output_text = format_mcp_data(block.content)

            # Format message text with tool name prefix (consistent with OpenRouter format)
            display_text = f"Tool: {current_tool_name or 'unknown'}\nResult:\n{output_text}"

            # Log tool results
            if on_message_logged:
                on_message_logged(
                    {
                        "type": "tool_output",
                        "text": display_text,
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": current_tool_name or "",
                        "tool_id": tool_use_id_for_result,
                        "task_name": agent_context,
                        "is_error": block.is_error,
                        "output": block.content,
                        "backend": "claude_agent",
                        "extras": {
                            "subagent_id": subagent_id,
                            "parent_tool_use_id": parent_tool_use_id,
                            "raw_api_message": raw_api_message,
                            "tool_name_abbrev": tool_name_abbrev,
                        },
                    }
                )

    return current_tool_id, current_tool_name


def parse_result_message(
    message: ResultMessage,
    prompt_index: int = 0,
    module_start_time: str | None = None,
    message_count: int = 0,
    model: str | None = None,
    tool_calls_count: dict | None = None,
) -> tuple[str, str, dict, int, dict | None]:
    """Parse a ResultMessage and build the standardized agent_summary dict.

    The returned ``summary_data`` carries every cost/token/tool field
    from the SDK plus a ``raw_api_message`` snapshot of the underlying
    ResultMessage. Callers emit it via ``message_callback`` so the run
    tree captures everything for future consumers (NodeStats reads the
    standardized cost/token fields; sinks may read extras for diagnostics
    or replay).

    Returns:
        (response_text, session_id, summary_data, num_turns, structured_output)
    """
    response_text = message.result or ""
    session_id = message.session_id or ""
    cost = message.total_cost_usd or 0.0

    # Capture complete API response for debugging
    raw_api_message = serialize_message_for_debug(message)

    # Raw usage dict from the SDK — fed straight into summary_data.
    usage_dict = getattr(message, "usage", {}) or {}

    duration_ms = getattr(message, "duration_ms", None) or 0
    duration_api_ms = getattr(message, "duration_api_ms", None) or 0
    num_turns = getattr(message, "num_turns", None) or 1
    is_error = getattr(message, "is_error", False)
    subtype = getattr(message, "subtype", None)  # "success" or error subtype
    structured_output = getattr(message, "structured_output", None)  # SDK native structured output

    runtime_seconds = duration_ms / 1000.0 if duration_ms else 0.0
    llm_time_seconds = duration_api_ms / 1000.0 if duration_api_ms else 0.0

    if is_error:
        status = subtype or "failed"
    else:
        status = subtype or "completed"

    tool_calls = tool_calls_count or {}

    # Server-side tool costs (flat fee per use, not per-token).
    # SDK WebSearch: $0.01/search. SDK WebFetch: FREE.
    # Our aii custom tools (aii_fast_web_search, aii_fast_web_fetch): FREE (billed separately).
    # All client-side tools (Write, Read, Bash, etc.): FREE (included in token pricing).
    # Format matches schema_utils.calculate_tool_costs(): {name: {"count", "unit", "total"}}
    _TOOL_COST_PER_CALL = {"WebSearch": 0.01}  # Only WebSearch has a per-call fee
    tool_costs: dict[str, dict] = {}
    tool_cost = 0.0
    for name, count in tool_calls.items():
        per_call = _TOOL_COST_PER_CALL.get(name, 0.0)
        if per_call > 0:
            tool_total = per_call * count
            tool_costs[name] = {"count": count, "unit": per_call, "total": tool_total}
            tool_cost += tool_total

    # Build summary data dict (always, for aggregation)
    summary_data = {
        # === Standardized SummaryMetrics fields ===
        "type": "summary",
        "text": _format_summary_line(
            num_turns=num_turns,
            input_tokens=(
                usage_dict.get("input_tokens", 0)
                + usage_dict.get("cache_creation_input_tokens", 0)
                + usage_dict.get("cache_read_input_tokens", 0)
            ),
            output_tokens=usage_dict.get("output_tokens", 0),
            tool_calls=tool_calls,
            tool_costs=tool_costs,
            total_cost=cost,
        ),
        "ts": datetime.now(UTC).isoformat(),
        "task_name": "",
        "tool": "",
        "tool_id": "",
        "is_error": is_error,
        "backend": "claude_agent",
        "total_cost": cost,  # SDK total_cost_usd (includes server tool costs)
        "token_cost": cost - tool_cost,  # Token cost only (total minus server tool fees)
        "tool_cost": tool_cost,  # Server-side tool fees (WebSearch: $0.01/call)
        "model": model or "",
        "status": status,
        "is_aggregated": False,
        "num_calls": num_turns,  # Number of conversation turns
        "runtime_seconds": runtime_seconds,
        "llm_time_seconds": llm_time_seconds,
        "input_tokens": usage_dict.get("input_tokens", 0),
        "output_tokens": usage_dict.get("output_tokens", 0),
        "reasoning_tokens": 0,  # Claude doesn't report reasoning tokens separately
        "cache_write_tokens": usage_dict.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage_dict.get("cache_read_input_tokens", 0),
        "tool_calls": tool_calls,
        "tool_costs": tool_costs,
        "extras": {
            "session_id": session_id,
            "final_result": response_text,
            "message_count": message_count,
            "raw_usage": usage_dict,
            "raw_api_message": raw_api_message,
        },
    }

    return response_text, session_id, summary_data, num_turns, structured_output
