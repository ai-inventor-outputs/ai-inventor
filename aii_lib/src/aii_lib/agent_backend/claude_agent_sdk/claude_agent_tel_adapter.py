"""Telemetry adapter for Claude Agent SDK backend.

Converts raw message dicts emitted by ``message_parser.py``,
``streaming.py``, and ``config.py`` into typed
:class:`aii_lib.run.messages.AgentMessage` subclass instances. The
caller (``Agent._message_callback``) routes the typed instance via
``run._on(msg)`` directly.
"""

from __future__ import annotations

from aii_lib.run.messages import (
    AgentConfigMessage,
    AgentHookMessage,
    AgentMessageDeltaMessage,
    AgentResponseMessage,
    AgentRetryMessage,
    AgentSummaryMessage,
    AgentSystemPromptMessage,
    AgentThinkMessage,
    AgentToolCallMessage,
    AgentToolResultMessage,
    AgentUserPromptMessage,
    BaseMessage,
)


def adapt(raw: dict, task_id: str, task_name: str) -> BaseMessage:
    """Map a raw Claude Agent SDK message dict to a typed Run-bus message."""
    msg_type = raw.get("type", "")
    text = raw.get("text", "") or ""
    extras = dict(raw.get("extras") or {})
    model = raw.get("model") or extras.get("model") or None
    backend = raw.get("backend", "claude_agent")

    if msg_type == "claude_msg":
        return AgentResponseMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            extras=extras or None,
        )

    if msg_type == "thinking":
        # ``signature`` is a top-level AgentThinkMessage field; pop
        # it out of ``extras`` to avoid the duplicate that used to land
        # in both places. ``thinking`` duplicates ``text`` (block content),
        # so drop it from extras too.
        signature = extras.pop("signature", None)
        extras.pop("thinking", None)
        return AgentThinkMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            signature=signature,
            extras=extras or None,
        )

    if msg_type == "tool_input":
        return AgentToolCallMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            tool=raw.get("tool", ""),
            tool_id=raw.get("tool_id", ""),
            extras={**extras, "input": raw.get("input")}
            if raw.get("input") is not None
            else (extras or None),
        )

    if msg_type == "tool_output":
        return AgentToolResultMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            tool=raw.get("tool", ""),
            tool_id=raw.get("tool_id", ""),
            is_error=bool(raw.get("is_error", False)),
            extras={**extras, "output": raw.get("output")}
            if raw.get("output") is not None
            else (extras or None),
        )

    if msg_type == "system":
        data = extras.get("data") or {}
        session_id = extras.get("session_id")
        cwd = extras.get("cwd")
        tools = data.get("tools") or []
        skills = data.get("skills") or []
        permission_mode = data.get("permissionMode")
        reasoning_effort = raw.get("effort")
        if data.get("system_prompt"):
            extras["system_prompt"] = data["system_prompt"]
        _model = (model or extras.get("model")) or "unknown"
        _tools_count = len(tools) if tools else None
        _skills_count = len(skills) if skills else None
        mcp_servers = data.get("mcp_servers") or []
        _mcp_count = len(mcp_servers) if mcp_servers else None

        # Build text from all available fields
        _parts = [f"Model: {_model}"]
        if reasoning_effort:
            _parts.append(f"Effort: {reasoning_effort}")
        if session_id:
            _parts.append(f"Session: {session_id}")
        if cwd:
            _parts.append(f"CWD: {cwd}")
        if _tools_count:
            _parts.append(f"Tools: {_tools_count}")
        if _skills_count:
            _parts.append(f"Skills: {_skills_count}")
        if _mcp_count:
            _parts.append(f"MCP Servers: {_mcp_count}")
        if permission_mode:
            _parts.append(f"Permission: {permission_mode}")

        # Slim extras (was ``{**extras, ...}`` which re-emitted the full
        # SDK system message on every session start: ``data.tools`` /
        # ``data.skills`` / ``data.mcp_servers`` lists + the entire
        # system prompt). The summarized counts already ride on
        # ``text`` + dedicated fields; multi-session runs (resume,
        # interrupt+continue) accumulated multiple copies of the same
        # ~kB payload in clone_log per session.
        return AgentConfigMessage(
            task_id=task_id,
            parent_id=task_id,
            text=" | ".join(_parts),
            backend=backend,
            model=_model,
            cwd=cwd,
            permission_mode=permission_mode,
            reasoning_effort=reasoning_effort,
            extras={
                "session_id": session_id,
                "tools_count": _tools_count,
                "skills_count": _skills_count,
                "mcp_count": _mcp_count,
            },
        )

    if msg_type == "prompt":
        prompt_source = raw.get("prompt_source", "pipeline")
        if prompt_source not in ("pipeline", "human"):
            prompt_source = "pipeline"
        return AgentUserPromptMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            prompt_source=prompt_source,
            prompt_index=raw.get("prompt_index", 0),
            extras=extras or None,
        )

    if msg_type == "s_prompt":
        return AgentSystemPromptMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            prompt_index=raw.get("prompt_index", 0),
            extras=extras or None,
        )

    if msg_type == "message_delta":
        # StreamEvent.event["type"] == "message_delta" — mid-stream usage
        # update for the agent's CURRENT LLM call. Carries per-call (NOT
        # cumulative-across-calls) values: ``input_tokens`` is fixed for
        # the duration of one call, ``output_tokens`` grows tick-by-tick.
        return AgentMessageDeltaMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_read_input_tokens=raw.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=raw.get("cache_creation_input_tokens", 0),
            extras=extras or None,
        )

    if msg_type == "summary":
        return AgentSummaryMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=raw.get("model") or model,
            total_cost=raw.get("total_cost", 0.0),
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_read_tokens=raw.get("cache_read_tokens", 0),
            cache_write_tokens=raw.get("cache_write_tokens", 0),
            extras={
                **extras,
                "token_cost": raw.get("token_cost", 0.0),
                "tool_cost": raw.get("tool_cost", 0.0),
                "num_calls": raw.get("num_calls", 0),
                "runtime_seconds": raw.get("runtime_seconds", 0.0),
                "llm_time_seconds": raw.get("llm_time_seconds", 0.0),
                "reasoning_tokens": raw.get("reasoning_tokens", 0),
                "tool_calls": raw.get("tool_calls") or {},
                "tool_costs": raw.get("tool_costs") or {},
            },
        )

    if msg_type == "warning":
        # Producers MUST set ``retry=True`` on the raw dict (or in extras)
        # to be classified as a retry — untagged warnings are plain status.
        if bool(raw.get("retry") or extras.get("retry")):
            return AgentRetryMessage(
                task_id=task_id,
                parent_id=task_id,
                text=text,
                backend=backend,
                model=model,
                reason=text,
                extras=extras or None,
            )
        return BaseMessage(
            type="status_public_warning",
            task_id=task_id,
            parent_id=task_id,
            text=text,
            extras=extras or None,
        )

    if msg_type == "hook":
        hook_type = extras.get("hook_type") or raw.get("hook_type")
        return AgentHookMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            extras={**extras, "hook_type": hook_type},
        )

    # Unknown type: pass through as status_public_warning
    extras["original_type"] = msg_type
    extras["original_raw"] = {k: v for k, v in raw.items() if k != "extras"}
    return BaseMessage(
        type="status_public_warning",
        task_id=task_id,
        parent_id=task_id,
        text=f"Unknown Claude agent message type: {msg_type}. {text}",
        extras=extras,
    )
