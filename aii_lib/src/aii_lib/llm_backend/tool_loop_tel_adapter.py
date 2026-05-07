"""Telemetry adapter for tool_loop.py shared messages.

Converts raw message dicts emitted by ``tool_loop.py`` (shared by all
LLM backends) into typed :mod:`aii_lib.run.messages` ``Llm*`` instances.
"""

from __future__ import annotations

from aii_lib.run.messages import (
    BaseMessage,
    LlmRetryMessage,
    LlmSchemaErrorMessage,
    LlmSummaryMessage,
    LlmSystemPromptMessage,
    LlmToolResultMessage,
    LlmUserPromptMessage,
)


def adapt(raw: dict, task_id: str, task_name: str) -> BaseMessage:
    """Map a raw tool_loop message dict to a typed Run-bus message."""
    msg_type = raw.get("type", "")
    text = raw.get("text", "") or ""
    extras = dict(raw.get("extras") or {})
    backend = raw.get("backend") or None

    if msg_type == "prompt":
        return LlmUserPromptMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            extras=extras or None,
        )

    if msg_type == "s_prompt":
        return LlmSystemPromptMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            extras=extras or None,
        )

    if msg_type == "tool_output":
        return LlmToolResultMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            tool=raw.get("tool", ""),
            tool_id=raw.get("tool_id", ""),
            output=raw.get("output"),
            is_error=bool(raw.get("is_error", False)),
            extras=extras or None,
        )

    if msg_type == "retry":
        return LlmRetryMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            attempt=extras.pop("retry_count", 0) or raw.get("attempt", 0),
            max_attempts=extras.pop("max_retries", 0) or raw.get("max_attempts", 0),
            reason=extras.pop("validation_error", "") or raw.get("reason", ""),
            schema_name=extras.pop("schema_name", "") or raw.get("schema_name", ""),
            extras=extras or None,
        )

    if msg_type == "schema_error":
        return LlmSchemaErrorMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            schema_name=extras.pop("schema_name", "") or raw.get("schema_name", ""),
            validation_error=extras.pop("validation_error", "") or raw.get("validation_error", ""),
            attempts_made=extras.pop("retry_count", 0) or raw.get("attempts_made", 0),
            extras=extras or None,
        )

    if msg_type == "summary":
        return LlmSummaryMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=raw.get("model"),
            total_cost=raw.get("total_cost", 0.0),
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_read_tokens=raw.get("cache_read_tokens", 0),
            cache_write_tokens=raw.get("cache_write_tokens", 0),
            extras={
                **(extras or {}),
                "token_cost": raw.get("token_cost", 0.0),
                "tool_cost": raw.get("tool_cost", 0.0),
                "reasoning_tokens": raw.get("reasoning_tokens", 0),
                "num_calls": raw.get("num_calls", 0),
                "runtime_seconds": raw.get("runtime_seconds", 0.0),
                "llm_time_seconds": raw.get("llm_time_seconds", 0.0),
                "tool_calls": raw.get("tool_calls") or {},
                "tool_costs": raw.get("tool_costs") or {},
                "finish_reason": raw.get("finish_reason"),
            },
        )

    extras["original_type"] = msg_type
    return BaseMessage(
        type="status_public_warning",
        task_id=task_id,
        parent_id=task_id,
        text=f"Unknown tool_loop message type: {msg_type}. {text}",
        extras=extras,
    )
