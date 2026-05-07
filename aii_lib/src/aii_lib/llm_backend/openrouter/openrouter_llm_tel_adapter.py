"""Telemetry adapter for OpenRouter LLM backend.

Converts raw message dicts emitted by ``OpenRouterClient`` into typed
:class:`aii_lib.run.messages.LlmMessage` subclass instances. Each
returned message carries provider-uniform typed slots; truly
provider-specific quirks ride in ``extras``.
"""

from __future__ import annotations

from aii_lib.run.messages import (
    BaseMessage,
    LlmConfigMessage,
    LlmResponseMessage,
    LlmSummaryMessage,
    LlmSystemPromptMessage,
    LlmThinkMessage,
    LlmToolCallMessage,
    LlmUserPromptMessage,
)

_PROVIDER = "openrouter"


def adapt(raw: dict, task_id: str, task_name: str) -> BaseMessage:
    """Map a raw OpenRouter message dict to a typed Run-bus message."""
    msg_type = raw.get("type", "")
    text = raw.get("text", "") or ""
    extras = dict(raw.get("extras") or {})
    model = raw.get("model") or None
    backend = raw.get("backend", _PROVIDER)

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

    if msg_type == "system":
        return LlmConfigMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            provider=raw.get("provider") or extras.pop("provider", _PROVIDER),
            reasoning_effort=raw.get("reasoning_effort") or extras.pop("reasoning_effort", None),
            max_tokens=raw.get("max_tokens") or extras.pop("max_tokens", None),
            temperature=raw.get("temperature") or extras.pop("temperature", None),
            top_p=raw.get("top_p") or extras.pop("top_p", None),
            response_format=raw.get("response_format") or extras.pop("response_format", None),
            context_window=raw.get("context_window") or extras.pop("context_window", None),
            tools=raw.get("tools") or extras.pop("tools", []),
            extras=extras or None,
        )

    if msg_type == "or_msg":
        return LlmResponseMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            finish_reason=raw.get("finish_reason"),
            extras=extras or None,
        )

    if msg_type == "or_reasoning":
        return LlmThinkMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            extras=extras or None,
        )

    if msg_type == "or_refusal":
        return LlmResponseMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            finish_reason="refusal",
            extras=extras or None,
        )

    if msg_type == "or_tool_in":
        return LlmToolCallMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=model,
            tool=raw.get("tool", ""),
            tool_id=raw.get("tool_id", ""),
            input=raw.get("input"),
            extras=extras or None,
        )

    if msg_type == "summary":
        return LlmSummaryMessage(
            task_id=task_id,
            parent_id=task_id,
            text=text,
            backend=backend,
            model=raw.get("model") or model,
            total_cost=raw.get("total_cost", 0.0),
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            extras={
                **(extras or {}),
                "token_cost": raw.get("token_cost", 0.0),
                "tool_cost": raw.get("tool_cost", 0.0),
                "reasoning_tokens": raw.get("reasoning_tokens", 0),
                "cache_read_tokens": raw.get("cache_read_tokens", 0),
                "cache_write_tokens": raw.get("cache_write_tokens", 0),
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
        text=f"Unknown OpenRouter message type: {msg_type}. {text}",
        extras=extras,
    )
