"""Tool loop helper - chat() for any LLM client.

Works with any client that has:
- chat(messages, tools, ...) -> response
- has_tool_calls(response) -> bool
- extract_tool_calls(response) -> list[dict]
- extract_json_from_response(response) -> str (optional)

Each emit lands on the active Run bus via ``current_run()._on(typed_msg)`` —
caller passes ``task_id`` / ``task_name`` so messages carry their identity.

Usage:
    from aii_lib import chat, ToolLoopResult, OpenRouterClient

    async with OpenRouterClient(api_key=key) as client:
        # With tools - automatic tool loop
        result = await chat(
            client=client,
            prompt="Generate a hypothesis about...",
            system="You are a researcher...",
            tools=abilities_to_openai_tools(["aii_web_tools__search"]),
            response_format=Hypothesis,
            task_id="task_x", task_name="Task X",
        )

        # Without tools - single call (loop runs once, exits immediately)
        result = await chat(
            client=client,
            prompt="Summarize this text...",
            system="You are an assistant...",
            task_id="task_x", task_name="Task X",
        )

        # If iterations exhausted but still has tool calls, continue:
        if result.hit_max_iterations:
            result = await chat(
                client=client,
                messages=result.messages,  # Resume with full context
                tools=tools,
                max_iterations=50,  # More iterations
                conversation_stats=result.stats,  # Continue tracking
                task_id="task_x", task_name="Task X",
            )
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aii_lib.run import get_current_run

from ._tool_helpers import (
    _emit_summary,
    _get_tool_abbrev,
    _validate_response_schema,
    execute_tool_calls,
)
from .openrouter.client import ConversationStats


@dataclass
class ToolLoopResult:
    """Result from chat() - supports resuming conversations."""

    response: Any  # Final LLM response
    stats: ConversationStats  # Aggregated stats
    messages: list[dict]  # Full message history (for resuming)
    iterations_used: int = 0  # How many iterations were used
    max_iterations: int = 0  # What the limit was

    @property
    def hit_max_iterations(self) -> bool:
        """True if loop exited due to reaching max iterations limit."""
        return self.iterations_used >= self.max_iterations

    @property
    def last_response_has_tool_calls(self) -> bool:
        """True if the last assistant message has tool calls (model wants more)."""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                return bool(tool_calls)
        return False


async def chat(
    client: Any,
    prompt: str | list | None = None,
    system: str | None = None,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    max_iterations: int = 100,
    response_format: type | None = None,
    schema_retries: int = 2,
    reasoning_effort: str | None = None,
    web_search_backend: str = "auto",
    timeout: float = 300,
    conversation_stats: ConversationStats | None = None,
    emit_summary: bool = True,
    local_tool_handlers: dict[str, Callable] | None = None,
    *,
    task_id: str = "",
    task_name: str = "",
) -> ToolLoopResult:
    """Chat with automatic tool loop - keeps calling until model stops.

    Each emitted message is routed onto the active Run bus via
    ``current_run()._on(typed_msg)``; the typed instance is built by
    :func:`aii_lib.llm_backend.tool_loop_tel_adapter.adapt`.

    Args:
        client: OpenRouterClient (the only LLM client; direct provider
            clients have been removed). For Claude calls, use the
            agent_backend (claude_agent_sdk) instead.
        prompt: User prompt - can be string or list of content blocks for multimodal
                (ignored if messages provided). List format:
                [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        system: System prompt (ignored if messages provided)
        messages: Existing message history (for resuming conversations)
        tools: List of tool definitions in OpenAI format
        max_iterations: Max tool loop iterations (safety limit)
        response_format: Pydantic model for structured output (applied on final call)
        schema_retries: Max retries if response_format validation fails (default 2)
        reasoning_effort: Reasoning effort level (low/medium/high)
        web_search_backend: Backend for web_search tool (auto/google/bing/etc.)
        timeout: Timeout per LLM call in seconds
        conversation_stats: Existing stats to continue (for resuming)
        emit_summary: Whether to emit summary at end (default True)
        local_tool_handlers: Dict mapping tool name to a callable that handles
            the tool locally (without going through the ability server).
        task_id: Task identity stamped onto every emitted Run-bus message.
        task_name: Display name stamped alongside task_id.

    Returns:
        ToolLoopResult with response, stats, messages, and continuation flag
    """
    from .tool_loop_tel_adapter import adapt as _adapt_tool_loop

    def _emit(raw: dict) -> None:
        run = get_current_run()
        if run is None:
            return
        run._on(_adapt_tool_loop(raw, task_id, task_name))

    # Build or use existing messages
    if messages is not None:
        chat_messages = list(messages)  # Copy to avoid mutation
        # Log the last user message if this is a continuation
        if conversation_stats is not None and chat_messages:
            last_msg = chat_messages[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    prompt_text = "\n".join(text_parts)
                else:
                    prompt_text = content
                _emit({"type": "prompt", "text": prompt_text})
    else:
        if prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.append({"role": "user", "content": prompt})

        # Emit system prompt and user prompt (only for new conversations)
        if system:
            _emit({"type": "s_prompt", "text": system})
        if isinstance(prompt, list):
            text_parts = [p.get("text", "") for p in prompt if p.get("type") == "text"]
            image_count = sum(1 for p in prompt if p.get("type") == "image_url")
            prompt_text = "\n".join(text_parts)
            if image_count:
                prompt_text += f"\n[{image_count} image(s) attached]"
        else:
            prompt_text = prompt
        _emit({"type": "prompt", "text": prompt_text})

    # Track stats across turns (use existing or create new)
    conv_stats = conversation_stats or ConversationStats()

    # Tool loop
    iteration = 0
    response = None
    is_first_turn = conversation_stats is None

    while iteration < max_iterations:
        iteration += 1

        emit_system_msg = is_first_turn and (iteration == 1)

        use_response_format = response_format if not tools else None

        response = await asyncio.wait_for(
            client.call(
                messages=chat_messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                response_format=use_response_format,
                emit_summary=False,
                emit_system=emit_system_msg,
                conversation_stats=conv_stats,
                task_id=task_id,
                task_name=task_name,
            ),
            timeout=timeout,
        )

        # Check if model wants to call tools
        if client.has_tool_calls(response):
            tool_calls = client.extract_tool_calls(response)

            local_results = []
            remote_calls = []
            for tc in tool_calls:
                if local_tool_handlers and tc["name"] in local_tool_handlers:
                    handler = local_tool_handlers[tc["name"]]
                    try:
                        result_str = handler(**tc.get("arguments", {}))
                    except Exception as e:
                        result_str = f"Error: {e}"
                    local_results.append(
                        {
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "original_name": tc["name"],
                            "result": result_str,
                            "error": None,
                            "cache_hit": False,
                        }
                    )
                else:
                    remote_calls.append(tc)

            remote_results = (
                await execute_tool_calls(
                    remote_calls,
                    web_search_backend=web_search_backend,
                )
                if remote_calls
                else []
            )
            tool_results = local_results + remote_results

            # Track cache hits for web search
            from aii_lib.abilities.endpoint_names import AII_WEB_SEARCH as _WS

            _WS_CACHE = f"{_WS}_cache_hit"
            for result in tool_results:
                if result.get("original_name") == _WS:
                    if result.get("cache_hit"):
                        conv_stats.tool_calls[_WS] = max(0, conv_stats.tool_calls.get(_WS, 0) - 1)
                        conv_stats.tool_calls[_WS_CACHE] = (
                            conv_stats.tool_calls.get(_WS_CACHE, 0) + 1
                        )

            if not response.choices:
                break
            assistant_msg = response.choices[0].message
            if not assistant_msg or not assistant_msg.tool_calls:
                break
            tool_call_dicts = []
            for tc in assistant_msg.tool_calls:
                try:
                    tool_call_dicts.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                except AttributeError as e:
                    raise AttributeError(f"Malformed tool call structure in response: {e}") from e
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": getattr(assistant_msg, "content", None),
                    "tool_calls": tool_call_dicts,
                }
            )

            # Add tool results to history + emit each
            for result in tool_results:
                result_content = json.dumps(
                    result.get("result", result.get("error", "No result")),
                    default=str,
                )
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result_content,
                    }
                )

                result_display = json.dumps(
                    result.get("result", result.get("error", "No result")),
                    indent=2,
                    default=str,
                )
                _emit(
                    {
                        "type": "tool_output",
                        "text": f"Tool: {result['name']}\nResult:\n{result_display}",
                        "tool": result["name"],
                        "tool_id": result["tool_call_id"],
                        "output": result.get("result"),
                        "backend": getattr(client, "provider_name", "unknown"),
                        "is_error": result.get("error") is not None,
                        "extras": {
                            "tool_name_abbrev": _get_tool_abbrev(result["name"], "_OUT"),
                        },
                    }
                )
        else:
            break

    model_finished = response and not client.has_tool_calls(response)

    # Final structured-output call if response_format requested + model done with tools
    if response_format and model_finished and tools:
        response = await asyncio.wait_for(
            client.call(
                messages=chat_messages,
                tools=None,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                emit_summary=emit_summary,
                emit_system=False,
                conversation_stats=conv_stats,
                task_id=task_id,
                task_name=task_name,
            ),
            timeout=timeout,
        )
        final_content = (
            getattr(response.choices[0].message, "content", "") if response.choices else ""
        )
        if final_content:
            chat_messages.append({"role": "assistant", "content": final_content})

    # Schema validation and retry
    if response_format and model_finished and hasattr(response_format, "model_validate_json"):
        schema_name = getattr(response_format, "__name__", "Schema")
        is_valid, validation_error = _validate_response_schema(response, response_format, client)

        retry_count = 0
        while not is_valid and retry_count < schema_retries:
            retry_count += 1
            _emit(
                {
                    "type": "retry",
                    "text": f"Schema validation failed ({retry_count}/{schema_retries}): {validation_error[:200]}",
                    "extras": {
                        "retry_count": retry_count,
                        "max_retries": schema_retries,
                        "schema_name": schema_name,
                        "validation_error": validation_error[:500],
                    },
                }
            )

            # Append feedback and retry
            feedback = f"Your JSON response has validation errors:\n\n{validation_error}\n\nFix the JSON to match the required schema exactly. Output only valid JSON."
            chat_messages.append({"role": "user", "content": feedback})

            _emit({"type": "prompt", "text": feedback})

            response = await asyncio.wait_for(
                client.call(
                    messages=chat_messages,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    emit_summary=False,
                    emit_system=False,
                    conversation_stats=conv_stats,
                    task_id=task_id,
                    task_name=task_name,
                ),
                timeout=timeout,
            )

            final_content = (
                getattr(response.choices[0].message, "content", "") if response.choices else ""
            )
            if final_content:
                chat_messages.append({"role": "assistant", "content": final_content})

            is_valid, validation_error = _validate_response_schema(
                response, response_format, client
            )

        if not is_valid:
            _emit(
                {
                    "type": "schema_error",
                    "text": f"JSON schema validation failed after {retry_count} retries\nSchema: {schema_name}\nErrors: {validation_error[:300]}",
                    "extras": {
                        "retry_count": retry_count,
                        "max_retries": schema_retries,
                        "schema_name": schema_name,
                        "validation_error": validation_error[:500],
                    },
                }
            )

    result = ToolLoopResult(
        response=response,
        stats=conv_stats,
        messages=chat_messages,
        iterations_used=iteration,
        max_iterations=max_iterations,
    )

    if model_finished and emit_summary:
        _emit_summary(conv_stats, client, task_id=task_id, task_name=task_name)

    return result
