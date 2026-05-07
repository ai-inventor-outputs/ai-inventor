"""Execution - main SDK streaming loop."""

import asyncio
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    UserMessage,
)
from claude_agent_sdk._errors import ProcessError

from ..utils.execution._parsers import parse_system_message, parse_user_message
from ..utils.execution.formatting import _fmt_tokens
from ..utils.execution.message_parser import (
    parse_assistant_message,
    parse_result_message,
)
from ..utils.execution.sdk_client import (
    AgentProcessError,
    StreamingExecutor,
    SubscriptionAccessError,
    _is_auth_crash,
)


class MessageTimeoutError(Exception):
    """Raised when a single SDK message exceeds message_timeout.

    Distinct from asyncio.TimeoutError so agent.py can handle it separately
    with its own retry budget (message_retries) before escalating to
    seq_prompt_retries.
    """


# Patterns that indicate subscription/access is unavailable (checked case-insensitive)
_SUBSCRIPTION_ERROR_PATTERNS = [
    "does not have access to claude",
    "organization does not have access",
    "hit your limit",
    "you've exceeded",
    "rate limit",
    "resets ",  # "resets 10pm (UTC)"
]


def _is_subscription_error_message(message: AssistantMessage) -> bool:
    """Check if an AssistantMessage indicates a subscription/access error.

    Detects via the SDK error field (authentication_failed) or by matching
    known error text patterns in TextBlock content.
    """
    # Check error field (SDK sets this for authentication errors)
    if getattr(message, "error", None) == "authentication_failed":
        return True

    # Check text content for known patterns
    for block in message.content:
        if isinstance(block, TextBlock):
            text_lower = block.text.lower()
            for pattern in _SUBSCRIPTION_ERROR_PATTERNS:
                if pattern in text_lower:
                    return True
    return False


async def execute_prompt_streaming(
    prompt: str,
    sdk_options: Any,  # ClaudeAgentOptions (pre-built in agent.py)
    execution_state: dict,
    message_callback: Any,
    message_timeout: int | None = None,
) -> tuple[str, str, dict, int, dict | None]:
    """Execute a single prompt using SDK streaming.

    The standardized ``summary_data`` dict (cost / tokens / tool calls)
    is emitted via ``message_callback`` when the ResultMessage arrives;
    NodeStats sums it via ``apply_leaf_summary``. We don't return cost
    or token-usage objects to the caller — the run tree owns them.

    Returns:
        (response_text, session_id, summary_data, num_turns, structured_output)
    """
    prompt_index = execution_state["prompt_index"]

    # Create executor
    executor = StreamingExecutor(sdk_options)

    # Initialize return values
    response_text = ""
    session_id = ""
    summary_data: dict = {}
    num_turns = 0
    structured_output = None

    # Track tool IDs across messages
    last_tool_id: str | None = None
    last_tool_name: str | None = None

    # Track Task tool invocations for subagent identification
    tool_id_to_agent_name: dict[str, str] = {}

    # Track all tool use IDs to names (for matching results to tools)
    tool_id_to_tool_name: dict[str, str] = {}

    # Track tool calls for summary metrics {tool_name: count}
    # Stored in execution_state so it survives timeout (dict is mutable)
    tool_calls_count: dict[str, int] = {}
    execution_state["_tool_calls_count"] = tool_calls_count

    # Deduplication: track seen tool result IDs to prevent duplicates
    # The SDK sometimes sends the same tool result in both AssistantMessage and UserMessage
    seen_tool_result_ids: set[str] = set()

    # Shared message dispatch — called from both timeout and non-timeout paths.
    # Closures capture all mutable state from the enclosing scope.
    def _process_message(message: Any) -> None:
        nonlocal response_text, session_id, summary_data, num_turns, structured_output
        nonlocal last_tool_id, last_tool_name

        if isinstance(message, SystemMessage):
            early_session_id, model = parse_system_message(
                message,
                prompt_index,
                message_callback,
                system_prompt=sdk_options.system_prompt,
                effort=getattr(sdk_options, "effort", None),
            )
            if early_session_id and not session_id:
                session_id = early_session_id
                execution_state["session_id"] = early_session_id
            if model:
                execution_state["current_model"] = model
        elif isinstance(message, AssistantMessage):
            # Check for subscription/access error BEFORE parsing
            # (parse logs 1 message, then we raise to stop the spam)
            is_sub_error = _is_subscription_error_message(message)
            last_tool_id, last_tool_name = parse_assistant_message(
                message,
                prompt_index,
                message_callback,
                last_tool_id,
                last_tool_name,
                tool_id_to_agent_name,
                tool_id_to_tool_name,
                model=execution_state["current_model"],
                tool_calls_count=tool_calls_count,
                seen_tool_result_ids=seen_tool_result_ids,
            )
            if is_sub_error:
                error_text = " ".join(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
                raise SubscriptionAccessError(error_text)
        elif isinstance(message, UserMessage):
            last_tool_id, last_tool_name = parse_user_message(
                message,
                prompt_index,
                message_callback,
                last_tool_id,
                last_tool_name,
                tool_id_to_agent_name,
                tool_id_to_tool_name,
                seen_tool_result_ids=seen_tool_result_ids,
            )
        elif isinstance(message, StreamEvent):
            # Mid-stream partial-message events. We only surface
            # ``message_delta`` (carries cumulative-within-call usage —
            # ``output_tokens`` ticks live, input/cache fields fixed for
            # the call). ``message_start`` / ``content_block_*`` /
            # ``message_stop`` are skipped: their content arrives via
            # AssistantMessage when the turn completes, and the per-tick
            # token counter is the only signal we need from streaming.
            ev = message.event or {}
            if ev.get("type") == "message_delta":
                usage = ev.get("usage") or {}
                in_t = int(usage.get("input_tokens") or 0)
                cw_t = int(usage.get("cache_creation_input_tokens") or 0)
                cr_t = int(usage.get("cache_read_input_tokens") or 0)
                out_t = int(usage.get("output_tokens") or 0)
                ctx_t = in_t + cw_t + cr_t
                # Console-friendly one-liner; mirrors the
                # ``Cost / In-Out`` style of agent_summary lines.
                text = f"Ctx Tok: {_fmt_tokens(ctx_t)} | Out Tok: {_fmt_tokens(out_t)}"
                message_callback(
                    {
                        "type": "message_delta",
                        "text": text,
                        "input_tokens": in_t,
                        "output_tokens": out_t,
                        "cache_read_input_tokens": cr_t,
                        "cache_creation_input_tokens": cw_t,
                    }
                )
        elif isinstance(message, ResultMessage):
            # Check for rate-limit text in the result (comes as normal response,
            # not ProcessError — must detect and raise to trigger wait-forever)
            result_text = (message.result or "").lower()
            if any(p in result_text for p in _SUBSCRIPTION_ERROR_PATTERNS):
                raise SubscriptionAccessError(message.result or "Rate limited")

            response_text, session_id, summary_data, num_turns, structured_output = (
                parse_result_message(
                    message,
                    prompt_index,
                    module_start_time=execution_state["module_start_time"],
                    message_count=execution_state["message_count"],
                    model=execution_state["current_model"],
                    tool_calls_count=tool_calls_count,
                )
            )
            if summary_data:
                # Always emit. Each ClaudeSDKClient instance produces exactly one
                # ResultMessage (one outer turn per ``client.query(string)``), so
                # there is no cumulative-overlap risk: NodeStats's
                # ``apply_leaf_summary`` correctly sums per-instance summaries
                # across multi-prompt / retry / injection scenarios.
                message_callback(summary_data)

    # Main execution loop - stream messages from SDK
    # When message_timeout is set, the SDK iterator runs in a background task
    # communicating via asyncio.Queue. This avoids cancelling anyio-managed
    # coroutines directly (which breaks anyio's cancel scope tracking).
    # The queue.get() timeout is pure asyncio and safe to interrupt.

    _SENTINEL_DONE = object()

    if message_timeout is not None:
        # --- Queue-based iteration with per-message timeout ---
        msg_queue: asyncio.Queue = asyncio.Queue()
        iter_error: list = []  # Mutable container to capture iterator exceptions

        async def _run_sdk_iterator():
            """Background task: pump SDK messages into the queue."""
            try:
                async for msg in executor.execute(
                    prompt, task_id=execution_state.get("run_id", "")
                ):
                    await msg_queue.put(msg)
                await msg_queue.put(_SENTINEL_DONE)
            except Exception as exc:
                iter_error.append(exc)
                await msg_queue.put(_SENTINEL_DONE)

        iter_task = asyncio.create_task(_run_sdk_iterator())
        try:
            while True:
                try:
                    message = await asyncio.wait_for(msg_queue.get(), timeout=message_timeout)
                except TimeoutError:
                    # No SDK message within message_timeout — trigger fork+resume
                    message_callback(
                        {
                            "type": "warning",
                            "text": f"Message-level timeout ({message_timeout}s) — triggering fork+resume",
                            "retry": True,
                        }
                    )
                    raise MessageTimeoutError(
                        f"No SDK message received for {message_timeout}s"
                    ) from None

                if message is _SENTINEL_DONE:
                    # Check if iterator ended with an error
                    if iter_error:
                        exc = iter_error[0]
                        if isinstance(exc, ProcessError):
                            if _is_auth_crash(exc):
                                raise SubscriptionAccessError(
                                    f"Failed to authenticate. {exc}"
                                ) from exc
                            error_msg = f"Agent subprocess terminated (will retry): {exc}"
                            message_callback({"type": "warning", "text": error_msg, "retry": True})
                            raise AgentProcessError(error_msg) from exc
                        raise exc
                    break

                _process_message(message)

        finally:
            # Clean up: cancel the background iterator task.
            # The SDK uses anyio internally — asyncio.Task.cancel() doesn't
            # reliably interrupt anyio coroutines, so use asyncio.wait with
            # a timeout to prevent the finally block from hanging indefinitely.
            if not iter_task.done():
                iter_task.cancel()
                try:
                    await asyncio.wait({iter_task}, timeout=10.0)
                except Exception:
                    pass

    else:
        # --- Standard iteration (no message_timeout) ---
        try:
            async for message in executor.execute(
                prompt, task_id=execution_state.get("run_id", "")
            ):
                _process_message(message)

        except ProcessError as e:
            if _is_auth_crash(e):
                raise SubscriptionAccessError(f"Failed to authenticate. {e}") from e
            error_msg = f"Agent subprocess terminated (will retry): {e}"
            message_callback({"type": "warning", "text": error_msg, "retry": True})
            raise AgentProcessError(error_msg) from e

    return response_text, session_id, summary_data, num_turns, structured_output
