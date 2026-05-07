"""Error recovery for prompt execution — message timeout and subscription errors.

These are extracted from Agent methods. They operate on an AgentState
container so they can update session_id, failure_reason, etc. without
needing a reference to the Agent instance.
"""

from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger

from aii_lib.run import emit

from .config import initialize_execution
from .prompts import build_continue_prompt
from .streaming import (
    AgentProcessError,
    MessageTimeoutError,
    SubscriptionAccessError,
    execute_prompt_streaming,
)


def _extend_worker_ttl(seconds: float) -> None:
    """Extend the worker pod self-destruct deadline (no-op if not on a worker)."""
    if seconds <= 0:
        return
    try:
        from aii_runpod.comms.entrypoint import extend_self_destruct

        extend_self_destruct(seconds)
    except ImportError:
        pass


@dataclass
class AgentState:
    """Mutable state bag shared between Agent and error recovery functions."""

    options: Any  # AgentOptions
    session_id: str | None = None
    last_failure_reason: str | None = None
    capacity_wait_total: float = 0.0
    deadlines: dict[str, float | None] = field(default_factory=dict)


async def execute_with_message_retry(
    state: AgentState,
    effective_prompt: str,
    original_prompt: str,
    sdk_options: Any,
    sdk_options_cache: Any,
    execution_state: dict,
    message_callback: Any,
    prompt_index: int,
    timeout: float | None,
    with_output_format: bool,
    get_monitor: Any,
) -> tuple:
    """Inner message retry loop — handles MessageTimeoutError and SubscriptionAccessError."""
    msg_retries_remaining = state.options.message_retries

    while True:
        execution_coro = execute_prompt_streaming(
            effective_prompt,
            sdk_options,
            execution_state,
            message_callback=message_callback,
            message_timeout=state.options.message_timeout,
        )

        try:
            if timeout is not None:
                state.deadlines["prompt"] = _time.time() + timeout
                result_tuple = await asyncio.wait_for(execution_coro, timeout=timeout)
            else:
                result_tuple = await execution_coro
            return result_tuple

        except MessageTimeoutError:
            (
                sdk_options,
                effective_prompt,
                execution_state,
                msg_retries_remaining,
            ) = await _handle_message_timeout(
                state,
                sdk_options,
                sdk_options_cache,
                execution_state,
                original_prompt,
                prompt_index,
                msg_retries_remaining,
                with_output_format,
                get_monitor,
            )
            continue

        except SubscriptionAccessError as e:
            (
                sdk_options,
                effective_prompt,
                execution_state,
            ) = await _handle_subscription_error(
                state,
                e,
                sdk_options,
                sdk_options_cache,
                execution_state,
                original_prompt,
                prompt_index,
                with_output_format,
                get_monitor,
            )
            continue

        except AgentProcessError:
            state.last_failure_reason = "process_error"
            raise

        except (TimeoutError, asyncio.CancelledError):
            # Capture session_id for plain-resume on retry.
            sid = execution_state.get("session_id")
            if sid:
                state.session_id = sid
            state.last_failure_reason = "seq_prompt_timeout"
            raise


async def _handle_message_timeout(
    state: AgentState,
    sdk_options: Any,
    sdk_options_cache: Any,
    execution_state: dict,
    original_prompt: str,
    prompt_index: int,
    msg_retries_remaining: int,
    with_output_format: bool,
    get_monitor: Any,
) -> tuple:
    """Handle MessageTimeoutError — open new client with plain resume."""
    sid = execution_state.get("session_id")
    if sid:
        state.session_id = sid
    msg_retries_remaining -= 1

    if msg_retries_remaining <= 0:
        raise TimeoutError(
            f"Message timeout exhausted after {state.options.message_retries} retries"
        )

    # Wait for capacity if rate limited
    monitor = get_monitor()
    if monitor.is_rate_limited():
        wait_start = _time.monotonic()
        emit.status_public_warning(
            "Usage threshold exceeded — waiting for capacity before message retry..."
        )
        await monitor.async_wait_for_capacity()
        state.capacity_wait_total += _time.monotonic() - wait_start

    emit.status_public_warning(
        f"Message timeout (attempt {state.options.message_retries - msg_retries_remaining}/{state.options.message_retries}) "
        f"— forking from {state.session_id[:12] if state.session_id else '?'}..."
    )

    # Plain resume from partial session — preserves conversation but starts
    # a fresh per-instance cumulative counter. Each instance emits its own
    # agent_summary; NodeStats sums them via apply_leaf_summary.
    state.last_failure_reason = "message_timeout"
    effective_prompt = original_prompt
    if state.session_id:
        sdk_options = replace(
            sdk_options_cache,
            resume=state.session_id,
            fork_session=False,
            continue_conversation=True,
        )
        if with_output_format and state.options.output_format:
            sdk_options = replace(sdk_options, output_format=state.options.output_format)
        effective_prompt = build_continue_prompt(
            original_prompt,
            state.last_failure_reason,
            state.options,
        )

    new_execution_state = initialize_execution(
        state.options,
        effective_prompt,
        prompt_index,
    )
    execution_state.clear()
    execution_state.update(new_execution_state)

    return sdk_options, effective_prompt, execution_state, msg_retries_remaining


async def _handle_subscription_error(
    state: AgentState,
    error: Exception,
    sdk_options: Any,
    sdk_options_cache: Any,
    execution_state: dict,
    original_prompt: str,
    prompt_index: int,
    with_output_format: bool,
    get_monitor: Any,
) -> tuple:
    """Handle SubscriptionAccessError — wait indefinitely, does NOT consume retries."""
    sid = execution_state.get("session_id")
    if sid:
        state.session_id = sid
    state.last_failure_reason = "subscription_error"

    try:
        emit.status_public_warning(
            f"Auth/access error — waiting 60s then retrying (refresh token to unblock): {error}"
        )
    except Exception:
        logger.warning(f"Auth/access error — waiting 60s: {error}")

    try:
        _extend_worker_ttl(65.0)
    except Exception:
        pass

    wait_start = _time.monotonic()
    await asyncio.sleep(60)

    # Wait for usage monitor capacity
    try:
        monitor = get_monitor()
        if monitor.is_rate_limited():
            emit.status_public_warning("Usage threshold also exceeded — waiting for capacity...")
            _rate_wait_iters = 0
            _max_rate_wait_iters = 120  # ~110 min max wait
            while monitor.is_rate_limited():
                await asyncio.sleep(55)
                _rate_wait_iters += 1
                if _rate_wait_iters >= _max_rate_wait_iters:
                    emit.status_public_error(
                        f"Rate limit wait exceeded {_max_rate_wait_iters} iterations — giving up"
                    )
                    break
                if monitor.is_rate_limited():
                    emit.status_public_warning("Still waiting for capacity — rate limit active")
                    _extend_worker_ttl(65.0)
    except Exception:
        pass

    # Fetch fresh credentials. Deferred import: server_url cycles through
    # utils → agent_to_llm → agent_backend.claude.agent → here.
    try:
        from aii_lib.server_url import ability_service_url

        _ability_url = ability_service_url()
        if _ability_url:
            from aii_lib.llm_backend.claude_max.autologin import _fetch_credentials_remote

            _refreshed = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _fetch_credentials_remote(_ability_url),
            )
            if _refreshed:
                emit.status_private_info(
                    "Got fresh credentials from ability server after rate limit"
                )
    except Exception:
        pass

    state.capacity_wait_total += _time.monotonic() - wait_start

    # Re-initialize for retry — plain resume, no fork
    effective_prompt = original_prompt
    try:
        if state.session_id:
            sdk_options = replace(
                sdk_options_cache,
                resume=state.session_id,
                fork_session=False,
                continue_conversation=True,
            )
            if with_output_format and state.options.output_format:
                sdk_options = replace(sdk_options, output_format=state.options.output_format)
            effective_prompt = build_continue_prompt(
                original_prompt,
                state.last_failure_reason,
                state.options,
            )
        new_execution_state = initialize_execution(
            state.options,
            effective_prompt,
            prompt_index,
        )
        execution_state.clear()
        execution_state.update(new_execution_state)
    except Exception:
        pass

    return sdk_options, effective_prompt, execution_state


async def run_with_deadline(
    coro_factory: Any,
    base_timeout: float,
    state: AgentState,
    get_monitor: Any,
) -> Any:
    """Execute a coroutine with a deadline that excludes capacity wait time.

    Args:
        coro_factory: Callable that returns the coroutine to run.
        base_timeout: Base timeout in seconds.
        state: AgentState for reading capacity_wait_total.
        get_monitor: Callable returning the UsageMonitor.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + base_timeout
    task = asyncio.create_task(coro_factory())
    prev_wait = 0.0
    last_check = loop.time()

    try:
        while not task.done():
            now = loop.time()
            elapsed = now - last_check
            last_check = now

            if state.capacity_wait_total > prev_wait:
                deadline += state.capacity_wait_total - prev_wait
                prev_wait = state.capacity_wait_total

            monitor = get_monitor()
            if monitor.is_rate_limited():
                deadline += elapsed

            remaining = deadline - now
            if remaining <= 0:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise TimeoutError(
                    f"Agent deadline exceeded (base={base_timeout}s, "
                    f"capacity_wait={state.capacity_wait_total:.0f}s)"
                )

            await asyncio.wait({task}, timeout=min(remaining, 2.0))

        return task.result()
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        raise
