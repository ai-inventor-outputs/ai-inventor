"""Registry of running ClaudeSDKClient sessions, keyed by task_id.

The pipeline's HTTP source (``aii_lib.run.sources.send_message``) needs to
inject user messages into a running task's session. The executor registers
itself here at the start of each ClaudeSDKClient lifetime; the source looks
up by ``task_id`` and calls :meth:`StreamingExecutor.inject(prompt)`, which
stores the next prompt and triggers ``client.interrupt()``. The executor's
loop catches the interrupt, drains the partial ResultMessage, then opens
a new client (plain ``resume=session_id``) with the injected prompt.

If the lookup misses (no live session for that task), the dashboard
should route the cold-path send-message through aii_server (which
owns boot/respawn) — the source returns 400.

Task ids are the canonical Run-domain ``Task.node_id`` strings;
they're globally unique without any prefix decoration, so the
registry keys with no normalization.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from claude_agent_sdk import ClaudeSDKClient

from .utils.execution.sdk_client import StreamingExecutor


@dataclass
class _Session:
    executor: StreamingExecutor
    client: ClaudeSDKClient
    loop: asyncio.AbstractEventLoop


_LOCK = threading.Lock()
_ACTIVE: dict[str, _Session] = {}


def register(
    task_id: str,
    *,
    executor: StreamingExecutor,
    client: ClaudeSDKClient,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Register a live SDK client + its owning executor under ``task_id``.

    Idempotent — last registration wins.
    """
    if not task_id:
        return
    with _LOCK:
        _ACTIVE[task_id] = _Session(executor=executor, client=client, loop=loop)


def deregister(task_id: str) -> None:
    """Remove the entry for ``task_id``."""
    if not task_id:
        return
    with _LOCK:
        _ACTIVE.pop(task_id, None)


def send_message(
    task_id: str,
    prompt: str,
    *,
    prompt_source: str = "human",
) -> bool:
    """Inject a user message into a running task's session.

    Thread-safe and non-blocking — stores the prompt in the executor's
    pending-injection slot and triggers ``client.interrupt()`` on the
    agent's loop. The executor's main loop sees the pending injection
    after the interrupt-induced ResultMessage drains, opens a new
    ``ClaudeSDKClient`` with ``resume=session_id`` + the new prompt as
    the first user message, and continues.

    Returns True on successful dispatch, False if no active session
    exists for ``task_id`` (the source raises 400 so the dashboard
    re-routes through aii_server's cold-path).
    """
    if not task_id:
        return False
    with _LOCK:
        sess = _ACTIVE.get(task_id)
    if sess is None:
        return False
    sess.executor.inject(prompt, prompt_source=prompt_source)
    return True


def is_active(task_id: str) -> bool:
    """True iff there's a live SDK session registered for ``task_id``."""
    if not task_id:
        return False
    with _LOCK:
        return task_id in _ACTIVE


def stop_all() -> int:
    """Trigger graceful stop on every active SDK session.

    For each registered executor: calls
    :meth:`StreamingExecutor.stop_gracefully` so the current turn
    drains, emits its ``agent_summary``, and the executor's loop
    returns instead of resuming. Used by the ``/stop`` source's soft
    path before its deferred SIGINT fallback.

    Returns the number of sessions that were signalled.
    """
    with _LOCK:
        sessions = list(_ACTIVE.values())
    for sess in sessions:
        try:
            sess.executor.stop_gracefully()
        except Exception:
            pass
    return len(sessions)


def wait_until_idle(timeout: float, *, poll_interval: float = 0.2) -> bool:
    """Block until the registry is empty or ``timeout`` seconds elapse.

    Originally part of the (now-deleted) ``/stop`` cooperative shutdown
    path: drain interrupted SDK sessions before final cleanup. Today's
    ``/stop`` is :func:`dbos.DBOS.cancel_workflow_async` + an
    infrastructure-level kill safety net, so this helper has no live
    callers — kept for symmetry / future re-introduction. Returns
    ``True`` if the registry drained inside the budget, ``False`` on
    timeout.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _LOCK:
            if not _ACTIVE:
                return True
        time.sleep(poll_interval)
    with _LOCK:
        return not _ACTIVE


def reset() -> None:
    """Test-only: clear every registration."""
    with _LOCK:
        _ACTIVE.clear()


__all__ = [
    "deregister",
    "is_active",
    "register",
    "reset",
    "send_message",
    "stop_all",
]
