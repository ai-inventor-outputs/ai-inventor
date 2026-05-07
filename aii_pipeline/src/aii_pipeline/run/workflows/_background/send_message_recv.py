"""``send_message_recv_workflow`` — DBOS-native ``/send_message`` consumer.

The server's ``POST /api/runs/{run_id}/send_message`` endpoint calls
:func:`dbos.DBOS.send_async` with topic ``"user_message"``, addressed
to this workflow (id = ``f"{run_id}-recv"``). The recv loop unblocks,
looks up the target module's task ids in the parent's journal, and
fans the prompt out to the SDK sessions via
:func:`aii_lib.agent_backend.claude_agent_sdk.active_sessions.send_message`.

Why a separate workflow rather than an ``asyncio.create_task`` sibling
of ``run_pipeline_workflow``? Concurrent ``@DBOS.step`` calls (here:
``DBOS.recv_async``) interleaving with the parent's body would assign
function ids non-deterministically — DBOS replay then sees the steps
in a different order and raises ``DBOSUnexpectedStepError``. As its
own ``@DBOS.workflow`` the recv has its own function-id space; the
parent waits for nothing.

Lifecycle:

  * Parent ``run_pipeline_workflow`` calls
    :func:`dbos.DBOS.start_workflow_async` with the deterministic id
    :func:`recv_workflow_id`. ``start_workflow_async`` is itself
    cached as a step in the parent's journal so replay is idempotent.

  * On parent's ``finally`` block (normal completion, cancellation,
    crash), parent calls :func:`dbos.DBOS.cancel_workflow_async` with
    the same deterministic id. The recv's blocked ``DBOS.recv_async``
    raises :class:`DBOSWorkflowCancelledError`; we exit cleanly.
"""

from __future__ import annotations

from dbos import DBOS
from dbos._error import DBOSWorkflowCancelledError
from loguru import logger

# Topic name shared with the server endpoint and the recv loop below.
SEND_MESSAGE_TOPIC = "user_message"

# Per-recv timeout — the loop wakes this often even when no messages
# arrive, so cancellation propagates promptly.
_RECV_TIMEOUT_S = 10.0


def recv_workflow_id(parent_run_id: str) -> str:
    """Deterministic DBOS workflow id for the recv-loop sibling.

    Both the parent (which starts + cancels the recv) and the server
    (which sends to it) compute the id this way — keeping the naming
    rule in one place avoids drift.
    """
    return f"{parent_run_id}-recv"


def summary_workflow_id(parent_run_id: str) -> str:
    """Deterministic DBOS workflow id for the interim-summary sibling.

    Lives in this module beside :func:`recv_workflow_id` so callers
    have one import for both background-workflow id helpers. The
    events endpoint reads it to merge summary events into the run's
    timeline.
    """
    return f"{parent_run_id}-summary"


@DBOS.workflow()
async def send_message_recv_workflow(parent_run_id: str) -> None:
    """Receive ``{"module_id", "prompt"}`` payloads and dispatch into SDK sessions.

    Looks up the target module's task ids from the **parent**'s
    journal (not its own — the recv workflow's journal is empty
    except for its recv steps), and routes the prompt into each
    matching task's live SDK session via the in-process
    :mod:`aii_lib.agent_backend.claude_agent_sdk.active_sessions`
    registry. Tasks that aren't currently active are silently
    dropped — the warning surfaces only when no tasks at all
    matched the target module.
    """
    from aii_lib.agent_backend.claude_agent_sdk.active_sessions import (
        send_message as send_message_to_session,
    )
    from aii_lib.run.journal import find_task_session_ids_under_module

    while True:
        try:
            msg = await DBOS.recv_async(topic=SEND_MESSAGE_TOPIC, timeout_seconds=_RECV_TIMEOUT_S)
        except DBOSWorkflowCancelledError:
            return

        if msg is None:
            continue

        try:
            module_id = (msg.get("module_id") if isinstance(msg, dict) else "") or ""
            prompt = (msg.get("prompt") if isinstance(msg, dict) else "") or ""
            if not module_id or not prompt:
                logger.warning(f"send_message_recv_workflow: malformed payload {msg!r}")
                continue
            sessions = find_task_session_ids_under_module(parent_run_id, module_id)
            delivered = [
                task_id
                for task_id in sessions
                if send_message_to_session(task_id, prompt, prompt_source="human")
            ]
            if not delivered:
                logger.warning(
                    f"send_message_recv_workflow: module {module_id!r} has no "
                    "active SDK sessions; message dropped"
                )
        except DBOSWorkflowCancelledError:
            return
        except Exception:
            logger.opt(exception=True).warning("send_message_recv_workflow: dispatch failed")


__all__ = [
    "SEND_MESSAGE_TOPIC",
    "recv_workflow_id",
    "send_message_recv_workflow",
    "summary_workflow_id",
]
