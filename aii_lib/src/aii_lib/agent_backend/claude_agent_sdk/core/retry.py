"""Retry helpers for Agent — callback logging, state reset, context building."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import AgentOptions


def make_retry_callback(
    max_retries: int,
    run_id: str,
    task_name: str,
) -> object:
    """Create a tenacity before_sleep callback that logs retries."""
    from aii_lib.utils.retry import make_retry_log

    return make_retry_log(max_retries=max_retries, label=f"claude agent: {task_name}")


def build_agent_retry_context(
    options: AgentOptions,
    last_error: str | None,
) -> str | None:
    """Build context string from the live Run's per-task event log.

    Uses ``options.run_id`` as the lookup key — that's the runtime
    task node_id ``find_task`` can resolve. ``options.agent_context``
    is a display name (``"gen_hypo_1"``, ``"data-0"`` etc.), not a
    lookup key — using it would cause ``find_task`` to miss and the
    retry preamble to silently be empty (the agent retried with the
    same prompt verbatim, instead of "previous attempt failed, pick
    up where it left off"). Gate on either field being set so the
    no-task-context call sites still skip.
    """
    if not (options.run_id or options.agent_context):
        return None

    from aii_lib.run import get_current_run

    run = get_current_run()
    if run is None:
        return None

    msgs = run.get_recent_message_text(
        task_id=options.run_id,
        n=options.retry_context_messages,
    )
    if not msgs:
        return None

    lines = "\n".join(f"  - {m}" for m in msgs)
    error_line = f"\nFailure reason: {last_error}" if last_error else ""
    return (
        f"PREVIOUS ATTEMPT FAILED{error_line}\n"
        f"Last actions before failure:\n{lines}\n\n"
        f"Use any partial work that exists from the previous attempt. "
        f"Do NOT start over — pick up where the previous attempt left off."
    )


def prepend_retry_context(prompts: list[str], context: str) -> list[str]:
    """Prepend retry context to the first prompt."""
    if not prompts:
        return prompts
    modified = list(prompts)
    modified[0] = f"{context}\n\n{modified[0]}"
    return modified
