"""RunLifecycle — explicit state-machine transitions for Run.Status.

Centralises the cleanup ordering for the Run.Status transitions so each
caller goes through the same canonical sequence. Replaces the scattered
status writes in start_run, stop_run, _stop_pipeline, hide_run, the
completion handler, and the kill-cascade — each of which had slightly
different cleanup ordering.

Composes with Fork (aii_lib/fork.py) — a fork starts via
``parent.fork(target_task)`` which hooks both objects.

This module defines the abstract surface; concrete callbacks
(DB write, SSE emit, tracker invalidate, tmux kill, fork-stop) are
injected so the same lifecycle works in tests and prod without coupling
to Django ORM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class RunStatus(StrEnum):
    """Match aii_server.dashboard.models.Run.Status — kept in sync."""

    STARTING = "starting"  # transient: row created, pipeline not yet up
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class LifecycleHooks:
    """Callbacks injected by the caller.

    Each is invoked at the canonical point in the transition sequence — see
    RunLifecycle docstrings. Default no-ops keep tests trivial; the
    production wiring (in aii_server/dashboard/services/) supplies real
    callbacks.
    """

    db_write_status: Callable[[str, RunStatus], None] = field(default=lambda *_: None)
    sse_emit_status: Callable[[str, RunStatus], None] = field(default=lambda *_: None)
    tracker_invalidate: Callable[[str], None] = field(default=lambda _: None)
    kill_pipeline: Callable[[str], None] = field(default=lambda _: None)
    stop_forks: Callable[[str], None] = field(default=lambda _: None)


@dataclass
class RunLifecycle:
    """One run's lifecycle.

    Methods are the only legal Status transitions from outside callers.
    """

    run_id: str
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks)

    # ── transitions ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Transition to RUNNING state.

        Called when pipeline emits its first telemetry; flips skeleton
        placeholders to live state.
        """
        self.hooks.db_write_status(self.run_id, RunStatus.RUNNING)
        self.hooks.sse_emit_status(self.run_id, RunStatus.RUNNING)

    def complete(self) -> None:
        """RUNNING → COMPLETED. Pipeline reports it finished cleanly.

        Order: DB write (so any concurrent reads see the new state) →
        tracker invalidate (so subsequent /view re-reads from disk) →
        SSE emit (clients re-render).
        """
        self.hooks.db_write_status(self.run_id, RunStatus.COMPLETED)
        self.hooks.tracker_invalidate(self.run_id)
        self.hooks.sse_emit_status(self.run_id, RunStatus.COMPLETED)

    def stop(self, *, reason: str = "user_stop") -> None:
        """RUNNING → STOPPED. User-initiated or external stop.

        Order: kill the pipeline process tree first (so it can't write
        more telemetry mid-transition) → DB write → fork-stop cascade →
        tracker invalidate → SSE emit. Cascading to forks last so their
        own kill paths don't race against this run's tracker reads.
        """
        self.hooks.kill_pipeline(self.run_id)
        self.hooks.db_write_status(self.run_id, RunStatus.STOPPED)
        self.hooks.stop_forks(self.run_id)
        self.hooks.tracker_invalidate(self.run_id)
        self.hooks.sse_emit_status(self.run_id, RunStatus.STOPPED)

    def fail(self, *, exc_summary: str = "") -> None:
        """Transition to FAILED state.

        Pipeline crashed. Same ordering as stop() but doesn't kill (process
        is already dead) and uses FAILED status. ``exc_summary`` is for
        logs/telemetry, not stored on the row.
        """
        self.hooks.db_write_status(self.run_id, RunStatus.FAILED)
        self.hooks.stop_forks(self.run_id)
        self.hooks.tracker_invalidate(self.run_id)
        self.hooks.sse_emit_status(self.run_id, RunStatus.FAILED)


__all__ = ["LifecycleHooks", "RunLifecycle", "RunStatus"]
