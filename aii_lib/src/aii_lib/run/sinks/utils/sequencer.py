"""Task-level output sequencer for parallel-task display.

Wraps a forward callback and routes typed :class:`BaseMessage` events
through a single FIFO-by-task-sequence pipeline so consumers (console,
sequenced clone) see one task's output stream contiguously even when
multiple tasks emit concurrently.

Usage::

    seq = TaskSequencer(
        forward=lambda ev: write_line(ev),
        sequence_lookup=run.task_sequence,
    )
    seq.feed(typed_event)         # route through the sequencer
    seq.flush_pending()           # drain everything on sink close

The sequencer drives itself off the typed lifecycle messages —
``task_start`` registers a task (with its sequence number resolved via
``sequence_lookup(task_id)``), other events are routed to the current
task's stream or buffered, and ``task_end`` flushes the task's buffer
and promotes the next task by sequence.

The sequence number is the task's position in its parent module's
``children`` list — looked up on demand through ``sequence_lookup``,
which the owning sink wires to ``run.task_sequence`` at ``bind_run``
time. When the lookup returns ``None`` (task not yet in the tree, or
no callback set), the sequencer falls back to an auto-incrementing
counter that orders by ``task_start`` arrival.

Errors (``status_public_error``) bypass buffering — they always reach
the forward callback immediately so a crash isn't held behind a
buffered task.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from aii_lib.run.messages import BaseMessage


class TaskSequencer:
    """Sequence parallel-task output for deterministic display.

    A single :class:`TaskSequencer` lives inside a sink (one per
    sink). The sink calls :meth:`feed` for each event it receives;
    the sequencer either forwards immediately or queues, depending
    on the current task and lifecycle state. On sink ``close`` the
    sink calls :meth:`flush_pending` to drain residual buffers.

    Thread-safety: a single ``RLock`` guards every state field.
    Mutations and queue drains happen under the lock; the forward
    callback is invoked outside the lock to avoid re-entrancy if a
    sink calls back into ``feed`` from inside ``forward``.
    """

    def __init__(
        self,
        forward: Callable[[BaseMessage], None],
        sequence_lookup: Callable[[str], int | None] | None = None,
    ) -> None:
        self._forward = forward
        self._sequence_lookup = sequence_lookup
        self._task_buffers: dict[str, list[BaseMessage]] = defaultdict(list)
        self._active_tasks: set[str] = set()
        self._completed_tasks: set[str] = set()
        self._task_sequence: dict[str, int] = {}
        self._next_sequence: int = 0
        self._current_task_id: str | None = None
        self._current_has_emitted: bool = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def feed(self, event: BaseMessage) -> None:
        """Route one typed event through the sequencer.

        Lifecycle messages (``task_start`` / ``task_end``) drive
        registration and queue draining. Other messages route to the
        current task's stream (immediate forward) or its buffer.
        """
        msg_type = getattr(event, "type", "") or ""
        task_id = self._task_id_of(event)

        if msg_type == "task_start" and task_id:
            # task_start always forwards immediately so the operator can
            # see every parallel task announce its start in real time;
            # only the *body* events (agent_*/llm_*) get sequenced behind
            # the current task.
            seq = self._sequence_lookup(task_id) if self._sequence_lookup else None
            self._task_start(task_id, sequence=seq)
            self._forward(event)
            return

        if msg_type == "task_end" and task_id:
            self._task_end(event, task_id)
            return

        # Errors bypass the buffer — never park a crash behind another task.
        if msg_type == "status_public_error":
            self._forward(event)
            return

        self._emit_for(event, task_id)

    def flush_pending(self) -> None:
        """Drain everything still in the buffers (called on sink close)."""
        to_emit: list[BaseMessage] = []
        with self._lock:
            for task_id in list(self._task_buffers.keys()):
                to_emit.extend(self._task_buffers.pop(task_id))
            self._active_tasks.clear()
            self._completed_tasks.clear()
            self._task_sequence.clear()
            self._next_sequence = 0
            self._current_task_id = None
            self._current_has_emitted = False
        for ev in to_emit:
            self._forward(ev)

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    def _task_start(self, task_id: str, sequence: int | None) -> None:
        with self._lock:
            self._active_tasks.add(task_id)
            if sequence is not None:
                self._task_sequence[task_id] = sequence
            else:
                self._task_sequence[task_id] = self._next_sequence
                self._next_sequence += 1
            if not self._current_has_emitted:
                self._select_lowest_locked()

    def _task_end(self, event: BaseMessage, task_id: str) -> None:
        to_forward: list[BaseMessage] = []
        promote = False
        with self._lock:
            is_current = self._current_task_id == task_id
            if is_current:
                # Flush this task's queued messages, then the task_end itself.
                to_forward.extend(self._task_buffers.pop(task_id, []))
                to_forward.append(event)
                self._active_tasks.discard(task_id)
                self._task_sequence.pop(task_id, None)
                self._current_has_emitted = False
                promote = True
            else:
                # Buffer the task_end alongside the rest of this task's queue.
                self._task_buffers[task_id].append(event)
                self._active_tasks.discard(task_id)
                self._completed_tasks.add(task_id)

        for ev in to_forward:
            self._forward(ev)
        if promote:
            self._promote_next()

    def _emit_for(self, event: BaseMessage, task_id: str) -> None:
        forward_now = False
        with self._lock:
            if not task_id:
                forward_now = True
            elif task_id not in self._active_tasks:
                # Task hasn't been registered (or already ended) — emit out of band.
                forward_now = True
            else:
                if not self._current_has_emitted:
                    self._select_lowest_locked()
                if self._current_task_id == task_id:
                    self._current_has_emitted = True
                    forward_now = True
                else:
                    self._task_buffers[task_id].append(event)

        if forward_now:
            self._forward(event)

    # ------------------------------------------------------------------
    # Selection / promotion
    # ------------------------------------------------------------------

    def _select_lowest_locked(self) -> None:
        if not self._active_tasks:
            self._current_task_id = None
            return
        self._current_task_id = min(
            self._active_tasks,
            key=lambda t: self._task_sequence.get(t, float("inf")),
        )

    def _promote_next(self) -> None:
        to_forward: list[BaseMessage] = []
        with self._lock:
            # Flush completed-task buffers in sequence order, but only
            # those whose sequence is lower than every still-active task.
            while self._completed_tasks:
                next_completed = min(
                    self._completed_tasks,
                    key=lambda t: self._task_sequence.get(t, float("inf")),
                )
                min_seq = self._task_sequence.get(next_completed, float("inf"))
                if any(
                    self._task_sequence.get(t, float("inf")) < min_seq for t in self._active_tasks
                ):
                    break
                to_forward.extend(self._task_buffers.pop(next_completed, []))
                self._completed_tasks.discard(next_completed)
                self._task_sequence.pop(next_completed, None)

            if not self._active_tasks:
                self._current_task_id = None
                self._current_has_emitted = False
            else:
                self._current_task_id = min(
                    self._active_tasks,
                    key=lambda t: self._task_sequence.get(t, float("inf")),
                )
                pending = self._task_buffers.pop(self._current_task_id, [])
                if pending:
                    to_forward.extend(pending)
                    self._current_has_emitted = True
                else:
                    self._current_has_emitted = False

        for ev in to_forward:
            self._forward(ev)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _task_id_of(event: BaseMessage) -> str:
        return getattr(event, "task_id", "") or ""


__all__ = ["TaskSequencer"]
