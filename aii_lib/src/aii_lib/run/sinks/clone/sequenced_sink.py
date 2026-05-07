"""Append-only event stream — task-grouped variant of CloneSink.

Same JSONL line format as :class:`CloneSink`, but events are routed
through a :class:`TaskSequencer` first so each task's stream lands
contiguously even when multiple tasks emit concurrently. The grouping
is positional (line order within each task block) — there is no
explicit ``seq`` field on the wire, the order IS the sequence. The
class is named "Sequenced" for historical reasons; "TaskGrouped" would
be more accurate but renaming would touch every consumer + the file
name on disk.

Used for human-readable archives (and the LLM interim-summary loop's
reader) where parallel task interleaving makes the unsorted log
unreadable. The unsequenced ``sinks/clone/clone_log.jsonl`` written by
:class:`CloneSink` remains the canonical replay source — sequenced
clones rearrange cross-task ordering and aren't guaranteed to replay
without dispatch surprises, so :meth:`load` is intentionally not
implemented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aii_lib.run.sink import ReplayPolicy, RunSink

from ..utils import TaskSequencer

if TYPE_CHECKING:
    from collections.abc import Callable

    from aii_lib.run.messages import BaseMessage


class SequencedCloneSink(RunSink):
    """Run-bus subscriber: per-task-grouped JSONL stream.

    Each ``BaseMessage`` enters via :meth:`flush`, gets routed
    through the internal :class:`TaskSequencer`, and the sequencer's
    forward callback writes one JSONL line per emitted event. Errors
    bypass buffering — the sequencer forwards them to the file
    immediately.

    Lifecycle:

      - ``flush`` — feeds the sequencer (no direct file write).
      - ``close`` — drains residual buffers, then closes the file.
      - ``map`` — same JSONL line as :class:`CloneSink` (inline).
      - ``load`` — inherited; raises ``NotImplementedError`` (the
        unsequenced clone is the canonical replay source).
    """

    # Default SKIP: legacy resume-replay's sequenced clone is already
    # complete on disk; re-emitting would double-write. DBOS-native
    # fork bypasses Run-bus replay entirely.
    replay_policy = ReplayPolicy.SKIP

    def __init__(
        self,
        path: Path,
        *,
        sequence_lookup: Callable[[str], int | None] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # buffering=1 = line-buffered.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._sequencer = TaskSequencer(
            forward=self._write,
            sequence_lookup=sequence_lookup,
        )

    # ------------------------------------------------------------------
    # RunSink hooks
    # ------------------------------------------------------------------

    def flush(self, event: BaseMessage) -> None:
        """Route the event through the task sequencer."""
        # Route through the sequencer; ``_write`` lands the JSONL line
        # for events that emerge from the sequencer's drain.
        self._sequencer.feed(event)

    def map(self, event: BaseMessage) -> str:
        """Serialize the event as JSONL."""
        return event.model_dump_json() + "\n"

    def close(self) -> None:
        """Drain the sequencer and close the file."""
        # Drain anything still parked in the sequencer (e.g. a task
        # whose ``task_end`` event never arrived because the run
        # crashed mid-task) so the on-disk archive isn't missing
        # buffered events.
        try:
            self._sequencer.flush_pending()
        except Exception:
            pass
        if self._fh is None or self._fh.closed:
            return
        try:
            self._fh.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal forward callback
    # ------------------------------------------------------------------

    def _write(self, event: BaseMessage) -> None:
        """Write an event to the file (best-effort)."""
        try:
            self._fh.write(self.map(event))
        except Exception:
            # Best-effort sink: a broken file handle shouldn't crash
            # the pipeline. The unsequenced clone is the durable source.
            pass


__all__ = ["SequencedCloneSink"]
