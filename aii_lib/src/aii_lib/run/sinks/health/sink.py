r"""HealthSink — heartbeat liveness file at ``<run_dir>/sinks/health/.heartbeat``.

While the run is alive the sink appends ``ok <iso-ts>\\n`` every
``HEARTBEAT_SECONDS`` on a daemon thread. On ``run_end`` it appends one
final ``complete <iso-ts>\\n`` (status == ``completed``) or
``paused <iso-ts>\\n`` (interrupted / stopped / failed) and quiesces the
thread.

Append-only on purpose — the trail of timestamps lets a human or post-
mortem tool see exactly when the heartbeat stopped, which is impossible
with a truncate-and-rewrite scheme. A 1-min cadence over a 24h run is
~1500 lines / ~50 KB; cheap.

The server's runs-index reader infers per-run liveness from this file:

  - last line ``ok <ts>`` AND ``now - ts < STALE_SECONDS``  → in_progress
  - last line ``ok <ts>`` AND ``now - ts ≥ STALE_SECONDS``  → paused
                                                              (process died
                                                              before it
                                                              could flip
                                                              the file)
  - last line ``complete <ts>``                             → complete
  - last line ``paused <ts>``                               → paused
                                                              (graceful)

The reader's threshold should be at least ``2 * HEARTBEAT_SECONDS`` so
a single missed tick (GC pause, brief I/O stall) doesn't fake a crash.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aii_lib.run.sink import RunSink

if TYPE_CHECKING:
    from aii_lib.run.messages import BaseMessage


class HealthSink(RunSink):
    r"""Run-bus subscriber that maintains the ``sinks/health/.heartbeat`` file.

    Implements the :class:`RunSink` contract:

      - ``flush(event)`` — channel: writes the line returned by
        :meth:`map` to the heartbeat file (no-op for events that
        don't translate to a status line).
      - ``map(event) → str`` — pure transform: ``run_end`` events
        become ``"complete <ts>\\n"`` or ``"paused <ts>\\n"``;
        everything else maps to ``""`` (skipped by ``flush``).
      - ``close()`` — quiesces the heartbeat thread.
      - ``load()`` — not implemented; this sink is one-way.

    ``_stop`` is the only coordination primitive. It doubles as the
    "sink is done" sentinel:

      - ``flush(run_end)`` writes the terminal line and sets ``_stop``.
      - ``close()`` sets ``_stop`` (no terminal line; this is the
        standard "sink released" semantic — once closed, further
        ``flush`` calls are no-ops).
      - The heartbeat thread's ``_stop.wait(60)`` returns True the
        moment ``_stop`` is set, so it exits within microseconds.

    No ``Lock`` is needed: ``Run._notify`` walks sinks serially from
    one thread, so two ``flush`` calls can't race against each other.
    """

    def __init__(self, run_dir: Path, *, heartbeat_seconds: float = 5.0):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "sinks" / "health" / ".heartbeat"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_seconds = heartbeat_seconds
        self._stop = threading.Event()
        # Land an initial ``ok`` line BEFORE the first event so a server
        # scan in the gap between sink construction and ``run_start``
        # already sees a fresh heartbeat (no false-positive "paused").
        self._append_line(self._format_line("ok"))
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="HealthSink-heartbeat",
        )
        self._thread.start()
        from aii_lib.run import emit, get_current_run

        run = get_current_run()
        if run is not None:
            emit.status_private_info(
                f"HealthSink started (heartbeat={self.heartbeat_seconds}s)",
            )

    # ── RunSink contract ────────────────────────────────────────────

    def map(self, event: BaseMessage) -> str:
        """Project an event to a heartbeat line, or ``""`` to skip.

        Only ``run_end`` events translate to a line — ``complete`` for
        success, ``paused`` for any non-success status. Every other
        event type returns ``""`` so :meth:`flush` short-circuits.
        """
        if getattr(event, "type", None) != "run_end":
            return ""
        status = getattr(event, "status", "completed")
        token = "complete" if status == "completed" else "paused"
        return self._format_line(token)

    def flush(self, event: BaseMessage) -> None:
        """Channel: write the heartbeat line for ``event`` (if any).

        Short-circuits when the sink is done (terminal already written
        OR ``close()`` was called). The heartbeat thread is quiesced
        via ``_stop.set()`` so it stops appending ``ok`` lines after
        the terminal write.
        """
        line = self.map(event)
        if not line or self._stop.is_set():
            return
        self._stop.set()
        self._append_line(line)

    def close(self) -> None:
        """Mark the sink done and quiesce the heartbeat thread.

        Idempotent. Once closed, subsequent ``flush`` calls are no-ops
        — standard "sink released" semantic. The pipeline's normal
        shutdown calls ``flush(run_end)`` first (which writes the
        terminal line then closes); ``close()`` directly is for tests
        or forced shutdown.
        """
        self._stop.set()

    # ── Heartbeat thread ────────────────────────────────────────────

    def _loop(self) -> None:
        # ``_stop.wait`` returns True the moment ``_stop.set()`` is
        # called, so the heartbeat exits within microseconds of the
        # terminal write — no extra terminal-flag check needed.
        while not self._stop.wait(self.heartbeat_seconds):
            self._append_line(self._format_line("ok"))

    # ── File I/O ────────────────────────────────────────────────────

    @staticmethod
    def _format_line(status: str) -> str:
        ts = datetime.now(UTC).isoformat(timespec="seconds")
        return f"{status} {ts}\n"

    def _append_line(self, line: str) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            from loguru import logger

            logger.warning(f"HealthSink: append {self.path} failed: {e}")


__all__ = ["HealthSink"]
