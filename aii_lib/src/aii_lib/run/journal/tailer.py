"""Daemon-thread tailer that streams DBOS journal events to RunSinks.

Polls ``dbos.operation_outputs`` via :mod:`aii_lib.run.journal.query`
on a fixed cadence, decodes new rows into typed
:class:`~aii_lib.run.messages.BaseMessage` instances, and dispatches
each one to every subscribed :class:`~aii_lib.run.sink.RunSink`.

In-process counterpart to the FE's ``GET /api/runs/{id}/events``
polling endpoint — both share :mod:`aii_lib.run.journal.query` for
SQL + decode, so the journal is the single source of truth: any
event written via
:func:`~aii_lib.run.journal.event_step.journal_event_step` (whether
a direct ``emit.X`` call or mirrored inline from the Run bus by
:meth:`aii_lib.run.run.Run._record`) reaches every subscribed sink
without going through the Run bus.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

from aii_lib.run.journal import decode_output, query_events

if TYPE_CHECKING:
    from aii_lib.run.sink import RunSink


# Per-poll batch limit. The FE endpoint defaults to 500 with a 5000
# ceiling; for in-process fan-out 1000 keeps memory bounded while
# letting bursts drain in one cycle.
_POLL_LIMIT = 1000


class JournalTailer:
    """Polls the DBOS journal and dispatches new events to subscribed sinks.

    Lifecycle: ``__init__`` captures the target ``workflow_id`` and
    poll cadence; :meth:`subscribe` registers sinks; :meth:`start`
    spawns the daemon polling thread; :meth:`stop` signals + joins.

    One tailer corresponds to one workflow (= one run). Subscribed
    sinks are called sequentially from the polling thread — sinks
    are expected to commit each ``flush`` quickly. Sink exceptions
    are logged and dropped so one bad sink can't poison the rest of
    the fan-out.

    The cursor (``ts_ms``, ``function_id``) advances PER ROW, not
    per batch — even an undecodable row advances it so the loop
    can't spin forever on a malformed entry.
    """

    def __init__(self, workflow_id: str, *, poll_interval_s: float = 0.5):
        self._workflow_id = workflow_id
        self._poll_interval_s = poll_interval_s
        self._sinks: list[RunSink] = []
        self._cursor_ts: int = 0
        self._cursor_fid: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sinks_lock = threading.Lock()

    def subscribe(self, sink: RunSink) -> None:
        """Register ``sink`` to receive every event the tailer dispatches.

        Safe to call before or after :meth:`start`. Subscribing
        mid-run starts the sink at the current cursor — earlier
        events are not replayed.
        """
        with self._sinks_lock:
            self._sinks.append(sink)

    def start(self) -> None:
        """Spawn the daemon polling thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"JournalTailer-{self._workflow_id[:8]}",
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 2.0, drain: bool = True) -> None:
        """Signal the polling thread to exit and join with ``timeout_s``.

        With ``drain=True`` (the default), poll once synchronously
        before signalling — flushes events that landed between the
        last loop tick and the stop call. Idempotent.
        """
        if drain:
            self._poll_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def close(self) -> None:
        """Alias for ``stop(drain=True)`` — fits the ``RunSink.close`` contract.

        Lets ``aii_pipeline.pipeline.close_all_sinks`` shut the tailer
        down uniformly with the rest of the sinks dict it iterates over.
        """
        self.stop(drain=True)

    # ── Polling loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        # ``Event.wait`` returns True the moment ``stop()`` flips the
        # flag, so the thread exits within microseconds of the signal.
        while not self._stop.wait(self._poll_interval_s):
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            rows = query_events(
                [self._workflow_id],
                after_ts_ms=self._cursor_ts,
                after_function_id=self._cursor_fid,
                limit=_POLL_LIMIT,
            )
        except Exception:
            logger.opt(exception=True).debug(
                f"JournalTailer({self._workflow_id[:8]}): query_events failed"
            )
            return

        for _wf_id, fid, ts_ms, raw_output in rows:
            self._cursor_ts = ts_ms
            self._cursor_fid = fid
            msg = decode_output(raw_output)
            if msg is None:
                continue
            with self._sinks_lock:
                sinks_snapshot = list(self._sinks)
            for sink in sinks_snapshot:
                try:
                    sink.flush(msg)
                except Exception:
                    logger.opt(exception=True).warning(
                        f"JournalTailer: {type(sink).__name__}.flush raised"
                    )


__all__ = ["JournalTailer"]
