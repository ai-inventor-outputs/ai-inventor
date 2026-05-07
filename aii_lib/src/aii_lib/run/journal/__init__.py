"""DBOS journal infrastructure — single source of truth for run events.

Three concerns, three modules:

* :mod:`.event_step` — the writer. ``journal_event_step`` is a
  ``@DBOS.step`` that records its input dict in the
  ``output`` column of ``dbos.operation_outputs`` (DBOS's default
  per-step storage). Called from :mod:`aii_lib.run.emit` (direct
  emit path) and from :meth:`aii_lib.run.run.Run._record` (the
  Run-bus → journal mirror inlined into the dispatch path).

* :mod:`.query` — pure SQL helpers. ``query_events`` paginates the
  journal by ``(started_at_epoch_ms, function_id)`` cursor;
  ``decode_output`` unpickles a row's ``output`` blob into a typed
  :class:`~aii_lib.run.messages.BaseMessage`;
  ``resolve_workflow_chain`` walks ``workflow_status.forked_from``
  for fork-stitching.

* :mod:`.tailer` — daemon-thread fan-out. :class:`JournalTailer`
  polls the journal via the helpers above and dispatches new
  messages to subscribed :class:`~aii_lib.run.sink.RunSink` instances
  in the host process. The FE polling endpoint
  (``GET /api/runs/{id}/events``) consumes the same query helpers
  out-of-process — both paths share one canonical view.
"""

from __future__ import annotations

from .event_step import journal_event_step
from .query import (
    decode_output,
    decode_output_raw,
    find_module_start_function_id,
    find_task_session_ids_under_module,
    query_events,
    resolve_workflow_chain,
)
from .tailer import JournalTailer

__all__ = [
    "JournalTailer",
    "decode_output",
    "decode_output_raw",
    "find_module_start_function_id",
    "find_task_session_ids_under_module",
    "journal_event_step",
    "query_events",
    "resolve_workflow_chain",
]
