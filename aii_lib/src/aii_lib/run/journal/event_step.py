"""Journal each Run-bus event into DBOS's ``operation_outputs`` table.

User direction (Phase 6): everything goes into DBOS — no parallel
event-store table. Each :class:`~aii_lib.run.messages.BaseMessage` that
flows through ``Run._on`` is mirrored via :func:`journal_event_step`
(below) so the events endpoint can query DBOS's journal as the single
source of truth.

The step is **sync** (DBOS supports both) and **returns its input** so
the message ends up in the ``output`` column of ``operation_outputs``
(DBOS doesn't journal step inputs separately — see
``dbos/_schemas/system_database.py``). Cursor pagination on the FE
endpoint keys off ``(started_at_epoch_ms, function_id)`` from that
table.

Sync + zero-side-effect by design: callable from any code path
without coroutine plumbing. Outside a DBOS workflow context it's a
no-op (DBOS short-circuits or raises depending on config) — every
caller (:mod:`aii_lib.run.emit`, :meth:`aii_lib.run.run.Run._record`)
catches and drops so non-workflow Run usage (tests, manual scripts)
keeps working.
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS


@DBOS.step()
def journal_event_step(message: dict[str, Any]) -> dict[str, Any]:
    """Record one Run-bus event in DBOS's journal.

    Returns the input verbatim — the wrap is the journaling effect.
    """
    return message
