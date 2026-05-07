"""Shared SQL helpers for reading the DBOS journal.

The journal lives in ``dbos.operation_outputs``: every call to
:func:`aii_lib.run.journal.event_step.journal_event_step` writes one
row whose ``output`` column is the base64-pickled message dict (DBOS's
default step-output serializer). Two consumers share these helpers:

  - The FE polling endpoint
    (``aii_server.dashboard.api.run_events.get_run_events``) — pages
    over the journal with a ``(ts_ms, function_id)`` cursor and ships
    typed envelopes to the dashboard.

  - The in-process :class:`aii_lib.run.journal.tailer.JournalTailer`
    — same query, same decoder, but dispatches each new
    :class:`~aii_lib.run.messages.BaseMessage` to subscribed
    :class:`~aii_lib.run.sink.RunSink` instances on a daemon thread.

Keeping the SQL + decoder in one place means both consumers see the
exact same view of the journal — no encoding drift, no cursor-format
divergence.
"""

from __future__ import annotations

import base64
import pickle

from sqlalchemy import text

from aii_lib.dbos_app import init_dbos
from aii_lib.run.messages import BaseMessage

# The function name DBOS records for ``journal_event_step``. Inlined
# table names below (``dbos.operation_outputs``, ``dbos.workflow_status``)
# are DBOS's fixed system-schema identifiers — never user input — so the
# SQL strings below carry them as literals rather than f-string
# interpolations (avoids spurious S608 SQL-injection lint hits).


def resolve_workflow_chain(workflow_id: str, *, max_depth: int = 32) -> list[str]:
    """Walk ``workflow_status.forked_from`` from this run up to the root.

    Returns the chain in root-to-leaf order ``[root, ..., child, this]``
    so cursor pagination using ``workflow_uuid IN (...)`` advances
    monotonically across all ancestors. Bounded by ``max_depth`` to
    prevent runaway loops if the schema ever cycles.
    """
    dbos = init_dbos()
    chain: list[str] = [workflow_id]
    seen = {workflow_id}

    with dbos._sys_db.engine.connect() as conn:  # type: ignore[attr-defined]
        cur = workflow_id
        for _ in range(max_depth):
            row = conn.execute(
                text("SELECT forked_from FROM dbos.workflow_status WHERE workflow_uuid = :wid"),
                {"wid": cur},
            ).first()
            parent = row[0] if row else None
            if not parent or parent in seen:
                break
            seen.add(parent)
            chain.append(parent)
            cur = parent

    return list(reversed(chain))


def query_events(
    workflow_uuids: list[str],
    *,
    after_ts_ms: int,
    after_function_id: int,
    limit: int,
) -> list[tuple[str, int, int, str | None]]:
    """SELECT journal rows for the given workflow ids, after cursor, in order.

    Returns ``(workflow_uuid, function_id, started_at_epoch_ms, output)``
    tuples ordered by ``(started_at_epoch_ms, function_id)``. ``output``
    is the raw base64-pickle blob from DBOS — caller decodes via
    :func:`decode_output`.

    A single-element ``workflow_uuids`` list is the common case
    (per-run polling); multi-element lists support fork-chain
    stitching for the FE.
    """
    dbos = init_dbos()
    with dbos._sys_db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text(
                "SELECT workflow_uuid, function_id, started_at_epoch_ms, output "
                "FROM dbos.operation_outputs "
                "WHERE workflow_uuid = ANY(:chain) "
                "AND function_name = 'journal_event_step' "
                "AND (COALESCE(started_at_epoch_ms, 0), function_id) > (:ts, :fid) "
                "ORDER BY started_at_epoch_ms ASC NULLS FIRST, function_id ASC "
                "LIMIT :lim"
            ),
            {
                "chain": workflow_uuids,
                "ts": after_ts_ms,
                "fid": after_function_id,
                "lim": limit,
            },
        ).all()
    return [(r[0], int(r[1]), int(r[2] or 0), r[3]) for r in rows]


def decode_output_raw(raw: str | None) -> dict | None:
    """Decode a DBOS step output blob into the raw event dict.

    No Pydantic validation — use this when you need flat field access
    (e.g. ``payload.get("module_id")``) without forcing a typed
    :class:`BaseMessage` round-trip. Suitable for fork-time journal
    queries that read schema-stable fields like ``type`` /
    ``module_id`` / ``task_id`` / ``session_id``.
    """
    if not raw:
        return None
    try:
        decoded = pickle.loads(base64.b64decode(raw))
    except (TypeError, ValueError, pickle.UnpicklingError, EOFError):
        return None
    return decoded if isinstance(decoded, dict) else None


def decode_output(raw: str | None) -> BaseMessage | None:
    """Decode a DBOS step output blob into a typed BaseMessage.

    Wraps :func:`decode_output_raw` and validates the dict against the
    discriminated :class:`BaseMessage` union. Returns ``None`` for
    non-decodable / non-message rows so callers can silently skip rather
    than crash on a malformed entry.
    """
    decoded = decode_output_raw(raw)
    if decoded is None:
        return None
    try:
        return BaseMessage.model_validate(decoded)
    except Exception:
        return None


def find_module_start_function_id(workflow_id: str, target_module_id: str) -> int | None:
    """Return the ``function_id`` of the target module's ``module_start`` event.

    Walks the workflow's journal in order, decodes each ``journal_event_step``
    output, and returns the function_id of the FIRST event whose payload
    matches ``{"type": "module_start", "module_id": target_module_id}``.

    Used by the server-side fork endpoint to compute ``start_step`` for
    ``DBOS.fork_workflow`` — the forked workflow re-executes from this
    step onward, so the target module's body re-runs with the override
    prompt while every prior step is served from cache.

    Returns ``None`` if no matching ``module_start`` event exists in the
    workflow's journal (e.g. the user passed a stale or wrong module id).
    """
    dbos = init_dbos()
    with dbos._sys_db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text(
                "SELECT function_id, output FROM dbos.operation_outputs "
                "WHERE workflow_uuid = :wid "
                "AND function_name = 'journal_event_step' "
                "ORDER BY function_id ASC"
            ),
            {"wid": workflow_id},
        ).all()
    for fid, raw in rows:
        payload = decode_output_raw(raw)
        if payload is None:
            continue
        if payload.get("type") == "module_start" and payload.get("module_id") == target_module_id:
            return int(fid)
    return None


def find_task_session_ids_under_module(workflow_id: str, target_module_id: str) -> dict[str, str]:
    """Map ``task_id → session_id`` for every task that lived under the target.

    The fork override needs each target-module child task's recorded
    ``session_id`` so the agent backend can resume the prior SDK
    conversation as a FORK session in the new workflow. The lookup walks
    the workflow's ``task_start`` events (to find tasks whose
    ``attach_under_id`` is the target module) and pairs each task with
    the latest ``agent_end`` event's ``session_id``.

    Tasks without a captured ``session_id`` (e.g. the agent hadn't
    completed before the fork point) are omitted — the fork can't resume
    a session that never existed.
    """
    dbos = init_dbos()
    with dbos._sys_db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(
            text(
                "SELECT function_id, output FROM dbos.operation_outputs "
                "WHERE workflow_uuid = :wid "
                "AND function_name = 'journal_event_step' "
                "ORDER BY function_id ASC"
            ),
            {"wid": workflow_id},
        ).all()

    tasks_under_target: set[str] = set()
    session_id_by_task: dict[str, str] = {}
    for _fid, raw in rows:
        payload = decode_output_raw(raw)
        if payload is None:
            continue
        ev_type = payload.get("type")
        if ev_type == "task_start" and payload.get("attach_under_id") == target_module_id:
            tasks_under_target.add(payload["task_id"])
        elif ev_type == "agent_end":
            tid = payload.get("task_id")
            sid = payload.get("session_id")
            if tid and sid:
                # Last write wins — agent retries can emit multiple
                # ``agent_end`` events; the latest captured session is the
                # one a fork would resume.
                session_id_by_task[tid] = sid

    return {tid: sid for tid, sid in session_id_by_task.items() if tid in tasks_under_target}


__all__ = [
    "decode_output",
    "decode_output_raw",
    "find_module_start_function_id",
    "find_task_session_ids_under_module",
    "query_events",
    "resolve_workflow_chain",
]
