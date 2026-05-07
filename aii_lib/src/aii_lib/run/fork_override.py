"""Side-table that stages a fork's prompt override before ``DBOS.fork_workflow``.

Why a side table instead of ``DBOS.send`` / ``DBOS.recv``? The forked
workflow body must NOT block waiting for a message at start_step (it'd
deadlock fresh runs that legitimately have no inbox message). The
override is pushed once at fork creation time and read once at the
forked workflow's entry — a plain INSERT-once row keyed by the fork's
``workflow_uuid`` is the simplest race-free contract:

  * Server inserts the row inside its own DB transaction, **before**
    calling ``DBOS.fork_workflow``. Postgres' transactional ordering
    guarantees the row is visible to any subsequent reader.
  * The forked workflow body reads the row at entry. ``read_fork_override``
    is a plain SELECT — deterministic per ``workflow_id`` because the
    table is INSERT-once and never updated. Inside a workflow, calling a
    non-step deterministic read is fine (replays return the same value
    every time).

The schema lives in DBOS's *application* database (``aii_inventor``)
alongside any other AII-specific tables we add later. ``init_dbos``
ensures the table exists at process start.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    func,
    insert,
    select,
)

from aii_lib.dbos_app import init_dbos

_metadata = MetaData()


fork_overrides = Table(
    "aii_fork_overrides",
    _metadata,
    Column("fork_workflow_id", String, primary_key=True),
    Column("target_module_id", String, nullable=False),
    Column("prompt", String, nullable=False),
    Column("session_ids", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)


def ensure_table_exists() -> None:
    """Create ``aii_fork_overrides`` if it does not exist.

    Idempotent — ``create_all`` checks first. Called by ``init_dbos``
    after DBOS has constructed its app-DB engine.
    """
    dbos = init_dbos()
    _metadata.create_all(dbos._app_db.engine, checkfirst=True)


def set_fork_override(
    *,
    fork_workflow_id: str,
    target_module_id: str,
    prompt: str,
    session_ids: dict[str, str],
) -> None:
    """Stage the fork's prompt + session id override.

    Caller MUST commit this insert before invoking ``DBOS.fork_workflow``
    so the forked workflow body sees the row at entry. Raises if the row
    already exists for ``fork_workflow_id`` (forks are unique).
    """
    dbos = init_dbos()
    with dbos._app_db.engine.begin() as conn:
        conn.execute(
            insert(fork_overrides).values(
                fork_workflow_id=fork_workflow_id,
                target_module_id=target_module_id,
                prompt=prompt,
                session_ids=session_ids,
            )
        )


def read_fork_override(workflow_id: str) -> dict[str, Any] | None:
    """Look up the override row for ``workflow_id``.

    Returns ``{"target_module_id": str, "prompt": str, "session_ids":
    dict[str, str]}`` for forks; ``None`` for fresh runs (no row).
    Deterministic per ``workflow_id`` — safe to call from inside a
    DBOS workflow body without wrapping in ``@DBOS.step``.
    """
    dbos = init_dbos()
    with dbos._app_db.engine.connect() as conn:
        row = conn.execute(
            select(
                fork_overrides.c.target_module_id,
                fork_overrides.c.prompt,
                fork_overrides.c.session_ids,
            ).where(fork_overrides.c.fork_workflow_id == workflow_id)
        ).first()
    if row is None:
        return None
    return {
        "target_module_id": row[0],
        "prompt": row[1],
        "session_ids": dict(row[2]) if row[2] else {},
    }


__all__ = [
    "ensure_table_exists",
    "fork_overrides",
    "read_fork_override",
    "set_fork_override",
]
