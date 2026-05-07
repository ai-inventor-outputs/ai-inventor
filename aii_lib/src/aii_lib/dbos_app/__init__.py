"""DBOS bootstrap for the AI Inventor pipeline + server.

Single source for the DBOS instance + connection string. The rest of the
codebase imports ``init_dbos`` / ``shutdown_dbos`` at the process entry
point and the ``@DBOS.workflow`` / ``@DBOS.step`` decorators (re-exported
from the ``dbos`` package directly) at module scope.

Why this lives here, not in ``aii_pipeline`` or ``aii_server``: both
processes share the same Postgres cluster and the same DBOS app name,
so a single config + lifecycle helper avoids parallel definitions.

Boot sequence (every entry point):

    from aii_lib.dbos_app import init_dbos, shutdown_dbos

    def main() -> None:
        init_dbos()           # reads aii_config/dbos.yaml, launches DBOS
        try:
            run_pipeline(...)
        finally:
            shutdown_dbos()

Or for tests / one-shot scripts:

    from aii_lib.dbos_app import dbos_app

    with dbos_app():
        ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dbos import DBOS, DBOSConfig
from loguru import logger

from aii_lib.utils.config_overrides import load_config_with_overrides

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "init_dbos",
    "shutdown_dbos",
    "dbos_app",
    "load_dbos_config",
]


# Resolve to ``<repo>/aii_config/dbos.yaml`` regardless of cwd. From
# ``aii_lib/src/aii_lib/dbos_app/__init__.py`` the repo root is four
# parents up (``dbos_app`` → ``aii_lib`` → ``src`` → ``aii_lib`` →
# ``ai-inventor``).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DBOS_CONFIG_PATH = _REPO_ROOT / "aii_config" / "dbos.yaml"


def _build_database_url(
    *, user: str | None, password: str | None, host: str, port: int, db_name: str
) -> str:
    """Compose a libpq URL using a Unix socket directory or TCP host.

    psycopg accepts ``host=`` as a directory path for the socket form;
    SQLAlchemy carries it through. Password is optional (trust auth on
    project-local sockets is the default).

    ``host`` is treated as a Unix-socket directory if it starts with ``/``
    OR if it's a relative path that doesn't look like a hostname. Relative
    paths are resolved against the repo root so a public ``dbos.yaml``
    can ship a portable default (``aii_data/db/sock``) without baking in
    the contributor's absolute home directory.
    """
    user = user or os.environ["USER"]
    auth = f"{user}:{password}@" if password else f"{user}@"
    # Treat ``./foo`` or ``foo/bar`` as a socket directory relative to the
    # repo root (matches what scripts/local/pg.sh creates). A bare token
    # like ``localhost`` or ``db.example.com`` stays a TCP host.
    looks_like_path = host.startswith(("/", "./", "../")) or "/" in host
    if looks_like_path:
        socket_dir = host if host.startswith("/") else str((_REPO_ROOT / host).resolve())
        # ``port`` belongs in the socket-form URL too — postgres encodes it
        # in the socket filename (``.s.PGSQL.<port>``), so omitting it
        # would silently fall back to libpq's default 5432 even when the
        # config asked for a different port.
        return f"postgresql+psycopg://{auth}/{db_name}?host={socket_dir}&port={port}"
    return f"postgresql+psycopg://{auth}{host}:{port}/{db_name}"


def load_dbos_config() -> DBOSConfig:
    """Read ``aii_config/dbos.yaml`` (+ private overlay) and shape into DBOSConfig.

    Returns the dict expected by ``DBOS(config=...)``. Connection strings
    are composed from the ``postgres:`` block — see ``aii_config/dbos.yaml``
    for the schema.
    """
    raw: dict[str, Any] = load_config_with_overrides(_DBOS_CONFIG_PATH)
    dbos_section: dict[str, Any] = raw.get("dbos", {})
    pg: dict[str, Any] = dbos_section.get("postgres", {})

    app_name: str = dbos_section.get("app_name", "aii_inventor")
    app_db = pg.get("app_db_name", "aii_inventor")
    sys_db = pg.get("sys_db_name", f"{app_db}_dbos_sys")

    # Default host: project-local socket dir (matches what
    # ``scripts/local/pg.sh start`` creates). Relative paths are resolved
    # against the repo root in ``_build_database_url``.
    default_host = "aii_data/db/sock"
    db_url = _build_database_url(
        user=pg.get("user"),
        password=pg.get("password"),
        host=pg.get("host", default_host),
        port=int(pg.get("port", 5432)),
        db_name=app_db,
    )
    sys_db_url = _build_database_url(
        user=pg.get("user"),
        password=pg.get("password"),
        host=pg.get("host", default_host),
        port=int(pg.get("port", 5432)),
        db_name=sys_db,
    )

    config: DBOSConfig = {
        "name": app_name,
        "database_url": db_url,
        "system_database_url": sys_db_url,
        # Default serializer (base64-pickle) — handles our Pydantic
        # workflow inputs (PipelineWorkflowInput, GenStratWorkflowInput,
        # …) which DBOS's PortableJSON can't. Stack is Python-only so
        # cross-language readability isn't a concern; events endpoint
        # decodes via pickle to match.
    }
    return config


_dbos_instance: DBOS | None = None


def init_dbos() -> DBOS:
    """Construct the global DBOS instance and launch the runtime.

    Idempotent: calling twice returns the same instance without
    re-launching. The first caller wins; subsequent callers see the
    same configured runtime (use ``shutdown_dbos`` to reset).
    """
    global _dbos_instance
    if _dbos_instance is not None:
        return _dbos_instance
    config = load_dbos_config()
    logger.info(
        f"DBOS init: app={config['name']!r} "
        f"app_db={config['database_url']} sys_db={config.get('system_database_url')}"
    )
    _dbos_instance = DBOS(config=config)
    DBOS.launch()
    # AII-side tables that share the DBOS app-DB cluster — created here
    # rather than via a migration tool because they're tiny and tightly
    # coupled to the DBOS lifecycle.
    from aii_lib.run.fork_override import ensure_table_exists as _ensure_fork_overrides

    _ensure_fork_overrides()
    logger.success("DBOS launched")
    return _dbos_instance


def shutdown_dbos() -> None:
    """Tear down the global DBOS instance. Safe to call when not initialized."""
    global _dbos_instance
    if _dbos_instance is None:
        return
    DBOS.destroy()
    _dbos_instance = None
    logger.info("DBOS shut down")


@contextmanager
def dbos_app() -> Iterator[DBOS]:
    """Context manager for tests + one-shot scripts.

    Usage::

        with dbos_app() as dbos:
            run_my_workflow()
    """
    dbos = init_dbos()
    try:
        yield dbos
    finally:
        shutdown_dbos()
