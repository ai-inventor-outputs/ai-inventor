"""DBOS-wrapped file writes for use inside future workflow bodies.

Per ``DETERMINISM_CONTRACT.md``: file I/O inside a ``@DBOS.workflow``
body is forbidden (it's a side effect that fires every replay). This
module provides ``write_text_durable`` and ``write_bytes_durable`` —
``@DBOS.step``-decorated wrappers that journal the write in Postgres.

On first execution: the write happens normally; (path, content) are
recorded as the step's input. On cached replay (workflow rerun after
crash, or fork from past this step): the cached `None` return is used
and the file is NOT rewritten. That matches the real-world semantics
of "create this file once" — we don't want to clobber it with stale
data on resume.

Outside a workflow context (current production until Phase 2),
``@DBOS.step`` is a transparent passthrough — the write happens
exactly as before. So adopting this helper at call sites is a no-op
behavior-wise until DBOS workflow contexts arrive.
"""

from __future__ import annotations

from pathlib import Path

from dbos import DBOS


@DBOS.step()
def write_text_durable(
    path: str,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Write ``content`` to ``path`` as text.

    Args:
        path: Destination file path (str — Path objects are JSON-noisy
            so callers should ``str(p)`` at the boundary).
        content: Text to write.
        encoding: Text encoding; defaults to UTF-8.

    Inside a DBOS workflow, journals (path, content) and short-circuits
    on cached replay. Outside a workflow, behaves identically to
    ``Path(path).write_text(content, encoding=encoding)``.
    """
    Path(path).write_text(content, encoding=encoding)


@DBOS.step()
def write_bytes_durable(path: str, content: bytes) -> None:
    """Write ``content`` to ``path`` as bytes. See ``write_text_durable``."""
    Path(path).write_bytes(content)
