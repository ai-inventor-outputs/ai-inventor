"""Filesystem prep for DBOS-native forks.

Forks inherit two kinds of disk state from their parent:

  1. The run dir tree under ``aii_data/users/<u>/runs/<run_id>/`` —
     phase iter dirs, .venvs, generated artifacts, the parent's clone
     log. The legacy fork mechanism rebuilt these from the on-the-fly
     Run aggregate's ``prep_fork`` calls; the DBOS-native server
     endpoint just ``shutil.copytree``s the whole tree once before
     calling ``DBOS.fork_workflow``.

  2. The Claude Code SDK's per-project session buckets at
     ``$CLAUDE_CONFIG_DIR/projects/<slug-of-cwd>/<session_id>.jsonl``.
     The slug is derived from the run_dir path, so a fork's run_dir
     means a fresh bucket — the SDK would exit 1 on FORK-resume with
     "No conversation found" because it can't find the session
     ``.jsonl`` files. We mirror the parent's bucket files into the
     fork's bucket so the SDK FORK-resume sees a populated bucket.

Both steps are server-side — they happen synchronously *before*
``DBOS.fork_workflow`` enqueues the fork, so the worker (cli
subprocess) sees a ready run_dir + populated SDK buckets when its
workflow body runs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def prepare_fork_filesystem(*, parent_run_dir: Path, fork_run_dir: Path) -> None:
    """Copy parent's run_dir + SDK session buckets into the fork's locations.

    Idempotent on the SDK side (overwrites any existing fork bucket
    files); errors out on the run_dir side if ``fork_run_dir`` already
    exists with content (we don't want to clobber).
    """
    if parent_run_dir == fork_run_dir:
        raise ValueError(
            f"prepare_fork_filesystem: parent and fork run_dir collide ({parent_run_dir})"
        )
    if not parent_run_dir.is_dir():
        raise FileNotFoundError(
            f"prepare_fork_filesystem: parent run_dir missing: {parent_run_dir}"
        )

    # ``copytree`` errors when ``fork_run_dir`` exists; the server
    # pre-generates a fresh fork_id (uuid4) so this should always be a
    # clean directory creation.
    shutil.copytree(parent_run_dir, fork_run_dir)
    _clone_session_buckets(parent_run_dir=parent_run_dir, fork_run_dir=fork_run_dir)


def _slugify_cwd(p: str) -> str:
    """Mirror Claude Code's project-bucket name: ``/`` and ``_`` → ``-``."""
    return p.replace("/", "-").replace("_", "-")


def _clone_session_buckets(*, parent_run_dir: Path, fork_run_dir: Path) -> None:
    """Copy SDK session jsonls from parent's project buckets to fork's.

    Claude Code keys session storage by ``$CLAUDE_CONFIG_DIR/projects/
    <slug-of-cwd>/<session_id>.jsonl``. Forking changes the run dir, so
    the slug — and thus the bucket — differs. Without this copy the
    fork's first FORK-resume exits 1 with "No conversation found".

    For each parent bucket whose path starts with
    ``slugify(parent_run_dir)`` we mirror its files into the
    corresponding fork bucket (same suffix, fork prefix). Cheap
    one-shot copy at fork-setup time.
    """
    config_dir_env = os.environ.get("CLAUDE_CONFIG_DIR")
    config_dir = Path(config_dir_env) if config_dir_env else Path.home() / ".claude"
    projects = config_dir / "projects"
    if not projects.is_dir():
        return

    parent_slug = _slugify_cwd(str(parent_run_dir.resolve()))
    fork_slug = _slugify_cwd(str(fork_run_dir.resolve()))
    prefix = parent_slug + "-"

    for bucket in projects.iterdir():
        if not bucket.is_dir() or not bucket.name.startswith(prefix):
            continue
        dest = projects / (fork_slug + bucket.name[len(parent_slug) :])
        dest.mkdir(parents=True, exist_ok=True)
        for f in bucket.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)


__all__ = ["prepare_fork_filesystem"]
