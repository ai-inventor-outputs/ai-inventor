"""Append-only event streamer — one JSONL line per typed Run-bus event.

The file IS the run: every event the dispatcher applies to mutate
domain state lands here in arrival order. ``CloneSink.load(path)``
reconstructs the Run by replaying every line through the dispatcher.

Lossless w.r.t. the bus: whatever event the dispatcher routes through
``Run._on`` lands on disk verbatim. The sink does not project,
debounce, re-serialize, or snapshot — each typed event is one
``model_dump_json()`` + one ``write()``. Append is naturally atomic
for sub-PIPE_BUF (~4KB) writes; a crash leaves at most a torn last
line, which the loader drops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from aii_lib.run.run import Run
from aii_lib.run.sink import ReplayPolicy, RunSink

if TYPE_CHECKING:
    from aii_lib.run.messages import BaseMessage


class CloneSink(RunSink):
    """Run-bus subscriber: one JSONL line per typed event, append-only.

    Wired via ``run.subscribe_sink(sink)``. Each ``BaseMessage`` arrives,
    serializes as JSON via :meth:`map`, and appends to ``path``.

    Implements every :class:`RunSink` hook:

      - ``flush`` — channel: writes the JSONL line to the file
        (line-buffered, so each call commits to the OS immediately).
      - ``close`` — release the file handle.
      - ``map`` — pure event → JSONL line transform (inline).
      - ``load`` — classmethod that reads the JSONL back and replays
        through the dispatcher (or a pluggable builder).
    """

    # Default SKIP: legacy resume-replay's clone log on disk already
    # has the events; re-emitting would double-write. DBOS-native fork
    # bypasses Run-bus replay entirely.
    replay_policy = ReplayPolicy.SKIP

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # buffering=1 = line-buffered: each newline-terminated write is
        # flushed to the OS immediately so per-event durability is the
        # default and no separate buffer-drain method is needed.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")

    def flush(self, event: BaseMessage) -> None:
        """Write the event as JSONL."""
        self._fh.write(self.map(event))

    def map(self, event: BaseMessage) -> str:
        r"""Pure transform: event → JSONL line.

        Just ``event.model_dump_json() + "\\n"`` — too trivial to be
        worth a sibling module. The loader uses ``json.loads`` to invert
        and hands the dict to ``parse_message`` for typed re-hydration.
        """
        return event.model_dump_json() + "\n"

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        stop_after_module_end: str | None = None,
    ) -> Run:
        """Reverse-read: replay the JSONL stream into a fresh Run.

        Tolerates a torn last line (crash mid-write): unparseable JSON
        at EOF is dropped. An empty file (e.g. a fresh run before any
        event landed) returns ``Run(node_id=path.parent.name or path.stem)``.

        ``stop_after_module_end`` (legacy resume use): replay events up
        through and including the ``module_end`` event for the given
        module id, then stop. Everything afterwards is naturally absent
        from the resulting Run because their events were never
        dispatched. (DBOS-native forks no longer use this path —
        ``DBOS.fork_workflow`` copies the parent's ``operation_outputs``
        directly; the fork override side-table re-applies the new
        prompt at workflow entry.)
        """
        path = Path(path)
        messages = _read_jsonl_messages(path)

        # First event with a run_id seeds the Run. Typically the first
        # line is run_start carrying it; fall back to the run dir's
        # name for empty-file edge cases.
        # The clone log layout is ``<run_dir>/sinks/clone/clone_log.jsonl`` so
        # the run dir is the file's third-level parent — not just
        # ``path.parent`` (that's the ``clone/`` subdir).
        if path.parent.name == "clone" and path.parent.parent.name == "sinks":
            fallback_id = path.parent.parent.parent.name or path.stem
        else:
            fallback_id = path.parent.name or path.stem
        run_id = next(
            (m.get("run_id") for m in messages if m.get("run_id")),
            fallback_id,
        )

        if not messages:
            return Run(node_id=run_id)

        run = Run(node_id=run_id)
        _replay_messages(run, messages, stop_after_module_end=stop_after_module_end)
        return run

    def close(self) -> None:
        """Close the file handle."""
        if self._fh is None or self._fh.closed:
            return
        try:
            self._fh.close()
        except Exception:
            pass


def _read_jsonl_messages(path: Path) -> list[dict]:
    """Read the JSONL stream into a list of dicts.

    Tolerates a torn last line (crash mid-write): unparseable JSON at EOF
    is dropped. Empty file / missing file → empty list.
    """
    messages: list[dict] = []
    if not path.is_file():
        return messages
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                # Crash-tolerance: torn last line. Stop here;
                # everything before this point is consistent.
                break
    return messages


def _replay_messages(
    run: Run,
    messages: list[dict],
    *,
    stop_after_module_end: str | None = None,
) -> None:
    """Dispatch every dict in ``messages`` against ``run``.

    Pushes ``run`` as the ambient ``current_run`` so NodeStats roll-up
    helpers (``apply_leaf_summary`` /
    ``update_derived_stats_from_message``) hit it instead of silently
    no-oping. ``stop_after_module_end``: break after the module_end
    event for that module_id (inclusive — that event IS dispatched).
    """
    from aii_lib.run.context import get_current_run, set_current_run
    from aii_lib.run.dispatch import dispatch_event

    prev_run = get_current_run()
    set_current_run(run)
    try:
        for m in messages:
            dispatch_event(run, m)
            if (
                stop_after_module_end is not None
                and m.get("type") == "module_end"
                and m.get("module_id") == stop_after_module_end
            ):
                break
    finally:
        set_current_run(prev_run)


__all__ = ["CloneSink"]
