"""aii_lib.run.sinks.clone — append-only event stream.

Each typed Run-bus event is one JSONL line in
``<run_dir>/sinks/clone/clone_log.jsonl``. The file IS the run: load is a
replay that dispatches every event onto a fresh Run.

Public surface::

    from aii_lib.run.sinks.clone import (
        CloneSink, load_run_clone,
    )

The sink is registered via ``run.subscribe_sink(...)``. Each
``BaseMessage`` serializes via ``model_dump_json`` and appends one
line — no debounce, no whole-Run snapshot, no projection.

Lossless w.r.t. the bus: any event the dispatcher applies lands here
verbatim. State mutations that bypass the bus (e.g. direct attribute
assignment in tests) aren't captured. The Run-domain side is
responsible for emitting events for every state change the clone
needs to preserve.

In-flight only: Run-class refactors that change event shapes still
break replay. The JSONL telemetry log covers archival; this stream
covers in-flight resume only.
"""

from __future__ import annotations

from .sequenced_sink import SequencedCloneSink
from .sink import CloneSink

# Canonical on-disk path (relative to the run dir) for the clone.
# Producers (this sink) and consumers (BFF /replay, run_sse_owner,
# resume-worker) all agree on one path.
CLONE_RELATIVE = "sinks/clone/clone_log.jsonl"

# Per-task-grouped sibling — used for human-readable archives and the
# interim-summary reader. Same line format as the unsequenced clone;
# the only difference is task interleaving order.
SEQUENCED_CLONE_RELATIVE = "sinks/clone/clone_log_sequenced.jsonl"

# Convenience alias for callers that don't have a sink instance.
# The reverse-read logic lives on :meth:`CloneSink.load`.
load_run_clone = CloneSink.load


__all__ = [
    "CLONE_RELATIVE",
    "SEQUENCED_CLONE_RELATIVE",
    "CloneSink",
    "SequencedCloneSink",
    "load_run_clone",
]
