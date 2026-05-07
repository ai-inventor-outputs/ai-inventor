"""aii_lib.run.sinks — generic Run-bus subscribers.

Each subpackage owns one channel:

  - ``clone``: full-Run event stream. :class:`CloneSink` writes one
    JSONL line per event in arrival order to ``sinks/clone/clone_log.jsonl``;
    :class:`SequencedCloneSink` writes a per-task-grouped sibling at
    ``sinks/clone/clone_log_sequenced.jsonl``. :func:`load_run_clone` is the
    inverse of the unsequenced clone.

  - ``console``: colorized stderr. :class:`ConsoleRunSink` wraps a
    :class:`TaskSequencer` so parallel-task output appears one task at
    a time even when underlying tasks emit concurrently.

  - ``health``: heartbeat liveness file at
    ``run_dir/sinks/health/.heartbeat``. The :class:`HealthSink` writes
    ``ok <ts>`` lines on a background thread and a final
    ``complete``/``paused`` line on ``run_end``; the server's
    runs-index reader uses it to flag dead runs.

  - ``otel``: OpenTelemetry traces + metrics. :class:`OTelRunSink`
    translates lifecycle messages into spans and summary messages into
    counter/histogram updates.

  - ``title``: writes ``sinks/title/.title`` when an LLM-generated
    title lands.

  - ``utils``: shared helpers — currently :class:`TaskSequencer`,
    used by the console sink and the sequenced clone variant.
"""

from __future__ import annotations

# Subpackages are imported on demand via fully-qualified paths
# (``from aii_lib.run.sinks.clone import CloneSink``). Avoiding eager
# imports here breaks the circular path
# ``run.py → sinks/__init__.py → clone/sink.py → Run``.
__all__: list[str] = []
