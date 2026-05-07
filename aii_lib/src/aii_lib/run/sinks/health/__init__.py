"""aii_lib.run.sinks.health — heartbeat liveness file.

While the run is alive the :class:`HealthSink` appends ``ok <iso-ts>``
lines to ``<run_dir>/sinks/health/.heartbeat`` on a 60-second daemon
thread. On ``run_end`` it writes one final ``complete <ts>`` (success)
or ``paused <ts>`` (failed/stopped/interrupted) line and quiesces the
thread.

The server's runs-index reader infers per-run liveness from the file:
last ``ok`` within ``2 * HEARTBEAT_SECONDS`` → in_progress; older →
paused (process died); ``complete`` → done; ``paused`` → graceful pause.
"""

from __future__ import annotations

from .sink import HealthSink

# Canonical on-disk path relative to the run dir. Producers (this
# sink) and consumers (server's runs-status writer) all agree on one path.
HEARTBEAT_RELATIVE = "sinks/health/.heartbeat"


__all__ = ["HEARTBEAT_RELATIVE", "HealthSink"]
