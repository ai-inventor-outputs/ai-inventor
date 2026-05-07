"""Console output sink for the Run bus.

Writes colorized, formatted lines to stdout for each typed event.
Wraps an internal :class:`TaskSequencer` so parallel-task output
appears one task at a time (deterministic display) even when the
underlying tasks emit concurrently.
"""

from __future__ import annotations

from .sink import ConsoleRunSink

__all__ = ["ConsoleRunSink"]
