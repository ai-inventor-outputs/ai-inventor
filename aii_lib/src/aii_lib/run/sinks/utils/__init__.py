"""Shared sink helpers — pieces multiple Run sinks reuse.

Currently:
  - :class:`TaskSequencer` — buffers messages from non-current tasks
    and flushes them in sequence order so console output / sequenced
    JSONL stays deterministic when tasks run in parallel.
"""

from __future__ import annotations

from .sequencer import TaskSequencer

__all__ = ["TaskSequencer"]
