"""RunSink — base contract for Run-bus subscribers.

A "sink" is one output channel for the Run bus. The contract has one
mandatory method (``flush`` — the channel that writes each event) and
three optional override hooks:

  - ``close`` — lifecycle housekeeping for sinks that hold handles
    (file writers, network sockets, asyncio queues, …). Each event
    write through ``flush`` is expected to commit immediately
    (line-buffered, unbuffered, or sync), so there's no separate
    drain-buffer method — sinks that need batching should debounce
    internally and commit on ``close``.
  - ``map`` — pure transform that turns an event into the wire format
    the channel will carry. Default is identity; override (or delegate
    to a sibling ``mapper.py``) when the wire shape differs from the
    raw event.
  - ``load`` — the reverse direction: read the channel's persisted
    output back into a fresh ``Run``. Default raises — most sinks are
    one-way. Override (or delegate to a sibling ``loader.py``) for
    bidirectional channels like ``clone``.

A Run holds its registered sinks in ``Run.sinks`` (a ``list[RunSink]``
populated by ``Run.subscribe``).

Concrete sinks live in ``aii_lib.run.sinks.<channel>.sink`` — each
output channel gets its own sub-package containing the Sink (the channel
implementation), an optional ``mapper.py`` (pure event → wire transform),
and an optional ``loader.py`` (the reverse — wire bytes → Run, when
round-trip is supported). The Sink class typically delegates ``map`` and
``load`` to those sibling modules so the encoding logic lives in one
place and can't drift between write + read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .messages import BaseMessage
    from .run import Run


class ReplayPolicy(StrEnum):
    """Per-sink behavior during replay-execute (``Run.playback_mode == "replay"``).

    Replay-execute re-runs ``execute()`` over a tree rebuilt from the
    clone log at boot. Most events would already be on disk in their
    original form, so re-emitting them through sinks would either
    double-write (CloneSink), double-export (OTel), or replay UX
    chrome the user already saw (Console).

    - ``SKIP`` — sink does NOT receive events while playback is replay.
                 Default + only value today; previous ``FIRE`` /
                 ``VERIFY`` modes were tied to the deleted Run-bus
                 fork-replay loop and have no live consumers.
    """

    SKIP = "skip"


class RunSink(ABC):
    """Base class for Run-bus subscribers.

    Subclasses implement ``flush(event)`` (mandatory) — that's the
    channel that writes each typed event onto the wire. Each call is
    expected to commit immediately; sinks that need batching should
    debounce internally and commit on ``close``. Optional overrides
    nudge concrete sinks toward a uniform shape:

      - ``close`` — lifecycle hook to release handles.
      - ``map`` — pure event → wire-format transform.
      - ``load`` — reverse read of the channel's persisted output.

    Run delivers events one at a time via ``flush``; sinks are expected
    to be best-effort and not raise (the bus catches exceptions to keep
    one bad sink from blocking the others, but cleaner sinks log +
    swallow internally).
    """

    replay_policy: ReplayPolicy = ReplayPolicy.SKIP
    """Behavior during ``Run.playback_mode == "replay"``. Always SKIP
    today — the on-disk projection files for legacy resume-replay are
    already complete, so re-emitting through any sink would double-write.
    See :class:`ReplayPolicy`."""

    @abstractmethod
    def flush(self, event: BaseMessage) -> None:
        """Channel: write one Run-bus event. The hot path.

        Must commit immediately — sinks that need to batch should
        debounce internally and commit on ``close``.
        """

    # ── Optional lifecycle hook ──────────────────────────────────────────

    def close(self) -> None:
        """Release any held resources (file handles, sockets, …).

        Default: no-op. The last chance to commit any internally batched
        state before the sink goes away.
        """
        return

    # ── Optional convention hooks (encouraged when applicable) ───────────

    def map(self, event: BaseMessage) -> Any:
        """Pure transform: event → wire format the channel will carry.

        Default: identity (returns the event itself). Override when the
        wire shape differs from the raw event — typically by delegating
        to a sibling ``mapper.py`` so the encoding logic is shared with
        ``load`` and can't drift between write + read.
        """
        return event

    @classmethod
    def load(cls, path: Path) -> Run:
        """Reverse: reconstruct a Run from the channel's persisted output.

        Default: raises ``NotImplementedError`` — most sinks are one-way
        (events flow out, nothing flows back). Override (or delegate to
        a sibling ``loader.py``) for bidirectional channels like
        ``clone`` where the on-disk format is rich enough to rebuild the
        Run from scratch.

        Implemented as a classmethod because the reverse-read needs no
        instance state — the file path is the only input — and callers
        often want to reconstruct without first constructing the sink.
        """
        raise NotImplementedError(
            f"{cls.__name__} is one-way; no loader is defined for this channel.",
        )


__all__ = ["ReplayPolicy", "RunSink"]
