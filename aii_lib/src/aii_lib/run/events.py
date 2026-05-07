"""Wire envelope for the cursor-based polling endpoint.

Phase-5 transport layer: a single ``GET /api/runs/{id}/events?since={cursor}``
returns a list of envelopes; FE polls every 500ms instead of subscribing to
SSE streams. The envelope wraps the existing :data:`aii_lib.run.messages.AnyMessage`
discriminated union with the small amount of journal metadata the FE needs
to project events onto the AIINode tree (workflow id, step ordinal, parent
linkage for fork stitching, monotonic timestamp).

Status: contract scaffold. The DBOS journal-stitching path that builds these
lands in Phase 5 once the BE has finished Phases 2-4 (Pipeline workflow,
Module child workflows, fork mechanism). This file exists now so the FE
hook + server route can be sketched against a stable shape — neither side
has to wait for the other.

Discriminator key on the wire is ``message.type`` (string), inherited from
:class:`aii_lib.run.messages.BaseMessage`. The envelope itself has no
discriminator — every entry is a :class:`RunEventEnvelope`.

Two-ID design — locked in:
    The envelope carries ``(workflow_id, function_id, ts_ms)`` for the
    journal coordinate (cursor pagination, parent-stitch on fork). The
    embedded :class:`AnyMessage` carries ``node_id`` / ``parent_id`` for
    the AIINode tree position (FE selection / hover / expansion).
    Conflating these — using only ``node_id`` for both jobs — breaks
    cursor monotonicity (node_ids reset across forks) and forces the FE
    to rewrite every component that keys off ``node_id`` today. Keeping
    them separate is the smallest possible Phase-5 surface change.

    BE follow-up (out of this file's scope): inside a ``@DBOS.workflow``
    body, ``aii_lib.run.node_id.generate_short_id`` MUST switch from
    ``random``-based to ``DBOS.uuid()`` so the same workflow on replay
    produces the same ``node_id`` and the cached step outputs line up.
    Callers don't change — just the generator under the hood. See
    ``DETERMINISM_CONTRACT.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .messages import BaseMessage


class RunEventEnvelope(BaseModel):
    """One journal entry, ready for the FE projection.

    The envelope is intentionally thin — anything richer than ``message``
    lives in the ``message`` payload itself (typed via the ``AnyMessage``
    discriminated union; see :mod:`aii_lib.run.messages`).
    """

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(
        ...,
        description=(
            "DBOS workflow id this event was recorded under. For pre-fork "
            "events stitched in from the parent run, this is the parent's "
            "workflow_id (so cursors stay monotonic across the boundary)."
        ),
    )
    parent_workflow_id: str | None = Field(
        default=None,
        description=(
            "Parent's workflow id when the run is a fork; ``None`` for "
            "top-level runs. Set on every envelope — convenience for the "
            "FE so it doesn't need a separate metadata fetch to know."
        ),
    )
    function_id: int = Field(
        ...,
        description=(
            "DBOS step ordinal within ``workflow_id`` — monotonic per "
            "workflow, starts at 1. Tie-breaker for ``ts_ms`` ordering "
            "and the second component of the cursor."
        ),
    )
    ts_ms: int = Field(
        ...,
        description=(
            "Step's ``started_at_epoch_ms`` from ``dbos.operation_outputs``. "
            "Primary sort key; cursor tail."
        ),
    )
    message: BaseMessage = Field(
        ...,
        description=(
            "Typed message — discriminated by ``message.type``. Concrete "
            "subclass picked via :data:`aii_lib.run.messages.AnyMessage`. "
            "The OpenAPI schema only sees ``BaseMessage`` here; the FE "
            "discriminates on ``type`` post-fetch the same way it does "
            "today for ``slim_message`` / ``trace`` / ``node_status``."
        ),
    )


class RunEventsResponse(BaseModel):
    """Response envelope for ``GET /api/runs/{id}/events``.

    Returns up to ``limit`` envelopes ordered by ``(ts_ms, function_id)``,
    plus the cursor the FE should pass on its next poll. Empty ``events``
    + same ``next_cursor`` is the idle case (poll again later).
    """

    model_config = ConfigDict(extra="forbid")

    events: list[RunEventEnvelope] = Field(
        default_factory=list,
        description="Envelopes since the request's ``since`` cursor (exclusive).",
    )
    next_cursor: str = Field(
        ...,
        description=(
            "Opaque cursor to pass on the next poll. Format is "
            "``<ts_ms>:<function_id>`` today; treat as opaque on the FE. "
            "Empty string ``''`` is the 'start from the beginning' value."
        ),
    )
    has_more: bool = Field(
        default=False,
        description=(
            "True if the BE truncated the response at ``limit`` and more "
            "events are available immediately. FE should poll again with "
            "no delay rather than waiting for the next interval."
        ),
    )


def parse_cursor(cursor: str) -> tuple[int, int]:
    """Parse the wire cursor format ``<ts_ms>:<function_id>``.

    Returns ``(0, 0)`` for the empty-string sentinel ('start from the
    beginning'). Malformed cursors raise :class:`ValueError` — the route
    converts that to a 400.
    """
    if not cursor:
        return 0, 0
    head, _, tail = cursor.partition(":")
    return int(head), int(tail)


def format_cursor(ts_ms: int, function_id: int) -> str:
    """Render the wire cursor format. Inverse of :func:`parse_cursor`."""
    return f"{ts_ms}:{function_id}"


__all__ = [
    "RunEventEnvelope",
    "RunEventsResponse",
    "parse_cursor",
    "format_cursor",
]
