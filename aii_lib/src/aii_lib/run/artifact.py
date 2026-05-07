"""Artifact — a published deliverable produced by a pipeline run.

Emitted at the end of the gen_paper_repo step (paper PDF + GitHub repo)
on a ``status_published`` event. The to_app mapper derives
``AppRun.published.artifacts`` from the latest such event scoped to a
completed gen_paper_repo group; the frontend renders these as cards in
the run overview header. There is no ``Run.artifacts`` accumulator —
``Run.events`` is the canonical source of truth.

``kind`` is open-ended (string) so future steps can publish other kinds
(e.g. "video", "dataset") without a domain change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Artifact(BaseModel):
    """A published deliverable surfaced on the run overview."""

    model_config = ConfigDict(extra="allow")

    kind: str
    url: str
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Artifact"]
