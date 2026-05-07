"""Timestamp — strict TZ-aware UTC datetime.

Hard requirements enforced by Pydantic validation:

  - Naive datetime → REJECTED (Pydantic ``AwareDatetime``).
  - Aware non-UTC datetime (e.g. ``+02:00``) → REJECTED by validator.
  - Wire format: ``dt.isoformat()`` always ends with ``+00:00`` —
    easy to spot a bad timestamp on disk.

Use ``Timestamp.now()`` as the canonical constructor at call sites.
Don't pass ``datetime.now()`` (no tz) anywhere — the validator will
reject it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    field_validator,
)

_UTC_OFFSET = UTC.utcoffset(None)


class Timestamp(BaseModel):
    """Frozen Pydantic wrapper around a UTC-only ``AwareDatetime``."""

    model_config = ConfigDict(frozen=True)

    dt: AwareDatetime
    """The underlying TZ-aware datetime. Must be UTC; the validator
    rejects any other offset."""

    @field_validator("dt")
    @classmethod
    def _must_be_utc(cls, v: datetime) -> datetime:
        if v.utcoffset() != _UTC_OFFSET:
            raise ValueError(
                f"Timestamp must be UTC; got tzinfo={v.tzinfo!r} with offset {v.utcoffset()!r}"
            )
        return v

    @classmethod
    def now(cls) -> Timestamp:
        """Canonical constructor — returns ``datetime.now(timezone.utc)``."""
        return cls(dt=datetime.now(UTC))

    @classmethod
    def parse(cls, s: str) -> Timestamp:
        """Parse an ISO-8601 string. Rejects naive + non-UTC."""
        return cls(dt=datetime.fromisoformat(s))

    @property
    def iso(self) -> str:
        """ISO-8601 string. Always ends with ``+00:00``."""
        return self.dt.isoformat()

    @property
    def unix(self) -> float:
        """Return Unix epoch timestamp."""
        return self.dt.timestamp()

    def __str__(self) -> str:
        return self.iso

    def __lt__(self, other: Timestamp) -> bool:
        return self.dt < other.dt

    def __le__(self, other: Timestamp) -> bool:
        return self.dt <= other.dt


__all__ = ["Timestamp"]
