"""NodeID — free-form string identifier for every AIINode.

Plain ``str`` alias. Two factories:

* :func:`generate_short_id` — bare random suffix, kept for places that
  legitimately want an opaque id (a Run boot id, etc.).
* :func:`gen_path_id` — the canonical factory for tree nodes: returns
  ``f"{name}_{uuid5_hex12}"`` where the hex suffix is a UUIDv5 hash of
  the structural ``path``. Same path → same id across all runs (and
  across parent / fork pairs), so cross-run identity is just string
  equality. The ``name`` prefix is purely for grep-ability in logs;
  uniqueness lives in the deterministic suffix.

Two nodes of the same Python type with the same ``node_id`` are the
same node.
"""

from __future__ import annotations

import uuid

from aii_lib.utils.gen_id import gen_id as generate_short_id

# NodeID is just a string. Kept as a name for type-hint readability.
NodeID = str


# Stable namespace UUID for path-derived node ids. Derived from a fixed
# project string so it's reproducible without storing a magic UUID
# constant. Don't change the seed string once shipped — it would
# invalidate every existing node id.
_PATH_NS: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "aii-inventor.run.path")


def gen_path_id(name: str, path: str) -> str:
    """Build a deterministic node_id of shape ``<name>_<uuid5_hex12>``.

    ``name`` is the role descriptor (``"gen_plan"``, ``"iter3"``,
    ``"upd_hypo"``); the suffix is the first 12 hex characters of
    ``uuid5(_PATH_NS, path)``. Same ``path`` always yields the same
    id — that's the point. Empty / falsy ``name`` raises.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"gen_path_id: name must be a non-empty string (got {name!r})")
    if not isinstance(path, str):
        raise TypeError(f"gen_path_id: path must be a string (got {type(path).__name__})")
    suffix = uuid.uuid5(_PATH_NS, path).hex[:12]
    return f"{name}_{suffix}"


__all__ = ["NodeID", "gen_path_id", "generate_short_id"]
