"""``gen_id`` — single source of truth for opaque short ids.

Used for every entity that needs a stable, collision-resistant
identifier without leaking timing info or relying on a central
allocator:

  * :class:`aii_lib.run.run.Run` node ids (every Run / MdGroup /
    LoopIteration / Module / Task in the tree).
  * Server boot ids (one per ``aii_server`` process — anchors the
    per-server data dir at ``aii_data/servers/<id>/``).

12 chars from a 64-symbol alphabet → ~72 bits of entropy. That's enough
for a few billion ids in the same namespace before birthday-collision
risk crosses 1e-6 — fine for our scale (one user, one machine, runs in
the thousands).
"""

from __future__ import annotations

import secrets

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_ID_LEN = 12


def gen_id() -> str:
    """Return a fresh 12-char opaque id.

    Cryptographically random via :func:`secrets.choice` so callers can
    treat them as unguessable handles in addition to "unique enough".
    """
    return "".join(secrets.choice(_ALPHABET) for _ in range(_ID_LEN))


__all__ = ["gen_id"]
