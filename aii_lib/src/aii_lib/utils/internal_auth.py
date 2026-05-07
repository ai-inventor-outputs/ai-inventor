"""Shared bearer-token auth for internal aii services.

The same key gates every internal HTTP surface the system runs:

  * ``/agent_abilities/*`` on aii_server
  * AppSink (run-bus read channel) and run sources (send_message, stop)
    on the orchestrator pod
  * any future cross-pod IPC

Resolution order (callers don't pick — the helpers here do):

  1. ``AII_INTERNAL_KEY`` env var
  2. ``<AII_DATA_DIR>/.internal_key`` file (written by aii_server at boot,
     readable from the shared RunPod network volume)
  3. fallback: ``<repo>/aii_data/.internal_key`` (local dev)

Usage::

    from aii_lib.utils.internal_auth import internal_headers
    httpx.get(url, headers=internal_headers())
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_internal_key() -> str | None:
    """Return the shared internal key, or ``None`` if unprovisioned."""
    env = os.environ.get("AII_INTERNAL_KEY", "").strip()
    if env:
        return env
    data_root = os.environ.get("AII_DATA_DIR", "").strip()
    candidates = []
    if data_root:
        candidates.append(Path(data_root) / ".internal_key")
    candidates.append(_PROJECT_ROOT / "aii_data" / ".internal_key")
    for p in candidates:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8").strip() or None
        except OSError:
            continue
    return None


def internal_headers() -> dict[str, str]:
    """Return ``{Authorization: Bearer <key>}`` or ``{}`` if no key resolvable.

    Empty headers fail closed at the server (401), which is the right
    behaviour — no silent unauth path.
    """
    key = _resolve_internal_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


__all__ = ["_resolve_internal_key", "internal_headers"]
