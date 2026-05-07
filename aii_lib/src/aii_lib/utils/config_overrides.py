"""Load YAML configs with optional ``<thing>.private.yaml`` overrides.

Convention: every tracked config file under ``aii_config/`` may have a
gitignored sibling with the suffix ``.private.yaml`` (matched by
``.gitignore``'s ``*.private.yaml`` glob). The sibling's keys are
deep-merged on top of the tracked file's keys at load time.

Use cases: per-machine secrets, personal account lists, RunPod template
ids, OTLP endpoints — anything that should never reach the public
repo via the ``sync-public`` flow.

Merge semantics:
  - dicts: recursive merge, overlay keys win
  - lists: overlay replaces (no append) — config lists almost always
    represent ordered alternatives where partial replacement is wrong
  - scalars: overlay replaces

Example:
    # aii_config/server/server.yaml             (tracked, public)
    server: {port: 10010, host: "0.0.0.0"}
    pod_lifecycle: {orchestrator: {template_id: ""}}

    # aii_config/server/server.private.yaml    (gitignored, override)
    pod_lifecycle: {orchestrator: {template_id: "tpl-XXXXXX"}}

    >>> load_config_with_overrides(Path("aii_config/server/server.yaml"))
    {'server': {'port': 10010, 'host': '0.0.0.0'},
     'pod_lifecycle': {'orchestrator': {'template_id': 'tpl-XXXXXX'}}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with ``overlay`` keys merged on top of ``base``.

    Dicts merge recursively; everything else (lists, scalars) is replaced
    wholesale by the overlay value. Inputs are not mutated.
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_with_overrides(public_path: Path) -> dict[str, Any]:
    """Load ``public_path`` and merge its ``.private.yaml`` sibling on top.

    A missing public file resolves to ``{}``; a missing private sibling
    is a no-op. Both files load via ``yaml.safe_load`` — no Python
    constructors, no remote includes.
    """
    public_path = Path(public_path)
    base: dict = {}
    if public_path.exists():
        base = yaml.safe_load(public_path.read_text()) or {}

    private_path = public_path.with_name(public_path.stem + ".private.yaml")
    if private_path.exists():
        overlay = yaml.safe_load(private_path.read_text()) or {}
        return deep_merge(base, overlay)
    return base


__all__ = ["deep_merge", "load_config_with_overrides"]
