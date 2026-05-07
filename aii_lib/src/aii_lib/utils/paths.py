"""Canonical path resolution for aii_data and its subdirs.

Single source of truth for AII_DATA_DIR resolution so aii_server,
aii_launcher, and aii_lib tooling stay in sync. Mirrors the logic in
``aii_server/config/settings.py`` (which still computes its own copy
because Django settings load before we may want aii_lib imports).

Resolution order:
  1. ``AII_DATA_DIR`` env var (set on RunPod to the network volume)
  2. ``data.root_dir`` in ``aii_config/server/server.yaml``
  3. ``<repo>/aii_data`` (local-dev fallback)
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repo root: the parent of aii_lib/, aii_server/, etc."""
    # aii_lib/src/aii_lib/utils/paths.py → repo root is 4 parents up.
    return Path(__file__).resolve().parents[4]


def aii_data_dir() -> Path:
    """Resolve AII_DATA_DIR — env > server.yaml > repo/aii_data."""
    env = os.environ.get("AII_DATA_DIR")
    if env:
        return Path(env)
    yaml_path = repo_root() / "aii_config" / "server" / "server.yaml"
    if yaml_path.exists():
        from aii_lib.utils.config_overrides import load_config_with_overrides

        cfg = load_config_with_overrides(yaml_path)
        root = (cfg.get("data") or {}).get("root_dir") or ""
        if root:
            return Path(root)
    return repo_root() / "aii_data"


def logs_dir(component: str) -> Path:
    """Per-component log directory under AII_DATA_DIR/logs.

    On RunPod this lives on the persistent network volume so logs
    survive pod restarts. Caller is responsible for mkdir.
    """
    return aii_data_dir() / "logs" / component
