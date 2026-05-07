"""Run mode detection — local vs RunPod.

Single source of truth for "where am I running?" — used by aii_server,
aii_pipeline, aii_lib.agents.* etc. Reads only env vars; no config
files, no DB, so it's safe to call before any framework boots.

Usage:
    from aii_lib.utils.run_mode import is_local, is_runpod, run_mode, pod_id
"""

from __future__ import annotations

import os
from typing import Literal

RunMode = Literal["local", "runpod"]


def pod_id() -> str:
    """RunPod's auto-injected pod identifier, or '' off-pod."""
    return os.environ.get("RUNPOD_POD_ID", "")


def is_runpod() -> bool:
    """True if running on a RunPod worker (RUNPOD_POD_ID is set)."""
    return bool(pod_id())


def is_local() -> bool:
    """True if running locally — i.e. not on a RunPod pod."""
    return not is_runpod()


def run_mode() -> RunMode:
    """Server-wide transport mode — 'runpod' on a pod, else 'local'."""
    return "runpod" if is_runpod() else "local"
