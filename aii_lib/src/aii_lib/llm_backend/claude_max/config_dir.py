"""Resolve the Claude config dir (default ``~/.claude``, overridable).

Allows AII processes (server, pipeline, autologin) to use a separate Claude
config dir from the user's personal terminal. Set
``CLAUDE_CONFIG_DIR=<repo>/aii_data/.claude`` in the inner shell of every aii
subprocess; do *not* set it for personal terminal/ssh sessions, so personal
Claude state in ``~/.claude`` stays untouched.

Mirrors the lookup the Claude SDK does at
``claude_agent_sdk._internal.sessions:_get_claude_config_home_dir`` so the
SDK and our autologin/credentials code agree on which dir to read.
"""

from __future__ import annotations

import os
from pathlib import Path


def aii_claude_dir() -> Path:
    """Return the Claude config dir.

    If ``CLAUDE_CONFIG_DIR`` is set in the env, returns that path; otherwise
    ``~/.claude``.
    """
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude"
