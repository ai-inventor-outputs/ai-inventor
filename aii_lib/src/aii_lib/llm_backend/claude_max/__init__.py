"""Claude Max plan llm_backend — OAuth/cookies-based access to Anthropic.

The Claude Agent SDK CLI authenticates against ``api.anthropic.com`` using
the user's Claude Max subscription cookies + OAuth refresh — no API key
spend, usage counts against the plan's session/weekly quotas.

There's no Python client to construct (the CLI does the auth itself).
This package owns the runtime *lifecycle* helpers:

  - ``autologin/`` — multi-account autologin: ``AccountManager``,
    ``Account``, OAuth refresh, browser cookie pickup.
  - ``monitor`` — ``UsageMonitor`` background poller that blocks calls
    when session/weekly quotas approach the configured thresholds.
  - ``usage`` — parses Claude CLI usage output into ``ClaudeUsage``.
  - ``config_dir`` — ``aii_claude_dir()`` for the CLI's per-user storage.

Configuration: ``aii_config/pipeline/harness/llm_backend.yaml`` under
``claude_max`` (account_priority, usage_tracking, auth, telemetry).
Bootstrap defaults: ``aii_lib/llm_backend/default_config.yaml::claude_max``.
"""

from __future__ import annotations

from .autologin import Account, AccountManager
from .config_dir import aii_claude_dir
from .monitor import (
    UsageMonitor,
    async_require_capacity,
    get_monitor,
    require_capacity,
)
from .usage import ClaudeUsage, get_claude_usage


def get_default_model() -> str:
    """Bootstrap default model for the Claude Max llm_backend.

    Reads ``llm_backend/default_config.yaml::claude_max.default_model``. The
    runtime override path is per-step ``claude_agent.model`` in pipeline.yaml,
    which falls back through ``claude_max.defaults.model`` in
    ``aii_config/pipeline/harness/llm_backend.yaml`` — this getter is only
    used when neither is in scope.
    """
    from ..config import get_claude_max_config

    return get_claude_max_config().get("default_model", "claude-sonnet-4-6")


__all__ = [
    "Account",
    "AccountManager",
    "ClaudeUsage",
    "UsageMonitor",
    "aii_claude_dir",
    "async_require_capacity",
    "get_claude_usage",
    "get_default_model",
    "get_monitor",
    "require_capacity",
]
