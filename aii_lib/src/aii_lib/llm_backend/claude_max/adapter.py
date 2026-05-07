"""``LlmBackendAdapter`` for claude_max.

claude_max routes the Claude Agent SDK CLI directly to ``api.anthropic.com``
using the user's OAuth/cookies. The CLI handles auth itself — no env
override needed.
"""

from __future__ import annotations

from typing import Any


class ClaudeMaxAdapter:
    """No-op adapter: the CLI uses its own OAuth flow."""

    def env_for_sdk(self, llm_backend_cfg: dict[str, Any]) -> dict[str, str]:
        """Return SDK CLI env overrides — none, since OAuth is intrinsic."""
        return {}


__all__ = ["ClaudeMaxAdapter"]
