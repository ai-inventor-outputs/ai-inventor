"""LlmBackendAdapter — agent_backend↔llm_backend boundary.

When the ``claude_agent_sdk`` agent_backend builds an SDK subprocess, it
needs to know what env the chosen llm_backend wants applied (e.g.
``ANTHROPIC_BASE_URL`` overrides). Each compatible llm_backend exposes
an adapter that translates its config block into env injection.

Supported pairings today (enforced at config load by
``PipelineConfig._validate_backend_pairings``):

    claude_agent_sdk + claude_max

The ``openrouter`` llm_backend is NOT compatible with any agent_backend
yet. Pairing attempts (in config or via direct ``AgentOptions``
construction) raise here — see ``get_adapter``. Steps that want
openrouter today must use the openrouter direct path
(``use_claude_agent: false`` + step-level model/effort, routed through
``OpenRouterClient.chat``), not the SDK.
"""

from __future__ import annotations

from typing import Any, Protocol


class LlmBackendAdapter(Protocol):
    """Tiny interface every SDK-compatible llm_backend exposes."""

    def env_for_sdk(self, llm_backend_cfg: dict[str, Any]) -> dict[str, str]:
        """Return env vars to inject into the SDK subprocess for this backend.

        Args:
            llm_backend_cfg: The ``llm_backend.<name>`` subtree from
                ``PipelineConfig.raw``.

        Returns:
            Dict of env var → value to ``setdefault`` into the SDK env.
            Empty dict means the SDK runs with its inherited env (e.g.
            claude_max relies on the CLI's own OAuth flow).
        """
        ...


def get_adapter(name: str) -> LlmBackendAdapter:
    """Look up the SDK adapter for an llm_backend by name.

    Raises ``NotImplementedError`` for llm_backends that have no
    SDK-compatible adapter (today: openrouter — no agent_backend
    compatible with it yet).
    """
    if name == "claude_max":
        from .claude_max.adapter import ClaudeMaxAdapter

        return ClaudeMaxAdapter()
    if name == "openrouter":
        # Translation layer (Anthropic SDK → LiteLLM proxy → OpenAI) was
        # found to be unreliable in practice — Anthropic's server-side
        # tools (``web_search_20250305`` etc.) don't translate to
        # OpenAI's function-call schema, every WebSearch returns 400, and
        # the agent fails before reaching its structured-output tool.
        # Keep this gate even though
        # ``PipelineConfig._validate_backend_pairings`` already rejects
        # the combo at config load — programmatic construction (tests,
        # ad-hoc scripts) bypasses that validator.
        raise NotImplementedError(
            "openrouter has no SDK-compatible agent_backend yet. "
            "Use the openrouter direct path instead: set "
            "``use_claude_agent: false`` on the step and configure "
            "``model`` / ``reasoning_effort`` directly. The Claude "
            "Agent SDK only works with claude_max today."
        )
    raise ValueError(f"Unknown llm_backend: {name!r}")


__all__ = ["LlmBackendAdapter", "get_adapter"]
