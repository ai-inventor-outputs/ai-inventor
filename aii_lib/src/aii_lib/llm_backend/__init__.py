"""LLM Backend — OpenRouter-only client surface.

Provides a single async client for accessing 300+ models via OpenRouter.
Direct provider clients (OpenAI, Anthropic, Gemini) have been removed —
all non-claude_agent_sdk LLM calls route through ``OpenRouterClient``.

The agent_backend (``claude_agent_sdk``) only runs against
``llm_backend.claude_max`` (Anthropic direct via OAuth/cookies). The
``llm_backend.openrouter`` config block exists for direct
``OpenRouterClient.chat`` callsites (summarize, multi-LLM hypothesis,
free_viz image gen, audit/rank judges) — it has no SDK-compatible
agent_backend yet, and ``PipelineConfig._validate_backend_pairings``
rejects any per-step ``claude_agent:`` block with
``llm_backend == "openrouter"`` at config load.

Last updated: May 2026
"""

from .openrouter import ConversationStats, OpenRouterClient
from .tool_loop import ToolLoopResult, chat

__all__ = [
    "OpenRouterClient",
    "ConversationStats",
    "chat",
    "ToolLoopResult",
]
