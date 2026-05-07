"""Utilities module - general utilities for aii_lib."""

from aii_lib.prompts import LLMPromptModel

from .agent_to_llm import (
    ClaudeAgentToLLMStructOut,
    ClaudeAgentToLLMStructOutResult,
)
from .cache_cleanup import cleanup_run_caches
from .model_utils import get_model_short


def __getattr__(name: str) -> object:
    """Lazy-load HTTP ability-client helpers to avoid circular import.

    ``ability_client`` itself imports ``make_retry_log`` from
    :mod:`aii_lib.utils.retry`, which would otherwise re-enter
    ``aii_lib.utils.__init__`` mid-load.
    """
    if name in ("call_server", "server_available"):
        from aii_lib.abilities.ability_server.ability_client import (
            call_server,
            server_available,
        )

        return {"call_server": call_server, "server_available": server_available}[name]
    raise AttributeError(f"module 'aii_lib.utils' has no attribute {name!r}")


__all__: list[str] = [
    # Ability client (HTTP-based, lazy-loaded via __getattr__)
    "call_server",
    "server_available",
    # Cache cleanup
    "cleanup_run_caches",
    # Model utilities
    "get_model_short",
    # Prompt model
    "LLMPromptModel",
    # ClaudeAgentToLLMStructOut
    "ClaudeAgentToLLMStructOut",
    "ClaudeAgentToLLMStructOutResult",
]
