"""ClaudeAgentToLLMStructOut - Use Claude Agent as an LLM with structured output.

Provides a clean interface for using the Claude Agent SDK
to produce validated JSON files matching Pydantic schemas.
"""

from .claude_agent_to_llm import (
    ClaudeAgentToLLMStructOut,
    ClaudeAgentToLLMStructOutResult,
)

__all__ = [
    "ClaudeAgentToLLMStructOut",
    "ClaudeAgentToLLMStructOutResult",
]
