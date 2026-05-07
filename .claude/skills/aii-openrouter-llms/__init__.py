"""OpenRouter skill - LLM search and API calls.

CLI script: openrouter_llms.py
Abilities: aii_openrouter_llms__search, aii_openrouter_llms__call, aii_openrouter_llms__get_params
"""

from .scripts.openrouter_llms import call_direct, search_direct

__all__ = ["call_direct", "search_direct"]
