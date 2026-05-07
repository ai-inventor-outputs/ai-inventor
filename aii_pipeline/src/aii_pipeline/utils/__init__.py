"""Utility functions for aii_pipeline."""

from collections.abc import Awaitable, Callable
from typing import Any

from aii_lib.prompts import LLMPromptModel

# Private import — for ad-hoc/non-model YAML formatting only.
# For model data, use model.to_prompt_yaml() instead.
from aii_lib.prompts.prompt_format import to_prompt_yaml, to_prompt_yaml_list

from aii_lib import get_model_short

from .pipeline_config import PipelineConfig, get_project_root, rel_path

# Default token-validity window for OAuth pre-refresh checks between steps.
# 1h — matches claude OAuth token pre-refresh window.
DEFAULT_MIN_TOKEN_VALIDITY_SECONDS = 3600


async def retry_until_result(
    fn: Callable[[], Awaitable[Any]],
    retries: int = 3,
) -> Any:
    """Call async `fn()` up to `retries` times, returning the first truthy result.

    Returns the last result (which will be falsy) if no attempt succeeds.
    """
    result = None
    for _ in range(retries):
        result = await fn()
        if result:
            return result
    return result


__all__ = [
    "PipelineConfig",
    "rel_path",
    "get_project_root",
    # Format helpers
    "get_model_short",
    "to_prompt_yaml",
    "to_prompt_yaml_list",
    "LLMPromptModel",
    # Retry helper
    "retry_until_result",
    # Constants
    "DEFAULT_MIN_TOKEN_VALIDITY_SECONDS",
]
