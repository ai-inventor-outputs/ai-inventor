"""Artifact executors for invention loop.

Each executor handles one artifact type:
- research: OpenAI GPT-5-mini with web search
- experiment: Claude agent with code execution
- dataset: Claude agent with HuggingFace/web access
- evaluation: Claude agent with artifact access
- proof: DeepSeek Prover V2 via OpenRouter with Lean 4

Skills are auto-discovered from .claude/skills/ via setting_sources=["project"].

Uses module-level helpers from aii_lib (setup_workspace, build_options,
end_task_*) for common patterns.
"""

import re

# Import late to avoid circular import (Artifact, ArtifactType used in helpers below)
# These are imported at module level by functions that need them


def sanitize_title(title: str, max_length: int = 60) -> str:
    """Sanitize artifact title for use as folder name.

    Converts to lowercase, replaces spaces/special chars with underscores.

    Args:
        title: The artifact title to sanitize
        max_length: Maximum length of the result (default 60)

    Returns:
        Sanitized folder name like "my_artifact_title"
    """
    if not title:
        return "untitled"
    # Convert to lowercase
    result = title.lower()
    # Replace spaces, hyphens, and special chars with underscores
    result = re.sub(r"[^a-z0-9]+", "_", result)
    # Remove leading/trailing underscores
    result = result.strip("_")
    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)
    # Truncate
    if len(result) > max_length:
        result = result[:max_length].rstrip("_")
    return result or "untitled"


from .dataset import execute_dataset
from .evaluation import execute_evaluation
from .experiment import execute_experiment
from .proof import execute_proof
from .research import execute_research

__all__ = [
    # Executors
    "execute_research",
    "execute_experiment",
    "execute_dataset",
    "execute_evaluation",
    "execute_proof",
    # Helper functions
    "sanitize_title",
]
