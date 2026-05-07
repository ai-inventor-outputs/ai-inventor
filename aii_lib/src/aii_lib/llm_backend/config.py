"""Configuration loader for LLM Backend (OpenRouter only).

Direct provider clients (OpenAI, Anthropic, Gemini) were removed; the only
remaining client is ``OpenRouterClient``, so this loader exposes a single
``get_openrouter_config()`` accessor over ``default_config.yaml``.
"""

from pathlib import Path

import yaml
from loguru import logger

_config = None


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches for
            ``default_config.yaml`` next to this module, then in the
            current working directory.

    Returns:
        Configuration dictionary.
    """
    global _config

    if config_path:
        path = Path(config_path)
    else:
        search_paths = [
            Path(__file__).parent / "default_config.yaml",
            Path.cwd() / "llm_backend" / "default_config.yaml",
            Path.cwd() / "default_config.yaml",
        ]
        path = next((p for p in search_paths if p.exists()), None)
        if path is None:
            logger.warning("No default_config.yaml found")
            return {}

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

    logger.info(f"Loaded config from: {path}")
    return _config


def get_config() -> dict:
    """Get cached config or load it."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_openrouter_config() -> dict:
    """Get OpenRouter-specific configuration."""
    return get_config().get("openrouter", {})


def get_claude_max_config() -> dict:
    """Get Claude Max plan bootstrap configuration."""
    return get_config().get("claude_max", {})


__all__ = [
    "load_config",
    "get_config",
    "get_openrouter_config",
    "get_claude_max_config",
]
