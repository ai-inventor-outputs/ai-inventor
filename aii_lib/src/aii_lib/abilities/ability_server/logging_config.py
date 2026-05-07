"""Config and retry logic for ability service."""

import functools
import random
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from aii_lib.abilities.aii_ability import DEFAULT_ABILITY_TIMEOUT

# =============================================================================
# Load config
# =============================================================================
from aii_lib.utils.config_overrides import load_config_with_overrides as _load_config

_CONFIG_FILE = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "aii_config"
    / "server"
    / "abilities.yaml"
)
_server_config: dict = _load_config(_CONFIG_FILE) if _CONFIG_FILE.exists() else {}

DEFAULT_TIMEOUT = float(_server_config.get("client", {}).get("timeout", DEFAULT_ABILITY_TIMEOUT))

# Crash log directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
_CRASH_LOG_DIR = _PROJECT_ROOT / "logs" / "ability_crashes"
_CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Endpoint log directory
_ENDPOINT_LOG_DIR = _PROJECT_ROOT / "logs" / "endpoints"
_ENDPOINT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Retry configuration
# =============================================================================

_retry_config = _server_config.get("retry", {})
RETRY_MAX_RETRIES = int(_retry_config.get("max_retries", 3))
RETRY_MIN_BACKOFF = float(_retry_config.get("min_backoff", 1.0))
RETRY_MAX_BACKOFF = float(_retry_config.get("max_backoff", 20.0))

_TRANSIENT_ERRORS = (
    "connection aborted",
    "connection reset",
    "connection refused",
    "remotedisconnected",
    "remote end closed",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "rate limit",
    "name resolution",
    "nameresolutionerror",
    "temporary failure in name resolution",
    "failed to resolve",
    "getaddrinfo failed",
    "nodename nor servname provided",
    "network is unreachable",
    "no route to host",
    "max retries exceeded",
)


def with_retry(
    func: Callable[[dict], dict] | None = None, *, max_retries: int | None = None
) -> object:
    """Decorator that adds retry logic with exponential backoff for transient errors."""
    retries = max_retries if max_retries is not None else RETRY_MAX_RETRIES

    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @functools.wraps(fn)
        def wrapper(req: dict) -> dict:
            for attempt in range(retries + 1):
                result = fn(req)
                if result.get("success", False):
                    return result
                error_msg = str(result.get("error", "")).lower()
                is_transient = any(err in error_msg for err in _TRANSIENT_ERRORS)
                if not is_transient or attempt >= retries:
                    return result
                backoff = min(RETRY_MIN_BACKOFF * (2**attempt), RETRY_MAX_BACKOFF)
                backoff = backoff * (0.8 + 0.4 * random.random())
                logger.warning(
                    f"Retrying ability {fn.__name__} ({attempt + 1}/{retries + 1}) "
                    f"in {backoff:.1f}s: {result.get('error')}"
                )
                time.sleep(backoff)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
