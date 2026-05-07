"""
Ability Client - HTTP client for calling the Ability Service.

Provides a simple interface to call ability endpoints via HTTP.
Expects aii_server (Django) to be running; the caller is responsible
for starting it.

Usage:
    from aii_lib.abilities.ability_server import call_server, server_available

    # Check if service is available
    if server_available():
        result = call_server("aii_hf_search_datasets", {"query": "ML", "limit": 5})
"""

from pathlib import Path
from typing import Any, NoReturn

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aii_lib.abilities.aii_ability import DEFAULT_ABILITY_TIMEOUT
from aii_lib.server_url import (
    SERVER_HOST,
    SERVER_PORT,
    ability_service_url,
)

# Load abilities.yaml (retry/cleanup/worker_memory live here; host/port in server.yaml)
# Use the deep-merge loader so an ``abilities.private.yaml`` sibling layered
# on top is honoured — keeps machine-specific tweaks out of git.
from aii_lib.utils.config_overrides import load_config_with_overrides as _load_config
from aii_lib.utils.retry import make_retry_log

_CONFIG_FILE = Path(__file__).resolve().parents[5] / "aii_config" / "server" / "abilities.yaml"
_server_config: dict = _load_config(_CONFIG_FILE) if _CONFIG_FILE.exists() else {}

# Host/port mirrored from server.yaml; ability client timeout from abilities.yaml.
DEFAULT_HOST = SERVER_HOST
DEFAULT_PORT = SERVER_PORT
DEFAULT_TIMEOUT = float(_server_config.get("client", {}).get("timeout", DEFAULT_ABILITY_TIMEOUT))

# Bearer auth for /agent_abilities/* — shared with run sinks/sources.
# Resolved via aii_lib.utils.internal_auth (env var or shared-volume file
# written by aii_server at boot). See that module's docstring for details.
from aii_lib.utils.internal_auth import internal_headers


def _ensure_server() -> None:
    """No-op — the ability server is now part of Django (aii_server).

    Previously auto-started a standalone FastAPI server in tmux.
    Now the caller is responsible for ensuring aii_server is running.
    """


# HTTP status codes that are transient (server bootstrapping, proxy hiccup).
# 524 = Cloudflare origin timeout (RunPod API overloaded).
_TRANSIENT_STATUS_CODES = frozenset({404, 429, 502, 503, 504, 524})

# Permanent client errors are anything not in ``_TRANSIENT_STATUS_CODES``
# (4xx other than 404/429, 5xx other than 502/503/504/524). Listed implicitly
# to avoid drift; the dispatcher in ``_classify_and_raise`` raises
# RuntimeError for them and the caller decides whether to surface or swallow.


class AbilityTransientError(Exception):
    """Transient ability server error — safe to retry.

    Distinct from RuntimeError so higher-level retries (e.g. worker_pod
    tenacity) can exclude transport retries and avoid double-retry stacking.
    """


def _classify_and_raise(endpoint: str, exc: Exception) -> NoReturn:
    """Re-raise httpx errors as AbilityTransientError or RuntimeError."""
    import httpx

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        raise AbilityTransientError(f"Ability server unavailable for '{endpoint}': {exc}") from exc
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = exc.response.text[:200]
        if status in _TRANSIENT_STATUS_CODES:
            raise AbilityTransientError(
                f"Ability server transient error for '{endpoint}': {status} {body}"
            ) from exc
        # TRY004 suppressed below: HTTP non-transient is a runtime issue from
        # the server's response code, not a wrong-type input — RuntimeError
        # is the right semantic.
        raise RuntimeError(  # noqa: TRY004
            f"Ability server error for '{endpoint}': {status} {body}"
        ) from exc
    raise TypeError(f"Unexpected error calling ability server '{endpoint}': {exc}") from exc


def get_ability_service_url() -> str:
    """Get the ability service URL.

    Checks ``AII_SERVER_URL`` env var first (RunPod HTTPS proxy),
    then falls back to host/port from aii_config/server/abilities.yaml.
    """
    url = ability_service_url()
    if url:
        return url
    return f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


def server_available(timeout: float = 2.0) -> bool:
    """
    Check if the ability service is available.

    Args:
        timeout: Timeout for health check request

    Returns:
        True if service is available, False otherwise
    """
    import httpx

    try:
        url = get_ability_service_url()
        response = httpx.get(
            f"{url}/agent_abilities/health",
            headers=internal_headers(),
            timeout=timeout,
        )
        return response.status_code == 200
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.TimeoutException,
    ):
        return False


def call_server(
    endpoint: str,
    request: dict[str, Any],
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """
    Call an ability endpoint via HTTP.

    Auto-starts the ability server in a tmux session if running locally
    and the server is not up. Retries on transient errors (connection drops,
    404 during bootstrap, 502/503/504 proxy hiccups, 429 rate limits).

    Args:
        endpoint: Endpoint name (e.g., 'hf_search', 'web_fetch')
        request: Request data dict
        timeout: Request timeout in seconds

    Returns:
        Response dict from the endpoint, or None if unavailable

    Raises:
        AbilityTransientError: On transient errors after all retries exhausted.
        RuntimeError: On permanent HTTP errors (400, 401, 403, etc.)
    """
    _ensure_server()

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    @retry(
        retry=retry_if_exception_type(AbilityTransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        before_sleep=make_retry_log(label="ability call"),
        reraise=True,
    )
    def _call_with_retry() -> dict[str, Any]:
        import httpx

        try:
            url = get_ability_service_url()
            response = httpx.post(
                f"{url}/agent_abilities/{endpoint}",
                json=request,
                headers=internal_headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            _classify_and_raise(endpoint, e)

    return _call_with_retry()


async def async_call_server(
    endpoint: str,
    request: dict[str, Any],
    timeout: float | None = None,
) -> dict[str, Any]:
    """
    Async version of call_server using httpx.AsyncClient.

    For use from async pipeline code (WorkerPod, exec_mode_router, etc.)
    to avoid blocking the event loop. Retries on transient errors (connection
    drops, 404 during bootstrap, 502/503/504 proxy hiccups, 429 rate limits).

    Args:
        endpoint: Endpoint name (e.g., 'aii_runpod__gen_pod')
        request: Request data dict
        timeout: Request timeout in seconds

    Returns:
        Response dict from the endpoint.

    Raises:
        AbilityTransientError: On transient errors after all retries exhausted.
        RuntimeError: On permanent HTTP errors (400, 401, 403, etc.)
    """
    import httpx
    from tenacity import AsyncRetrying

    _ensure_server()

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(AbilityTransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        before_sleep=make_retry_log(label="ability call"),
        reraise=True,
    ):
        with attempt:
            try:
                url = get_ability_service_url()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{url}/agent_abilities/{endpoint}",
                        json=request,
                        headers=internal_headers(),
                        timeout=timeout,
                    )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                _classify_and_raise(endpoint, e)
    # Unreachable: ``reraise=True`` raises on exhaustion. ruff can't infer.
    raise RuntimeError("AsyncRetrying loop exited without raising or returning")


# Aliases for convenience
call_ability = call_server
ability_available = server_available
async_call_ability = async_call_server


__all__ = [
    "AbilityTransientError",
    "ability_available",
    "async_call_ability",
    "async_call_server",
    "call_ability",
    "call_server",
    "get_ability_service_url",
    "internal_headers",
    "server_available",
]
