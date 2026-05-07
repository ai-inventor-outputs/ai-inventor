"""Request logging middleware — fans every request through ``logger``.

Emits ``server_request`` / ``server_error`` events with source-based
tagging. Source determines color in the console: abilities=gray,
dashboard=blue, auth=purple, sse=teal, static=dim, etc.
"""

import json
import time

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from loguru import logger

_REDACT_KEYS = frozenset(
    {
        "accessToken",
        "refreshToken",
        "access_token",
        "refresh_token",
        "password",
        "current_password",
        "new_password",
        "secret",
        "api_key",
        "apiKey",
        "token",
    }
)


def _redact(obj):
    """Deep-redact sensitive keys from a JSON-serializable object."""
    if isinstance(obj, dict):
        return {k: "***" if k in _REDACT_KEYS else _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _summarize_dict(d: dict) -> str:
    """Summarize a dict as key=value pairs, truncating each value."""
    n: int = settings.AII_LOG_TRUNCATE_CHARS
    parts = []
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        s = str(v)
        if len(s) > n:
            s = f"{s[:n]}…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _classify_source(path: str) -> str:
    """Map request path to a source tag for coloring."""
    if path.startswith("/agent_abilities/"):
        return "abilities"
    if path.startswith("/api/"):
        return "dashboard"
    if path.startswith(("/accounts/", "/auth/")):
        return "auth"
    if path.startswith(("/static/", "/_next/")) or path == "/favicon.ico":
        return "dist"
    return "dashboard"


class RequestLogger:
    """Log all requests via the server logger with source-based tagging."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start = time.perf_counter()
        response = self.get_response(request)
        elapsed = time.perf_counter() - start

        # Ninja swallows view exceptions and returns a JSON 500 with no
        # traceback, so the only thing that lands in the log is "Internal
        # Server Error: <path>" and we can't see what actually broke. The
        # exception object lives on the request after dispatch — copy its
        # traceback into the response so the 500 branch below can log it.
        request_exc = getattr(request, "_aii_view_exception", None)
        if request_exc is not None and not hasattr(response, "_aii_traceback"):
            response._aii_traceback = request_exc

        method = request.method
        path = request.path
        status = response.status_code
        source = _classify_source(path)

        # Build log message
        parts = [f"{method} {path} {status} ({elapsed:.1f}s)"]

        # Request body summary (POST/PUT/PATCH) — skip multipart (file uploads)
        content_type = request.content_type or ""
        if method in ("POST", "PUT", "PATCH") and "multipart" not in content_type and request.body:
            try:
                body = _redact(json.loads(request.body))
                parts.append(f"req: {_summarize_dict(body)}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Response summary (JSON only, secrets redacted)
        if hasattr(response, "content") and response.get("Content-Type", "").startswith(
            "application/json"
        ):
            try:
                resp_data = _redact(json.loads(response.content))
                if isinstance(resp_data, dict):
                    parts.append(f"resp: {_summarize_dict(resp_data)}")
                elif isinstance(resp_data, list):
                    parts.append(f"resp: [{len(resp_data)} items]")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        log_msg = " | ".join(parts)

        if status == 503 and path.endswith("/health"):
            # Health 503 during startup — expected, not an error.
            pass
        elif status >= 500:
            # server_error with traceback if available
            error_msg = log_msg
            tb = getattr(response, "_aii_traceback", None)
            if tb:
                error_msg += f"\n{tb}"
            elif hasattr(response, "content"):
                try:
                    err_data = json.loads(response.content)
                    body_tb = err_data.get("traceback", "")
                    if body_tb:
                        error_msg += f"\n{body_tb}"
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            logger.bind(type="server_error", source=source).error(error_msg)
        else:
            logger.bind(type="server_request", source=source).info(log_msg)

        return response
