"""Abilities API — Django Ninja router for all ability endpoints.

Dynamically creates POST /agent_abilities/{name} for each @aii_ability.
Each ability runs in a persistent worker process with its own venv.
Also serves /agent_abilities/claude/* and /agent_abilities/health.
"""

import inspect
import json
import time
from pathlib import Path

from aii_lib.abilities.ability_server.logging_config import with_retry
from aii_lib.abilities.aii_ability import ability_to_openai_tool, get_registry
from django.conf import settings as django_settings
from django.http import HttpRequest, JsonResponse
from loguru import logger
from ninja import NinjaAPI

from .worker import get_or_create_worker, start_all_workers

_internal_auth = getattr(django_settings, "INTERNAL_AUTH", None)

abilities_api = NinjaAPI(
    title="AI Abilities",
    version="1.0.0",
    urls_namespace="abilities",
    auth=_internal_auth,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

_ready = False


def mark_ready():
    """Called after preflight completes."""
    global _ready
    _ready = True


@abilities_api.get("/health", auth=None)
def health(request: HttpRequest):
    """Readiness: preflight passed, ready for traffic.

    ``auth=None`` overrides the API-level bearer requirement — this is a
    liveness probe consumed by RunPod proxy + ``aii_launcher --runpod`` (which
    has no way to know the pod's auto-generated AII_INTERNAL_KEY). The
    response carries no secrets — just an endpoint name list.
    """
    registry = get_registry()
    status = "ok" if _ready else "starting"
    code = 200 if _ready else 503
    return JsonResponse(
        {
            "status": status,
            "endpoints": sorted(registry.keys()),
            "count": len(registry),
        },
        status=code,
    )


# ---------------------------------------------------------------------------
# Tools schema (OpenAI format)
# ---------------------------------------------------------------------------


@abilities_api.get("/tools")
def tools(request: HttpRequest):
    registry = get_registry()
    return [ability_to_openai_tool(name) for name in sorted(registry.keys())]


# ---------------------------------------------------------------------------
# Claude credentials / usage / accounts
# ---------------------------------------------------------------------------


@abilities_api.get("/claude/credentials")
def claude_credentials(request: HttpRequest, reason: str = ""):
    from .state import ensure_credentials_state

    ensure_credentials_state()
    from aii_lib.abilities.ability_server.credentials import api_get_credentials

    status, data = api_get_credentials(reason=reason or None)
    return JsonResponse(data, status=status)


@abilities_api.get("/claude/usage")
def claude_usage(request: HttpRequest):
    from .state import ensure_credentials_state

    ensure_credentials_state()
    from aii_lib.abilities.ability_server.credentials import api_get_usage

    status, data = api_get_usage()
    return JsonResponse(data, status=status)


@abilities_api.get("/claude/accounts")
def claude_accounts(request: HttpRequest):
    from .state import ensure_credentials_state

    ensure_credentials_state()
    from aii_lib.abilities.ability_server.credentials import api_get_accounts

    status, data = api_get_accounts()
    return JsonResponse(data, status=status)


# ---------------------------------------------------------------------------
# Dynamic ability endpoints — persistent worker processes
# ---------------------------------------------------------------------------


def register_ability_routes():
    """Create POST routes for each discovered @aii_ability.

    Each ability gets a persistent worker process with:
    - Its own venv (declared in @aii_ability decorator)
    - worker_init run once at startup
    - ThreadPoolExecutor for concurrent requests within the process

    Called after discovery runs (from apps.py ready()).
    """
    registry = get_registry()
    log = logger.bind(source="abilities")

    for name, meta in sorted(registry.items()):
        func = meta.get("func")
        if func is None:
            continue

        # Resolve venv path from decorator metadata
        venv_path = None
        venv_rel = meta.get("venv")
        if venv_rel:
            try:
                script_dir = Path(inspect.getfile(func)).resolve().parent
                venv_path = str((script_dir / venv_rel).resolve())
            except (TypeError, OSError):
                pass

        # Resolve worker_init function
        worker_init_fn = None
        worker_init_name = meta.get("worker_init")
        if worker_init_name:
            import sys

            module = sys.modules.get(func.__module__)
            if module:
                worker_init_fn = getattr(module, worker_init_name, None)

        max_workers = meta.get("max_workers", 10)
        ability_timeout = meta.get("timeout", 180.0)
        num_retries = meta.get("retries", 3)

        # Wrap handler with retry logic (runs inside worker process)
        # Use factory function to avoid late-binding closure bug
        # Filter request keys to only accepted params (prevents unknown kwarg crashes)
        def _make_handler(_func, _retries):
            import inspect as _inspect

            sig = _inspect.signature(_func)
            _accepted = set(sig.parameters.keys())
            _has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

            def _filter(req: dict) -> dict:
                if _has_kwargs:
                    return _func(**req)
                filtered = {k: v for k, v in req.items() if k in _accepted}
                return _func(**filtered)

            if _retries > 0:

                @with_retry(max_retries=_retries)
                def _retried(req: dict) -> dict:
                    return _filter(req)

                return _retried

            def _direct(req: dict) -> dict:
                return _filter(req)

            return _direct

        handler = _make_handler(func, num_retries)

        # Create the worker handle (process starts lazily on first request or at boot)
        worker = get_or_create_worker(
            name=name,
            handler=handler,
            worker_init=worker_init_fn,
            max_workers=max_workers,
            venv_path=venv_path,
        )

        def _make_endpoint(_name, _worker, _timeout):
            async def endpoint(request: HttpRequest):
                try:
                    body = json.loads(request.body) if request.body else {}
                except json.JSONDecodeError:
                    return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

                # Extract timeout override from request body
                req_timeout = body.pop("_timeout", None)
                timeout = float(req_timeout) if req_timeout else _timeout

                start = time.perf_counter()
                result = await _worker.call(body, timeout=timeout)
                elapsed = time.perf_counter() - start

                success = result.get("success", True)
                if success:
                    status = 200
                elif "traceback" in result:
                    status = 500
                    # Strip traceback from response in production
                    from django.conf import settings as django_settings

                    if not django_settings.DEBUG:
                        result.pop("traceback", None)
                else:
                    # success=False without traceback = handled error returned
                    # by the ability (search miss, upstream capacity, expected
                    # failure). The HTTP layer succeeded — the body conveys
                    # the operation result. Returning 200 lets the caller
                    # branch on ``result["success"]`` instead of HTTP status,
                    # avoids spamming 422 WARNING logs for benign cases (e.g.
                    # Loogle "no match"), and prevents the ability_client's
                    # transient-vs-permanent heuristic from misclassifying
                    # transient upstream issues (RunPod capacity) as 422.
                    status = 200
                log.debug(f"POST /{_name} {status} ({elapsed:.1f}s)")
                return JsonResponse(result, status=status)

            endpoint.__name__ = f"ability_{_name}"
            endpoint.__qualname__ = f"ability_{_name}"
            return endpoint

        view = _make_endpoint(name, worker, ability_timeout)
        abilities_api.post(f"/{name}")(view)

    # Catch-all for unknown ability names (must be registered last)
    @abilities_api.post("/{name}")
    def ability_not_found(request, name: str):
        return JsonResponse({"detail": f"Unknown ability: {name}"}, status=404)

    # Start all worker processes
    start_all_workers()
    log.info(f"Registered {len(registry)} ability routes (worker processes)")
