#!/usr/bin/env python3
"""aii_server — Unified AI Inventor server (Django + abilities).

Usage:
    aii_server                      # start server (uvicorn, port 8020)
    aii_server --dev-frontend       # also start Next.js dev server in tmux
    aii_server                      # auto-detects RunPod from RUNPOD_POD_ID
    aii_server --port 9000          # custom port
    aii_server --workers 4          # multiple uvicorn workers
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from aii_lib.server_url import DEFAULT_SERVER_PORT

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

_cleaned_up = False


_server_port = DEFAULT_SERVER_PORT  # set by main() before any mode runs


def _cleanup():
    """Kill all child sessions and processes. Safe to call multiple times."""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    # Kill child tmux sessions
    from aii_lib.utils.tmux import kill_session

    for session in ("aii-dev-frontend", "aii-prod-frontend", "claude_usage_persistent"):
        kill_session(session)
    # Kill everything on our port (catches uvicorn workers, auto-reloader, etc.)
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{_server_port}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for pid in result.stdout.strip().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError, PermissionError):
                pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


import atexit

atexit.register(_cleanup)
signal.signal(signal.SIGHUP, lambda *_: (_cleanup(), sys.exit(129)))


def _shutdown(*_):
    _cleanup()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_migrations():
    """Run pending migrations silently, then ensure default admin exists.

    Default admin credentials (overridable via env):
      AII_ADMIN_USERNAME (default: admin)
      AII_ADMIN_EMAIL    (default: admin@aii.local)
      AII_ADMIN_PASSWORD (default: admin)

    Idempotent: if a user with that username already exists, do nothing.
    Skipped entirely when WEB_APP_MODE is False (no auth/users in that mode).
    """
    import django

    django.setup()
    from django.conf import settings as _s
    from django.core.management import call_command

    call_command("migrate", "--run-syncdb", verbosity=0)
    if not getattr(_s, "WEB_APP_MODE", False):
        return
    from django.contrib.auth import get_user_model
    from loguru import logger

    User = get_user_model()
    username = os.environ.get("AII_ADMIN_USERNAME", "admin")
    email = os.environ.get("AII_ADMIN_EMAIL", "admin@aii.local")
    password = os.environ.get("AII_ADMIN_PASSWORD", "admin")
    if User.objects.filter(username=username).exists():
        return
    user = User.objects.create_superuser(username=username, email=email, password=password)
    # Mark the email as verified so email-based login works too. Without
    # this, allauth's mandatory verification would block email login (the
    # username-login path would still work).
    try:
        from allauth.account.models import EmailAddress

        EmailAddress.objects.get_or_create(
            user=user,
            email=email,
            defaults={"verified": True, "primary": True},
        )
    except ImportError:
        pass
    logger.success(f"Created default admin user '{username}'")


def _dump_openapi_snapshot():
    """Write the live OpenAPI spec to ``aii_frontend/lib/api/openapi.json``.

    Must run AFTER ``django.setup()`` and AFTER all ``AppConfig.ready()``
    hooks have fired — calling ``api.get_openapi_schema()`` freezes every
    NinjaAPI router reachable from ``ROOT_URLCONF`` (including
    ``abilities_api``), so doing this from inside ``ready()`` would
    block ``agent_abilities`` from registering its dynamic routes.

    No-op when the FE dir isn't present (Docker server image, runpod
    worker image, etc.) — those builds use the snapshot baked into
    the FE build context separately.
    """
    import json

    from loguru import logger

    try:
        from dashboard.api import api as ninja_api
    except Exception as e:
        logger.warning(f"OpenAPI snapshot dump skipped (no dashboard api): {e}")
        return
    fe_spec = PROJECT_ROOT / "aii_frontend" / "lib" / "api" / "openapi.json"
    if not fe_spec.parent.is_dir():
        return
    try:
        spec = ninja_api.get_openapi_schema()
        new_text = json.dumps(spec, indent=2, default=str)
        # Skip the write when content is identical — saves ~50ms file IO
        # per server boot and keeps the file's mtime stable so orval /
        # codegen tools that watch it don't get spurious "changed" pings.
        if fe_spec.exists():
            try:
                if fe_spec.read_text(encoding="utf-8") == new_text:
                    logger.debug(f"OpenAPI snapshot unchanged: {fe_spec}")
                    return
            except OSError:
                pass
        fe_spec.write_text(new_text, encoding="utf-8")
        logger.success(f"OpenAPI snapshot written: {fe_spec}")
    except Exception as e:
        logger.warning(f"OpenAPI snapshot dump failed: {e}")


def _start_frontend_dev():
    """Start Next.js dev server + Storybook in tmux (for --dev-frontend)."""
    from aii_lib.utils.tmux import launch_in_tmux

    frontend_dir = PROJECT_ROOT / "aii_frontend"
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not (frontend_dir / "package.json").exists():
        return

    launch_in_tmux(
        session="aii-dev-frontend",
        cmd="npm run dev",
        cwd=str(frontend_dir),
        log_file=str(logs_dir / "aii_frontend.log"),
    )
    launch_in_tmux(
        session="aii-storybook",
        cmd="npm run storybook -- --no-open",
        cwd=str(frontend_dir),
        log_file=str(logs_dir / "storybook.log"),
    )

    from loguru import logger

    logger.info("Next.js dev server started (tmux: aii-dev-frontend)")
    logger.info("Storybook started on :6006 (tmux: aii-storybook)")


def _start_frontend_prod():
    """Build + serve the production Next.js frontend (for --prod-frontend).

    Boot sequence: ``npm run build && npm run start`` chained inside one
    tmux session so the build runs first and ``next start`` only starts
    once the build artifacts exist. The build typically takes 1–2 min;
    port 3000 is unreachable until it finishes — that's expected.

    No Storybook (dev-only). Same port (3000) and same shape as
    ``--dev-frontend``, just optimized output.
    """
    from aii_lib.utils.tmux import launch_in_tmux

    frontend_dir = PROJECT_ROOT / "aii_frontend"
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not (frontend_dir / "package.json").exists():
        return

    launch_in_tmux(
        session="aii-prod-frontend",
        cmd="npm run build && npm run start",
        cwd=str(frontend_dir),
        log_file=str(logs_dir / "aii_frontend.log"),
    )

    from loguru import logger

    logger.info(
        "Next.js prod frontend launching in tmux 'aii-prod-frontend' "
        "(build + start; port 3000 reachable after build completes — typically 1-2 min)"
    )


def _prewarm_claude_credentials_background(port: int):
    """Trigger one-time Claude credential refresh so the first agent doesn't 401.

    ``AccountManager`` only loads cookie files at boot — it never proactively
    refreshes the OAuth token in ``.credentials.json``. The first caller to
    hit Claude with an aged token gets 401 and falls into the 60s autologin
    retry path (``error_recovery._handle_subscription_error``). Hitting
    ``/agent_abilities/claude/credentials`` once at boot triggers
    refresh-if-expired so the first agent reads a fresh token.

    Idempotent (preflight also calls this endpoint); endpoint serialises
    concurrent refreshes itself.
    """

    def _worker():
        import time

        import httpx
        from aii_lib.utils.internal_auth import internal_headers
        from loguru import logger

        # Wait for Django to be serving (any response on health, even 503)
        url_health = f"http://127.0.0.1:{port}/agent_abilities/health"
        url_creds = f"http://127.0.0.1:{port}/agent_abilities/claude/credentials"
        for _ in range(60):
            try:
                r = httpx.get(url_health, headers=internal_headers(), timeout=2)
                if r.status_code in (200, 503):
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            return  # Server never came up — nothing to pre-warm against.

        try:
            r = httpx.get(url_creds, headers=internal_headers(), timeout=180)
            if r.status_code == 200:
                logger.success("Claude credentials pre-warmed")
            else:
                logger.warning(
                    f"Credential pre-warm returned {r.status_code} — "
                    f"first agent may hit the 60s 401 path",
                )
        except Exception as e:
            logger.warning(f"Credential pre-warm skipped: {e}")

    t = threading.Thread(
        target=_worker,
        name="claude-creds-prewarm",
        daemon=True,
    )
    t.start()


def _run_preflight_background(port: int) -> threading.Thread:
    """Run preflight in a background thread. Health returns 503 until done."""

    def _worker():
        import asyncio
        import time

        import httpx
        from agent_abilities.api import mark_ready
        from loguru import logger

        # Wait for Django to be serving (any response on health, even 503)
        for _i in range(1, 121):
            try:
                from aii_lib.utils.internal_auth import internal_headers

                r = httpx.get(
                    f"http://localhost:{port}/agent_abilities/health",
                    headers=internal_headers(),
                    timeout=2,
                )
                if r.status_code in (200, 503):
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            mark_ready()
            return

        try:
            sys.path.insert(0, str(PROJECT_ROOT / "tests"))
            from preflight.ability import main as preflight_main

            rc = asyncio.run(preflight_main(include_runpod=True, include_login=True))
            if rc != 0:
                logger.warning("Preflight had failures")
        except ModuleNotFoundError:
            # Public/open-source builds ship without tests/preflight — skip silently.
            pass
        except Exception as e:
            logger.exception(f"Preflight crashed: {e}")

        mark_ready()
        logger.success("aii_server ready")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _free_port(port: int):
    """Kill any process occupying the port (stale orphan from previous run)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = result.stdout.strip().split()
        if pids:
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            import time

            time.sleep(0.5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _run_server(host: str, port: int, workers: int):
    """Default mode: uvicorn with background preflight."""
    import uvicorn

    _free_port(port)
    _prewarm_claude_credentials_background(port)
    _run_preflight_background(port)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        uvicorn.run(
            "config.asgi:application",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            log_config=None,
        )
    finally:
        _cleanup()


def _run_runpod(host: str, port: int, workers: int):
    """RunPod mode: gh auth, start server, credentials, preflight, then serve.

    This is the "server pod" bootstrap. Worker-pod bootstrap lives in
    aii_launcher/src/aii_launcher/container_init.py. The two paths intentionally
    diverge — keep these differences in mind when changing either:
      * gh_auth — IDENTICAL (uses ensure_gh_auth, same args).
      * server health-wait — server pod waits for *itself* (localhost); worker
        pod waits for the *remote* server (get_ability_service_url()).
      * credentials — server pod GETs /claude/credentials from itself to prime
        autologin (no disk write); worker pod GETs from remote and writes to
        ~/.claude/.credentials.json.
      * preflight — IDENTICAL (same script, same cwd, same timeout).
    """
    import time

    import httpx
    import uvicorn
    from agent_abilities.api import mark_ready
    from loguru import logger

    _free_port(port)

    # GitHub CLI auth (mirrors container_init step 1 exactly)
    logger.info("--- Authentication ---")
    from aii_lib.utils.gh_auth import ensure_gh_auth

    ensure_gh_auth(PROJECT_ROOT)

    # Start Django in background thread
    logger.info("--- Starting aii_server ---")
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": "config.asgi:application",
            "host": host,
            "port": port,
            "workers": workers,
            "log_level": "info",
            "log_config": None,
        },
        daemon=True,
    )
    server_thread.start()

    # Wait for uvicorn to start serving — accept any response (200 or 503).
    # Health returns 503 ("starting") until mark_ready() is called below
    # after preflight, so we'd never see 200 here. We just need uvicorn up.
    for i in range(1, 121):
        try:
            from aii_lib.utils.internal_auth import internal_headers

            r = httpx.get(
                f"http://localhost:{port}/agent_abilities/health",
                headers=internal_headers(),
                timeout=2,
            )
            if r.status_code in (200, 503):
                count = r.json().get("count", "?")
                logger.success(f"aii_server up ({i}s) — {count} abilities")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        logger.error("aii_server not responding after 120s")
        sys.exit(1)

    # Claude credentials
    logger.info("--- Claude Code Authentication ---")
    try:
        from aii_lib.utils.internal_auth import internal_headers

        r = httpx.get(
            f"http://localhost:{port}/agent_abilities/claude/credentials",
            headers=internal_headers(),
            timeout=360,
        )
        if r.status_code == 200:
            data = r.json()
            logger.success(
                f"Credentials ready ({data.get('active_account', '?')}, "
                f"expires in {data.get('expires_in_human', '?')})"
            )
        else:
            logger.warning("Credential fetch failed — preflight will retry")
    except Exception as e:
        logger.warning(f"Credential fetch failed: {e}")

    # Preflight (blocking — RunPod needs it done before signaling ready)
    logger.info("--- Preflight ---")
    preflight_path = PROJECT_ROOT / "tests" / "preflight" / "ability.py"
    if preflight_path.exists():
        result = subprocess.run([sys.executable, str(preflight_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            logger.warning("Preflight had failures (continuing anyway)")

    # Flip health to 200 — preflight failures aren't fatal for the dashboard,
    # so always mark ready (mirrors _run_server's _run_preflight_background
    # which calls mark_ready() unconditionally after the subprocess returns).
    mark_ready()
    logger.success("Server ready")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server_thread.join()
    except KeyboardInterrupt:
        _cleanup()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_server_config() -> dict:
    """Load aii_config/server/server.yaml + server.private.yaml override for CLI defaults."""
    from aii_lib.utils.config_overrides import load_config_with_overrides

    cfg_path = PROJECT_ROOT / "aii_config" / "server" / "server.yaml"
    if cfg_path.exists():
        return load_config_with_overrides(cfg_path)
    return {}


def main():
    """Just run the server. No tmux, no orchestration — aii_launcher handles that."""
    cfg = _load_server_config().get("server", {})

    parser = argparse.ArgumentParser(description="AI Inventor server", prog="aii_server")
    fe_group = parser.add_mutually_exclusive_group()
    fe_group.add_argument(
        "--dev-frontend", action="store_true", help="Start Next.js dev server in tmux"
    )
    fe_group.add_argument(
        "--prod-frontend",
        action="store_true",
        help="Build + serve the production Next.js frontend in tmux",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=cfg.get("port", DEFAULT_SERVER_PORT),
        help="Port (default: from server.yaml)",
    )
    parser.add_argument(
        "--host",
        default=cfg.get("host", "0.0.0.0"),
        help="Host (default: from server.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg.get("workers", 1),
        help="Workers (default: from server.yaml)",
    )
    args = parser.parse_args()

    global _server_port
    _server_port = args.port

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    os.chdir(Path(__file__).resolve().parent)

    # Apply Claude SDK telemetry transport env vars before Django boots —
    # autologin / per-run dispatch will spawn ClaudeSDKClient subprocesses
    # which inherit os.environ, and the CLI reads OTEL_EXPORTER_OTLP_* at
    # subprocess-launch time.
    from aii_lib.agent_backend.claude_agent_sdk.sdk_telemetry import configure_sdk_telemetry

    configure_sdk_telemetry()

    # No LiteLLM proxy boot. The proxy existed only to bridge
    # claude_agent_sdk + openrouter, which is no longer a supported pair
    # (rejected at config load by
    # ``PipelineConfig._validate_backend_pairings``). The openrouter
    # llm_backend's direct consumption path goes through
    # ``OpenRouterClient.chat`` straight to OpenRouter, no proxy needed.

    import django

    django.setup()
    from django.conf import settings

    if settings.WEB_APP_MODE:
        _run_migrations()
        _dump_openapi_snapshot()

    if settings.WEB_APP_MODE and args.dev_frontend:
        _start_frontend_dev()
    elif settings.WEB_APP_MODE and args.prod_frontend:
        _start_frontend_prod()

    from aii_lib.utils.run_mode import is_runpod

    if settings.WEB_APP_MODE and is_runpod():
        _run_runpod(args.host, args.port, args.workers)
    else:
        _run_server(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()
