"""Container bootstrap — runs inside RunPod pods before aii_server/aii_pipeline.

Handles: gh auth, wait for aii_server, fetch Claude credentials, preflight.

Usage (from entrypoint scripts):
    python -m aii_launcher.container_init
    python -m aii_launcher.container_init --skip-server-wait   # server pod (IS the server)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def container_init(skip_server_wait: bool = False) -> bool:
    """Run container bootstrap steps. Returns True on success."""
    # 1. GitHub CLI auth
    logger.info("--- Authentication ---")
    from aii_lib.utils.gh_auth import ensure_gh_auth

    ensure_gh_auth(PROJECT_ROOT)

    # 2. Wait for aii_server (skip on server pod — it IS the server)
    if not skip_server_wait:
        import httpx
        from aii_lib.abilities.ability_server import get_ability_service_url
        from aii_lib.utils.internal_auth import internal_headers

        url = get_ability_service_url()
        logger.info(f"--- Waiting for aii_server at {url} ---")
        for i in range(1, 121):
            try:
                r = httpx.get(
                    f"{url}/agent_abilities/health",
                    headers=internal_headers(),
                    timeout=2,
                )
                if r.status_code == 200:
                    count = r.json().get("count", "?")
                    logger.success(f"aii_server reachable ({count} abilities, {i}s)")
                    break
            except Exception as e:
                if i == 1:
                    logger.info(f"Waiting for aii_server... ({e.__class__.__name__})")
            time.sleep(1)
        else:
            logger.warning("aii_server not reachable after 120s — continuing anyway")

    # 3. Fetch Claude credentials
    if skip_server_wait:
        # We ARE the server — credentials are handled by the /claude/credentials
        # endpoint on first request (autologin triggers lazily). Skip HTTP fetch.
        logger.info("--- Credentials ---")
        logger.info("Server pod — credentials will be initialized on first request")
    else:
        logger.info("--- Credentials ---")
        import httpx
        from aii_lib.abilities.ability_server import get_ability_service_url
        from aii_lib.llm_backend.claude_max import aii_claude_dir
        from aii_lib.utils.internal_auth import internal_headers

        url = get_ability_service_url()

        creds_path = aii_claude_dir() / ".credentials.json"
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 37):
            try:
                r = httpx.get(
                    f"{url}/agent_abilities/claude/credentials",
                    headers=internal_headers(),
                    timeout=360,
                )
                if r.status_code == 200:
                    data = r.json()
                    creds = data.get("credentials")
                    if creds:
                        creds_path.write_text(json.dumps(creds))
                        logger.success(
                            f"Credentials ready ({data.get('active_account', '?')}, "
                            f"expires in {data.get('expires_in_human', '?')})"
                        )
                        break
            except Exception as e:
                if attempt % 6 == 0:
                    logger.warning(
                        f"Credential fetch attempt {attempt}: {e.__class__.__name__}: {e}"
                    )
            if attempt == 1:
                logger.info("Waiting for credentials (autologin in progress)...")
            time.sleep(10)
        else:
            logger.warning("Failed to fetch credentials after 360s")

    # 4. Preflight — skip on server pod (ability server isn't up yet; run_server.sh
    # hasn't started it. The orchestrator pod runs preflight against the already-running
    # ability server, which is where the test actually makes sense.)
    if skip_server_wait:
        logger.info("--- Preflight --- (skipped on server pod)")
    else:
        logger.info("--- Preflight ---")
        preflight_path = PROJECT_ROOT / "tests" / "preflight" / "ability.py"
        if preflight_path.exists():
            result = subprocess.run(
                [sys.executable, str(preflight_path)], cwd=PROJECT_ROOT, timeout=300
            )
            if result.returncode != 0:
                logger.warning("Preflight had failures (continuing anyway)")

    return True


def main() -> None:
    """Container bootstrap CLI entry point."""
    parser = argparse.ArgumentParser(description="Container bootstrap for RunPod pods")
    parser.add_argument(
        "--skip-server-wait",
        action="store_true",
        help="Skip waiting for aii_server (used on server pod)",
    )
    args = parser.parse_args()

    # Pre-pipeline boot: ``logger.*`` already routes through loguru; the
    # pipeline.py boot path sets up the live Run + Run-side sinks once
    # config is loaded. Nothing to wire here.

    success = container_init(skip_server_wait=args.skip_server_wait)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
