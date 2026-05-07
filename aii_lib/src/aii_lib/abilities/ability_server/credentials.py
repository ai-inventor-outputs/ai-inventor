"""Claude credentials and usage scraping for ability server.

Account management, OAuth token refresh, usage monitoring,
and credential endpoints.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from aii_lib.server_url import SERVER_PORT

from .logging_config import _server_config

if TYPE_CHECKING:
    from aii_lib.llm_backend.claude_max.autologin import AccountManager

# Port for self-referencing calls (usage scraper hitting credentials endpoint).
# Mirrored from server.yaml via aii_lib.server_url.
DEFAULT_PORT = SERVER_PORT

_PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _init_account_manager() -> AccountManager | None:
    """Initialize AccountManager for multi-account support.

    Returns None if no accounts are configured (autologin deps missing, etc.)
    """
    log = logger.bind(source="server")
    try:
        from aii_lib.llm_backend.claude_max.autologin import AccountManager

        mgr = AccountManager()
        if not mgr.accounts:
            log.warning("AccountManager has no accounts configured")
            return None
        log.info(f"AccountManager initialized with {len(mgr.accounts)} account(s)")
        return mgr
    except ImportError:
        log.warning("autologin module not available — no account management")
        return None
    except Exception as e:
        log.exception(f"AccountManager init failed: {e}")
        return None


def _inject_local_cookies(mgr: AccountManager) -> None:
    """Extract cookies from local Chrome profiles and write to account dirs.

    Local-mode equivalent of RunPod's init_pod_runpod.sh B64 decoding.
    Reuses the same export_cookies() used for remote deployment.
    """
    log = logger.bind(source="claude-creds")

    # Build account_priority list in the format export_cookies() expects,
    # tracking which Account each entry maps to (indices may differ if some
    # accounts lack chrome_profile).
    accounts_with_profiles: list = []
    account_priority = []
    for acc in mgr.accounts:
        if not acc.chrome_profile:
            log.warning(f"{acc}: no chrome_profile configured — skipping cookie injection")
            continue
        accounts_with_profiles.append(acc)
        account_priority.append({"email": acc.claude_email, "chrome_profile": acc.chrome_profile})

    if not account_priority:
        log.warning("No accounts with chrome_profile — skipping local cookie injection")
        return

    try:
        import base64

        from aii_runpod.deploy.cookies import export_cookies

        env_vars = export_cookies(account_priority)

        for env_key, b64_val in env_vars.items():
            # ``export_cookies`` returns paired ``_B64`` + ``_KEY`` entries
            # per account; act on ``_B64`` only (the ``_KEY`` siblings are
            # consumed by RunPod transport, irrelevant in local mode since
            # AccountManager already knows the right subdir).
            if not env_key.endswith("_B64"):
                continue
            idx = int(env_key.split("_")[3])
            account = accounts_with_profiles[idx]
            account.ensure_dirs()

            session_data = base64.b64decode(b64_val).decode()
            account.web_session_path.write_text(session_data)
            log.success(f"Injected Chrome cookies for {account}")

    except ImportError:
        log.warning("browser_cookie3 not installed — cannot extract Chrome cookies locally")
    except Exception as e:
        log.exception(f"Local cookie injection failed: {e}\n{traceback.format_exc()}")


def _identify_and_align_active_account(mgr: AccountManager) -> None:
    """Align AccountManager's active index to the account matching standard credentials.

    Align AccountManager's active index to the account whose per-account
    credentials match the current standard ``~/.claude/.credentials.json``.

    Compares OAuth access tokens directly — does NOT use ``claude auth status``
    because it returns the org owner's email, not the authenticating user's email,
    which causes cross-contamination of credentials between accounts.
    """
    from aii_lib.llm_backend.claude_max import aii_claude_dir

    log = logger.bind(source="claude-creds")
    creds_path = aii_claude_dir() / ".credentials.json"

    if not creds_path.exists():
        log.info("No standard credentials — skipping alignment")
        return

    try:
        std_creds = json.loads(creds_path.read_text())
        std_token = std_creds.get("claudeAiOauth", {}).get("accessToken", "")
        if not std_token:
            log.warning("Standard credentials have no access token — skipping alignment")
            return
    except Exception as e:
        log.warning(f"Could not read standard credentials: {e}")
        return

    # Match by comparing access tokens directly
    for i, acc in enumerate(mgr.accounts):
        if not acc.credentials_path.exists():
            continue
        try:
            acc_creds = json.loads(acc.credentials_path.read_text())
            acc_token = acc_creds.get("claudeAiOauth", {}).get("accessToken", "")
            if acc_token == std_token:
                mgr._active_index = i
                log.info(f"Active account aligned to {acc} (token match)")
                return
        except Exception:
            continue

    # No match found — standard creds belong to an unknown account (e.g. the
    # user's own Claude Code session).  Just set the index; do NOT call
    # activate() which would overwrite/delete the existing credentials and
    # log out the running Claude Code instance. Logged at info (was warning)
    # — this is the designed safeguard path, not an actual problem.
    log.info(
        "Standard credentials don't match any account — setting index to Account#1 (preserving existing creds)"
    )
    mgr._active_index = 0


# Global account manager (set during bootstrap)
_account_manager: AccountManager | None = None

# Shared state for account switching — accessed by both /claude/credentials
# and /claude/usage endpoints.
_refresh_lock = threading.Lock()
_switch_generation: int = 0

# Shared usage cache — invalidated on account switch so /claude/usage
# returns fresh data for the new account immediately.
_usage_cache: dict = {"usage": None, "timestamp": 0.0}
_usage_lock = threading.Lock()  # serializes scrape calls + cache writes

# Signals that autologin is complete — usage poll thread waits for this
# before its first scrape (Claude CLI isn't authenticated until then).
_autologin_done = threading.Event()


def _invalidate_usage_cache() -> None:
    """Clear the usage cache and kick the poll thread to refresh now.

    Without the kick, the next poll runs at _USAGE_POLL_INTERVAL (5min)
    so any /usage call within that window would wait the full 120s for
    the cache to populate. After a switch we want the new account's
    usage available within seconds, not minutes.
    """
    _usage_cache["usage"] = None
    _usage_cache["timestamp"] = 0.0
    # Fire an out-of-band scrape in a daemon thread so the next /usage
    # call hits a populated cache instead of timing out.
    #
    # Anthropic's session/usage data takes time to propagate after a
    # switch — empirically up to ~40s before the new account's session
    # is reflected. We wait 20s before the first attempt, then retry up
    # to 3 times spaced 20s apart. Any one success populates the cache
    # and exits early.
    if _standalone_scrape_usage is not None:

        def _kick():
            log = logger.bind(source="claude-usage")
            for attempt in range(1, 4):
                time.sleep(20)
                try:
                    with _usage_lock:
                        result = _standalone_scrape_usage()
                        if result:
                            _usage_cache["usage"] = result
                            _usage_cache["timestamp"] = time.time()
                            log.info(f"Post-switch usage cache populated (attempt {attempt}/3)")
                            return
                        log.info(
                            f"Post-switch scrape attempt {attempt}/3 returned "
                            f"empty — will retry in 20s"
                        )
                except Exception as e:
                    log.warning(f"Post-switch scrape attempt {attempt}/3 failed: {e}")
            log.warning(
                "Post-switch scrape failed after 3 attempts (60s+ wait) — "
                "next regular poll will populate cache"
            )

        threading.Thread(target=_kick, daemon=True, name="claude-usage-kick").start()


def _switch_and_auth(gen_at_request: int | None = None) -> bool:
    """Switch to next account and authenticate it. Blocking.

    Tries each remaining account in round-robin order. If OAuth fails
    on one account, advances to the next until all have been tried.

    Args:
        gen_at_request: The switch generation when the caller decided to
            switch. If another switch happened while we waited for the
            lock (generation advanced), we skip the redundant switch
            and just return True (credentials already refreshed).
    """
    global _switch_generation
    with _refresh_lock:
        log = logger.bind(source="claude-creds")

        # If a switch already happened since the caller's request, skip.
        if gen_at_request is not None and _switch_generation > gen_at_request:
            log.info(
                f"Switch already happened (gen {gen_at_request} → {_switch_generation}) "
                f"— skipping redundant switch, active: {_account_manager.active}"
            )
            return True

        from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token_for_account

        # Try each remaining account (at most N-1 switches for N accounts)
        max_attempts = len(_account_manager.accounts) - 1
        for _attempt in range(max_attempts):
            new_account = _account_manager.switch_to_next()
            if not new_account:
                return False
            log.info(f"Switched to {new_account} — authenticating...")
            try:
                success = ensure_oauth_token_for_account(new_account, max_retries=2)
                if success:
                    _switch_generation += 1  # only increment after successful auth
                    _account_manager.save_credentials_for_active()
                    log.success(f"Authenticated {new_account} (gen={_switch_generation})")
                    return True
                log.warning(f"OAuth failed for {new_account} — trying next account...")
            except Exception as e:
                log.exception(
                    f"Failed to authenticate {new_account}: {e}\n{traceback.format_exc()} — trying next account..."
                )

        log.error("All accounts exhausted — no working credentials")
        return False


# =============================================================================
# Framework-agnostic API (callable from Django or any HTTP framework)
# =============================================================================
# These functions encapsulate the endpoint logic without FastAPI dependency.
# They return (status_code, response_dict) tuples.

_creds_initialized = False
_creds_init_lock = threading.Lock()
_usage_poll_started = False
_usage_poll_lock = threading.Lock()


def _ensure_claude_onboarding_state() -> None:
    """Ensure .claude.json marks onboarding as complete.

    The Claude CLI shows theme/syntax/login screens on first run and only
    sets ``hasCompletedOnboarding`` after the user clicks through them. When
    we set ``CLAUDE_CONFIG_DIR=aii_data/.claude``, the CLI starts with a
    fresh dir and shows those screens — which the OAuth/usage scrapers
    can't always navigate cleanly, so the inner CLI ends up at the login
    prompt and the scrape raises UsageRateLimitedError.

    Pre-marking the dir as onboarded lets the CLI go straight to the main
    TUI (only the workspace-trust dialog remains, which existing code
    already handles).
    """
    import json as json_mod

    from aii_lib.llm_backend.claude_max import aii_claude_dir

    cfg_path = aii_claude_dir() / ".claude.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json_mod.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    except (json_mod.JSONDecodeError, OSError):
        cfg = {}
    if cfg.get("hasCompletedOnboarding") is True:
        return
    cfg["hasCompletedOnboarding"] = True
    cfg_path.write_text(json_mod.dumps(cfg, indent=2))
    logger.info(f"Marked {cfg_path} as onboarded — inner CLI will skip first-run screens")


def init_credentials_state() -> None:
    """Initialize account manager + cookies (call once before using API functions)."""
    global _creds_initialized, _account_manager
    if _creds_initialized:
        return
    with _creds_init_lock:
        if _creds_initialized:
            return
        _ensure_claude_onboarding_state()
        _account_manager = _init_account_manager()
        from aii_lib.utils.run_mode import is_local

        if is_local() and _account_manager:
            _inject_local_cookies(_account_manager)
            _identify_and_align_active_account(_account_manager)
        # Release the autologin gate up-front when creds are already valid;
        # otherwise the usage-poll thread blocks on _autologin_done until
        # /credentials is hit and every pipeline run pays a 120s tax.
        from aii_lib.llm_backend.claude_max.autologin.token_utils import (
            check_oauth_token_expired,
        )

        if not check_oauth_token_expired():
            _autologin_done.set()
            logger.info("Existing OAuth token valid — autologin gate released")
        _creds_initialized = True


def start_usage_polling() -> None:
    """Start the background usage polling thread (call once)."""
    global _usage_poll_started
    if _usage_poll_started:
        return
    with _usage_poll_lock:
        if _usage_poll_started:
            return
        _do_start_usage_poll()
        _usage_poll_started = True


def _do_start_usage_poll() -> None:
    """Start a background thread that polls Claude usage."""
    from dataclasses import asdict

    from aii_lib.utils.config_overrides import load_config_with_overrides

    _USAGE_POLL_INTERVAL = 300.0

    # Load thresholds — must merge ``llm_backend.private.yaml`` overlay so
    # local-machine overrides (e.g. zeroed thresholds for switching tests)
    # are honoured. A raw ``yaml.safe_load`` of the public file silently
    # discards the private sibling.
    _cfg_path = _PROJECT_ROOT / "aii_config" / "pipeline" / "harness" / "llm_backend.yaml"
    _usage_thresholds: dict = {}
    try:
        pcfg = load_config_with_overrides(_cfg_path)
        _usage_thresholds = (
            pcfg.get("claude_max", {}).get("usage_tracking", {}).get("thresholds", {})
        )
    except Exception as e:
        logger.bind(source="claude-usage").warning(
            f"Failed to load usage thresholds from {_cfg_path}: {e}"
        )

    def _is_over_threshold(usage_dict: dict) -> bool:
        for key in (
            "current_session",
            "current_week_all_models",
            "current_week_sonnet",
        ):
            val = usage_dict.get(key)
            th = _usage_thresholds.get(key)
            if th is not None and val is not None and val >= th:
                return True
        return False

    def _build_result(usage_dict: dict, **extra) -> dict:
        return {
            "success": True,
            "usage": usage_dict,
            "thresholds": _usage_thresholds,
            "over_threshold": _is_over_threshold(usage_dict),
            **extra,
        }

    def _scrape_once() -> dict | None:
        try:
            from aii_lib.llm_backend.claude_max.usage import get_claude_usage

            usage = get_claude_usage()
            usage_dict = asdict(usage)
            active_email = (
                _account_manager.active.claude_email
                if _account_manager and _account_manager.active
                else None
            )
            return _build_result(usage_dict, active_account=active_email)
        except Exception as e:
            logger.bind(source="claude-usage").error(f"Usage scrape failed: {e}")
            return None

    # Store scrape function for external use
    global _standalone_scrape_usage
    _standalone_scrape_usage = _scrape_once

    def _poll_loop():
        log = logger.bind(source="claude-usage")
        log.info("Usage poll thread started — waiting for autologin...")
        _autologin_done.wait()
        log.info(f"Autologin done — starting usage polling (interval={_USAGE_POLL_INTERVAL}s)")
        try:
            with _usage_lock:
                result = _scrape_once()
                if result:
                    _usage_cache["usage"] = result
                    _usage_cache["timestamp"] = time.time()
                    log.info("Initial usage scrape OK")
        except Exception as e:
            log.exception(f"Initial usage scrape crashed: {e}")
        while True:
            time.sleep(_USAGE_POLL_INTERVAL)
            try:
                with _usage_lock:
                    result = _scrape_once()
                    if result:
                        _usage_cache["usage"] = result
                        _usage_cache["timestamp"] = time.time()
            except Exception as e:
                log.exception(f"Usage poll crashed: {e}")

    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()


# Placeholder for standalone scrape (set by _do_start_usage_poll)
_standalone_scrape_usage = None


def api_get_credentials(reason: str | None = None) -> tuple[int, dict]:
    """Get credentials (sync). Returns (http_status, response_dict)."""
    import json as json_mod

    from aii_lib.llm_backend.claude_max import aii_claude_dir

    init_credentials_state()
    log = logger.bind(source="claude-creds")

    creds_path = aii_claude_dir() / ".credentials.json"
    claude_cfg = _server_config.get("claude") or {}
    refresh_threshold_s = claude_cfg.get("token_refresh_threshold_seconds", 300)

    def _get_remaining() -> float:
        if not creds_path.exists():
            return 0.0
        try:
            creds = json_mod.loads(creds_path.read_text())
            oauth = creds.get("claudeAiOauth", {})
            expires_at_ms = oauth.get("expiresAt", 0)
            if not expires_at_ms or not oauth.get("accessToken"):
                return 0.0
            now_ms = int(time.time() * 1000)
            return max(0.0, (expires_at_ms - now_ms) / 1000.0)
        except Exception:
            return 0.0

    did_switch = False
    if reason == "rate_limited" and _account_manager and _account_manager.has_fallback:
        gen = _switch_generation
        did_switch = _switch_and_auth(gen)
        if did_switch:
            _invalidate_usage_cache()
        else:
            return 503, {"error": "Account switch failed after rate limit"}
    elif reason == "rate_limited":
        log.warning("Rate limit reported but no fallback accounts available")

    remaining = _get_remaining()

    # Refresh if needed
    if remaining < refresh_threshold_s and _account_manager:
        log.info(f"OAuth token expires in {remaining:.0f}s — refreshing...")
        with _refresh_lock:
            if _get_remaining() < refresh_threshold_s:
                if _account_manager and _account_manager.active:
                    from aii_lib.llm_backend.claude_max.autologin import (
                        ensure_oauth_token_for_account,
                    )

                    success = ensure_oauth_token_for_account(_account_manager.active, max_retries=2)
                    if success:
                        _account_manager.save_credentials_for_active()
                    elif _account_manager.has_fallback:
                        _switch_and_auth()
                    else:
                        return 503, {"error": "OAuth flow failed"}
        remaining = _get_remaining()

    # Read credentials
    with _refresh_lock:
        if not creds_path.exists():
            return 404, {"error": ".credentials.json not found"}
        creds = json_mod.loads(creds_path.read_text())

    active_email = (
        _account_manager.active.claude_email
        if _account_manager and _account_manager.active
        else None
    )
    result = {
        "credentials": creds,
        "remaining_seconds": remaining,
        "expires_in_human": f"{remaining / 3600:.1f}h"
        if remaining > 3600
        else f"{remaining / 60:.0f}m",
        "active_account": active_email,
    }
    if reason == "rate_limited":
        result["switched"] = did_switch
    _autologin_done.set()
    # Kick off usage polling eagerly so cache is warm by the time
    # /claude/usage is called (creating the tmux session takes ~45s).
    start_usage_polling()
    return 200, result


def api_get_usage() -> tuple[int, dict]:
    """Get cached usage (sync). Returns (http_status, response_dict).

    Waits up to 120s for the background poll thread to populate the cache.
    The poll thread (started by api_get_credentials or here) creates a tmux
    session with Claude CLI (~45s) then scrapes /usage (~5s). We wait for
    that instead of doing a competing standalone scrape that races for the
    same tmux session.
    """
    init_credentials_state()
    start_usage_polling()

    # Wait for background poll to populate cache (up to 120s).
    # The poll thread creates the tmux session + scrapes; doing a standalone
    # scrape here would race for the same tmux session and cause failures.
    for _ in range(120):
        cached = _usage_cache.get("usage")
        if cached:
            return 200, cached
        time.sleep(1)

    return 503, {"success": False, "error": "Usage not yet available (scrape timeout)"}


def api_get_accounts() -> tuple[int, dict]:
    """Get account list (sync). Returns (http_status, response_dict)."""
    init_credentials_state()

    if not _account_manager:
        return 200, {"accounts": [], "active_index": None}

    accounts = []
    for acc in _account_manager.accounts:
        remaining = acc.get_oauth_remaining_seconds()
        accounts.append(
            {
                "index": acc.index,
                "claude_email": acc.claude_email,
                "has_web_session": acc.web_session_path.exists(),
                "has_credentials": acc.credentials_path.exists(),
                "token_remaining_seconds": remaining,
                "token_remaining_human": (
                    f"{remaining / 3600:.1f}h" if remaining > 3600 else f"{remaining / 60:.0f}m"
                ),
            }
        )
    return 200, {
        "accounts": accounts,
        # 1-based to match ``accounts[*].index`` (display number).
        # Internally ``AccountManager._active_index`` is a 0-based list
        # position; expose +1 here so callers can compare against
        # ``accounts[i]["index"]`` directly.
        "active_index": _account_manager._active_index + 1,
        "active_email": _account_manager.active.claude_email if _account_manager.active else None,
        "switch_count": _account_manager._switch_count,
    }
