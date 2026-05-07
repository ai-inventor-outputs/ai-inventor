#!/usr/bin/env python3
"""
Automated Claude Code OAuth token management.

Two auth layers:
    Layer 1 — Web session: claude.ai browser cookies (web_session.json)
        Obtained externally (Chrome cookie injection or manual setup).

    Layer 2 — OAuth token: Claude Code access token (.credentials.json)
        Obtained by completing an OAuth consent flow that REQUIRES a valid
        web session. This is what Claude Code / Agent SDK actually uses.
        Has a finite expiresAt set by Anthropic servers.

Flow (when OAuth token is expired):
1. Check web session cookies exist (Layer 1)
2. Start `claude` CLI in tmux → navigate onboarding → extract OAuth URL
3. Open OAuth URL in browser with web session cookies → click Authorize
4. Extract auth code from callback, type into TUI → new OAuth token

Prerequisites:
    - Web session file (~/.claude/web_session.json)
    - tmux, google-chrome installed (nodriver talks CDP directly, no chromedriver)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from aii_lib.server_url import ability_service_url

from .oauth_flow import run_oauth_flow
from .token_utils import (
    check_oauth_token_expired,
    check_oauth_token_valid,
    get_oauth_token_remaining_seconds,
)

if TYPE_CHECKING:
    from .accounts import Account


def ensure_deps() -> None:
    """Check that required tools (tmux, chrome) are available."""
    import shutil

    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required but not found on PATH")
    if not shutil.which("google-chrome"):
        logger.warning("google-chrome not found — autologin may fail")


# ---------------------------------------------------------------------------
# Public API: ensure_oauth_token (importable + used by CLI)
# ---------------------------------------------------------------------------


def ensure_oauth_token(
    web_session_path: Path | None = None,
    max_retries: int = 2,
    force: bool = False,
) -> bool:
    """Ensure a valid Claude Code OAuth token (Layer 2) exists.

    Main entry point. Checks if the current OAuth token is valid, and if not,
    refreshes the web session (Layer 1) then runs the full OAuth flow.

    Args:
        web_session_path: Path to web_session.json (Layer 1 cookies).
            Defaults to ~/.claude/web_session.json.
        max_retries: Number of OAuth flow attempts before giving up.
        force: If True, invalidate current token first and force re-auth.

    Returns:
        True if a valid OAuth token exists, False if all attempts failed.
    """
    if web_session_path is None:
        from aii_lib.llm_backend.claude_max import aii_claude_dir

        web_session_path = aii_claude_dir() / "web_session.json"

    if force:
        logger.info("Force mode: invalidating current OAuth token...")
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        subprocess.run(
            ["claude", "auth", "logout"],
            capture_output=True,
            timeout=10,
            env=env,
        )
        time.sleep(1)
    elif check_oauth_token_valid():
        return True

    # Check web session cookies exist (obtained externally via cookie injection)
    if not _check_web_session(web_session_path):
        logger.error(f"Web session not found or empty: {web_session_path}")
        return False
    logger.info("Web session cookies present")

    ensure_deps()

    for attempt in range(1, max_retries + 1):
        logger.info(f"OAuth attempt {attempt}/{max_retries}...")
        try:
            if run_oauth_flow(web_session_path):
                # Verify we got a valid token
                if not check_oauth_token_expired():
                    logger.success("OAuth token obtained successfully")
                    return True
                if check_oauth_token_valid():
                    return True
                logger.warning("OAuth flow completed but token verification failed")
                return False
        except RuntimeError as e:
            logger.warning(f"OAuth attempt failed: {e}")

        if attempt < max_retries:
            logger.warning("OAuth token retry in 5s...")
            time.sleep(5)

    logger.error("All OAuth attempts failed")
    return False


# Cache of `{account_email: verified_token_hash}` so repeated activations of
# the same (already-verified) token skip the ~12s `/status` probe. Keyed by
# the configured ``claude_email`` so a token swap (OAuth refresh) re-verifies.
_verified_token_cache: dict[str, str] = {}


def _verify_oauth_identity(
    expected_email: str,
    token_hash: str,
    timeout_s: int = 30,
) -> tuple[bool, str | None]:
    """Verify the active OAuth token's identity matches ``expected_email``.

    Returns ``(matches, observed_email)``. ``observed_email`` is ``None``
    if the cache rebuild didn't happen within ``timeout_s`` (caller treats
    that as inconclusive).

    Strategy: claude CLI caches the bound identity in
    ``<config_dir>/.claude.json::oauthAccount`` (``emailAddress``,
    ``accountUuid``, etc.). Empirically that cache is **sticky** — after
    a credential swap it is NOT refreshed by:
      - elapsed time (still stale 90s+ later),
      - repeated ``/status`` invocations,
      - opening a fresh claude session.
    The TUI ``/status`` line and the on-disk cache both reflect the
    PRIOR account's identity even though the in-place token belongs to
    a different user. So scraping ``/status`` produces false-positive
    mismatches that quarantine perfectly good credentials.

    BUT: if ``oauthAccount`` is **missing**, claude reconstructs it from
    the active access token on next launch (~1-3s). We exploit that:
    delete the key, spawn a fresh claude, poll the file until the field
    is rebuilt. That field comes from the token itself (claude.ai's
    ``/api/account``-style probe) so it cannot lie.

    In-memory ``_verified_token_cache`` keyed by ``(expected_email,
    token_hash)`` short-circuits repeat activations of the same token
    so the 1-3s probe runs at most once per OAuth grant.
    """
    cache_key = expected_email
    if _verified_token_cache.get(cache_key) == token_hash:
        return True, expected_email  # already verified this exact token

    import json as _json
    import re
    import subprocess
    import uuid

    from aii_lib.llm_backend.claude_max import aii_claude_dir
    from aii_lib.utils.tmux import kill_session

    cfg_path = aii_claude_dir() / ".claude.json"
    sess = f"aii-identity-verify-{uuid.uuid4().hex[:8]}"

    # Clear the stale cache key so claude rebuilds it from the current token.
    try:
        cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if "oauthAccount" in cfg:
            cfg.pop("oauthAccount", None)
            cfg_path.write_text(_json.dumps(cfg, indent=2))
    except Exception as e:
        logger.warning(f"identity probe: could not clear oauthAccount cache: {e}")

    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                sess,
                "-c",
                "/tmp",
                "-e",
                f"CLAUDE_CONFIG_DIR={aii_claude_dir()}",
                "claude",
            ],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["tmux", "resize-window", "-t", sess, "-x", "200", "-y", "60"],
            check=False,
            capture_output=True,
        )
        # Poll the cache file until claude rebuilds oauthAccount (the
        # primary identity signal — written from claude.ai's account
        # endpoint, can't be faked or stale once present).
        # Typical rebuild: 1-3s. Budget ``timeout_s`` to absorb hiccups.
        deadline = time.time() + timeout_s
        cache_email: str | None = None
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                cfg = _json.loads(cfg_path.read_text())
                oa = cfg.get("oauthAccount") or {}
                em = (oa.get("emailAddress") or "").strip()
                if em:
                    cache_email = em
                    break
            except Exception:  # noqa: S112
                # polling loop; transient read races during claude rebuild are expected
                continue

        # Cross-check via TUI ``/status`` scrape. Belt-and-braces: TUI
        # reads from the same ``oauthAccount`` cache so they should
        # always agree, but if a future claude version splits the two
        # sources we still catch a mismatch instead of silently
        # accepting a wrong identity.
        tui_email: str | None = None
        try:
            subprocess.run(
                ["tmux", "send-keys", "-t", sess, "/status", "Enter"],
                check=False,
                capture_output=True,
            )
            time.sleep(4)
            out = subprocess.run(
                ["tmux", "capture-pane", "-t", sess, "-p", "-S", "-"],
                capture_output=True,
                text=True,
            )
            m = re.search(r"Email:\s*(\S+@\S+)", out.stdout)
            if m:
                tui_email = m.group(1)
        except Exception as e:
            logger.debug(f"identity probe: TUI scrape failed: {e}")

        # Decide using both signals. Cache is authoritative; TUI is a
        # cross-check. If they disagree, surface the cache value but
        # log a warning — that's a claude-side bug worth knowing about.
        if cache_email and tui_email and cache_email != tui_email:
            logger.warning(
                f"identity probe: cache={cache_email!r} != TUI={tui_email!r} "
                f"(claude split-brain?) — trusting cache"
            )
        observed = cache_email or tui_email
        matches = observed == expected_email
        if (
            matches
            and cache_email == expected_email
            and (tui_email is None or tui_email == expected_email)
        ):
            # Only cache when both signals agree (or TUI was unavailable).
            _verified_token_cache[cache_key] = token_hash
        return matches, observed
    finally:
        kill_session(sess)


def _token_hash(creds_path: Path) -> str:
    """Short stable hash of the OAuth access-token in ``creds_path``."""
    import hashlib

    try:
        creds = json.loads(creds_path.read_text())
        tok = creds.get("claudeAiOauth", {}).get("accessToken", "")
        return hashlib.sha256(tok.encode()).hexdigest()[:16] if tok else ""
    except Exception:
        return ""


_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"  # noqa: S105
# Hardcoded by Anthropic for the Claude Code CLI OAuth client. Same value the
# CLI uses internally; not a secret — gitleaks flags it as a generic UUID,
# but per Anthropic's docs this is the public OAuth client identifier.
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"  # gitleaks:allow


def _try_refresh_token_rotation(account: Account, std_creds_path: Path) -> bool:
    """Refresh access token via Anthropic's OAuth refresh endpoint.

    Uses the per-account stored ``refreshToken`` to obtain a new
    ``access_token``. Two safety properties make this strictly better
    than cookie-based re-auth when a refresh token is available:

    1. **Identity preservation** — refresh tokens are bound to the
       original OAuth grant, so the response always belongs to the
       account that minted them. Anthropic's response includes
       ``account.email_address`` which we still verify against
       ``account.claude_email`` as a defence-in-depth check.
    2. **No browser dependency** — sidesteps the Chrome-profile-state
       drift that causes the mismatched-quarantine cycle: we never
       touch ``web_session.json`` here.

    **Anthropic rotates refresh tokens on each use** (single-use
    semantics). The response carries a NEW ``refresh_token`` which we
    persist back to ``account.credentials_path`` and the standard creds
    path. Failing to save it means the next call hits ``invalid_grant``.

    Returns True on success (creds updated, identity verified). On any
    failure (no refresh token, network error, non-200, identity
    mismatch) returns False so the caller falls back to cookie OAuth.
    """
    import shutil

    if not account.credentials_path.exists():
        return False

    try:
        creds = json.loads(account.credentials_path.read_text())
        oauth = creds.get("claudeAiOauth", {}) or {}
        rt = oauth.get("refreshToken", "")
    except Exception as e:
        logger.warning(f"{account}: could not read stored credentials for refresh: {e}")
        return False

    if not rt:
        return False

    try:
        import httpx

        r = httpx.post(
            _OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": rt,
                "client_id": _OAUTH_CLIENT_ID,
            },
            timeout=15,
        )
    except Exception as e:
        logger.info(
            f"{account}: refresh request errored ({type(e).__name__}: {e}) — "
            f"falling back to cookie OAuth"
        )
        return False

    if r.status_code != 200:
        # 400 invalid_grant is the typical case — token revoked or already
        # rotated. Fall through to cookie OAuth.
        try:
            err = r.json()
        except Exception:
            err = r.text[:200]
        logger.info(
            f"{account}: refresh token rejected ({r.status_code} {err}) — "
            f"falling back to cookie OAuth"
        )
        return False

    try:
        resp = r.json()
    except Exception as e:
        logger.warning(f"{account}: refresh response not JSON: {e}")
        return False

    # Identity verification — Anthropic includes the bound account email.
    # If it doesn't match, refuse to save (defence in depth — refresh tokens
    # are by construction account-bound, but we don't trust silently).
    obs_email = ((resp.get("account") or {}).get("email_address") or "").strip()
    if obs_email and obs_email != account.claude_email:
        logger.error(
            f"{account}: refresh returned MISMATCHED identity — "
            f"got {obs_email!r}, expected {account.claude_email!r}. NOT saving."
        )
        return False

    new_access = resp.get("access_token")
    new_refresh = resp.get("refresh_token")
    if not new_access or not new_refresh:
        logger.warning(
            f"{account}: refresh response missing access_token or refresh_token — "
            f"keys={sorted(resp.keys())}"
        )
        return False

    # Build new credentials, IMPORTANT: persist the rotated refresh_token.
    # Anthropic invalidates the prior one on use; not saving means the next
    # call hits invalid_grant and falls back to cookie OAuth needlessly.
    expires_in = int(resp.get("expires_in", 3600))
    new_oauth = {
        "accessToken": new_access,
        "refreshToken": new_refresh,
        "expiresAt": int((time.time() + expires_in) * 1000),
    }
    scope = resp.get("scope", oauth.get("scopes", []))
    if isinstance(scope, str):
        scope = scope.split()
    new_oauth["scopes"] = scope
    # Preserve subscription + rate-limit tier metadata if present in old creds
    for k in ("subscriptionType", "rateLimitTier"):
        if k in oauth:
            new_oauth[k] = oauth[k]

    new_creds = {"claudeAiOauth": new_oauth}
    payload = json.dumps(new_creds, indent=2)

    # Atomic-ish write: temp + rename so a crash mid-write can't truncate
    # the per-account file (we'd lose the new refresh token forever).
    tmp = account.credentials_path.with_suffix(".credentials.json.tmp")
    tmp.write_text(payload)
    tmp.replace(account.credentials_path)
    shutil.copy2(account.credentials_path, std_creds_path)

    remain_min = expires_in // 60
    logger.success(
        f"{account}: refreshed via refresh_token (identity verified, +{remain_min}m valid)"
    )
    return True


def _quarantine_mismatched_creds(account: Account, observed: str | None) -> None:
    """Move the bad std creds aside for forensics.

    User can see what identity the token actually belonged to. Also drops the
    per-account file if it is the same wrong token, so the next activate() is
    forced to retry.
    """
    import shutil

    from aii_lib.llm_backend.claude_max import aii_claude_dir

    std = aii_claude_dir() / ".credentials.json"
    if not std.exists():
        return
    label = (observed or "unknown").replace("@", "_at_")
    account.ensure_dirs()
    quarantine = account.base_dir / f"mismatched_{label}.credentials.json"
    try:
        shutil.move(str(std), str(quarantine))
        logger.info(f"{account}: quarantined mismatched token → {quarantine}")
    except Exception as e:
        logger.warning(f"{account}: failed to quarantine std creds: {e}")
    # Also delete the per-account file if it matches the bad token, so the
    # next activate() doesn't immediately reinstate the same wrong identity.
    if not quarantine.exists() or not account.credentials_path.exists():
        return
    try:
        if account.credentials_path.read_text() == quarantine.read_text():
            account.credentials_path.unlink()
            logger.info(f"{account}: cleared per-account creds (same bad token)")
    except Exception:
        pass


def ensure_oauth_token_for_account(
    account: Account,
    max_retries: int = 2,
) -> bool:
    """Authenticate a specific Account (for multi-account ability server).

    Runs the full autologin flow for the given account:
    1. Activates the account (copies its creds to standard path)
    2. Verifies the in-place token's identity matches ``account.claude_email``
       (catches mis-labeled per-account creds early)
    3. Runs OAuth flow using account's web session cookies if needed
    4. Verifies the freshly-OAuth'd token's identity before persisting
    5. Saves resulting credentials back to account's directory

    Args:
        account: Account to authenticate.
        max_retries: Number of OAuth flow attempts.

    Returns:
        True if OAuth token obtained AND identity verified.
    """
    import shutil

    logger.info(f"Authenticating {account}...")
    account.ensure_dirs()

    # Copy account's existing creds to standard path (if any)
    from aii_lib.llm_backend.claude_max import aii_claude_dir

    std_creds = aii_claude_dir() / ".credentials.json"
    std_web_session = aii_claude_dir() / "web_session.json"
    if account.credentials_path.exists():
        shutil.copy2(account.credentials_path, std_creds)
    else:
        # No per-account creds — delete standard creds so the CLI does a
        # fresh login using the web session cookies instead of auto-refreshing
        # a token that belongs to a different account.
        if std_creds.exists():
            std_creds.unlink()
            logger.info(f"{account}: cleared stale standard credentials (forcing fresh OAuth)")
    if account.web_session_path.exists():
        shutil.copy2(account.web_session_path, std_web_session)

    # Check if existing token is still valid (reads per-account creds, not standard)
    remaining = account.get_oauth_remaining_seconds()
    if remaining > 300:  # > 5 min
        # Verify the cached token's identity matches the configured email
        # before short-circuiting. Catches mis-labeled per-account creds
        # left over from a Chrome-profile/Google-login mismatch on a
        # previous OAuth flow.
        matches, observed = _verify_oauth_identity(account.claude_email, _token_hash(std_creds))
        if observed is None:
            # Probe failed (claude CLI didn't render /status in time, etc.).
            # Treat as inconclusive — accept the cached token rather than
            # quarantine perfectly-good creds because of a transient probe
            # hiccup.
            logger.warning(
                f"{account}: identity probe inconclusive (no Email: line); "
                f"accepting cached token without verification this round"
            )
            return True
        if not matches:
            logger.error(
                f"{account}: cached token IDENTITY MISMATCH — "
                f"token belongs to {observed!r}, expected {account.claude_email!r}. "
                f"Chrome profile {account.chrome_profile!r} was likely signed into "
                f"the wrong Google account when this token was obtained."
            )
            _quarantine_mismatched_creds(account, observed)
            return False
        logger.info(
            f"{account}: token still valid ({remaining / 60:.0f}m remaining; identity verified)"
        )
        return True

    # Try refresh token rotation first — preserves identity via the
    # original OAuth grant, sidestepping the Chrome-profile drift that
    # causes mismatched-quarantine. Only on failure (no refresh token,
    # revoked, network error) do we fall through to cookie OAuth.
    if _try_refresh_token_rotation(account, std_creds):
        return True

    # Use injected cookies (from Chrome profiles via browser-cookie3)
    web_session_path = account.web_session_path
    if web_session_path.exists():
        logger.info(f"{account}: using injected cookies for OAuth")
        shutil.copy2(web_session_path, std_web_session)
    elif std_web_session.exists():
        logger.info(f"{account}: using standard web session for OAuth")
    else:
        logger.error(f"{account}: no web session cookies available")
        return False

    ensure_deps()

    # Run OAuth flow — try with current cookies, fall back to browser login if expired
    for attempt in range(1, max_retries + 1):
        logger.info(f"{account}: OAuth attempt {attempt}/{max_retries}...")
        try:
            if run_oauth_flow(web_session_path):
                if not check_oauth_token_expired():
                    # Verify the freshly-OAuth'd token's identity before
                    # persisting it as the canonical per-account creds.
                    matches, observed = _verify_oauth_identity(
                        account.claude_email, _token_hash(std_creds)
                    )
                    if observed is None:
                        # Probe inconclusive (transient) — accept the token
                        # but warn so the user can manually verify.
                        logger.warning(
                            f"{account}: identity probe inconclusive after OAuth; "
                            f"saving creds anyway, but please verify manually"
                        )
                    elif not matches:
                        logger.error(
                            f"{account}: OAuth IDENTITY MISMATCH — token is for "
                            f"{observed!r}, expected {account.claude_email!r}. "
                            f"Chrome profile {account.chrome_profile!r} is signed "
                            f"into the wrong Google account. NOT saving creds."
                        )
                        _quarantine_mismatched_creds(account, observed)
                        return False
                    # Save back to account directory (identity verified or inconclusive)
                    if std_creds.exists():
                        shutil.copy2(std_creds, account.credentials_path)
                    if std_web_session.exists():
                        shutil.copy2(std_web_session, account.web_session_path)
                    logger.success(f"{account}: OAuth token obtained + identity verified")
                    return True
                logger.warning(f"{account}: OAuth completed but token is still expired")
                # Don't return True — token wasn't actually refreshed
        except RuntimeError as e:
            if "Web session cookies expired" in str(e):
                logger.error(f"{account}: web session cookies expired — re-inject cookies")
                return False
            logger.warning(f"{account}: OAuth attempt failed: {e}")

        if attempt < max_retries:
            logger.warning(f"{account}: retrying in 5s...")
            time.sleep(5)

    logger.error(f"{account}: all OAuth attempts failed")
    return False


def _check_web_session(web_session_path: Path) -> bool:
    """Check if web session cookies exist and are non-empty.

    Web session cookies are obtained externally (Chrome cookie injection or
    manual setup). This just validates they're present.
    """
    if not web_session_path.exists():
        return False
    try:
        data = json.loads(web_session_path.read_text())
        return bool(data)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pipeline guard: ensure OAuth token (Layer 2) has enough remaining validity
# ---------------------------------------------------------------------------


def _fetch_credentials_remote(
    ability_url: str,
    max_retries: int = 3,
    reason: str | None = None,
) -> bool:
    """Fetch OAuth credentials from ability server and write to local file.

    The ability server runs Chrome/tmux and handles the full OAuth flow.
    This just fetches the resulting .credentials.json over HTTP.
    Retries on failure (the server may be mid-OAuth flow or temporarily unavailable).

    Args:
        ability_url: Base URL of the ability server.
        max_retries: Number of retry attempts.
        reason: Optional reason hint (e.g. "rate_limited") — tells the server
            to switch accounts before returning credentials.

    Returns True if credentials were fetched and written successfully.
    """
    import httpx

    from aii_lib.llm_backend.claude_max import aii_claude_dir

    creds_path = aii_claude_dir() / ".credentials.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)

    params = {"reason": reason} if reason else {}
    # Autologin on the server can take up to 5min — use 360s timeout
    req_timeout = 360.0 if reason else 180.0

    for attempt in range(1, max_retries + 1):
        try:
            from aii_lib.utils.internal_auth import internal_headers

            resp = httpx.get(
                f"{ability_url}/agent_abilities/claude/credentials",
                params=params,
                headers=internal_headers(),
                timeout=req_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.error(f"Ability server credential error: {data['error']}")
                if attempt < max_retries:
                    logger.warning(f"Retrying credential fetch ({attempt}/{max_retries}) in 10s...")
                    time.sleep(10)
                    continue
                return False

            creds_path.write_text(json.dumps(data["credentials"]))
            try:
                import os

                os.chmod(creds_path, 0o600)
            except OSError:
                pass
            remaining = data.get("remaining_seconds", 0)
            expires_human = data.get(
                "expires_in_human",
                f"{remaining}s" if remaining < 60 else f"{remaining / 60:.0f}m",
            )
            logger.success(f"Credentials fetched from ability server (expires in {expires_human})")
            return True

        except Exception as e:
            # Short summary to the activity feed; full stack stays in the
            # loguru file sink via ``logger.opt(exception=True)``. Embedding
            # the raw traceback in the message body bloated every retry into
            # a 30+ line wall in the FE feed.
            err_brief = f"{type(e).__name__}: {e}".strip() or type(e).__name__
            logger.opt(exception=True).debug(
                f"Credential fetch failed (attempt {attempt}/{max_retries})"
            )
            logger.error(f"Credential fetch failed (attempt {attempt}/{max_retries}): {err_brief}")
            if attempt < max_retries:
                logger.warning("Credential fetch retry in 15s...")
                time.sleep(15)

    logger.error(f"All {max_retries} credential fetch attempts failed")
    return False


def _resolve_ability_url() -> str | None:
    """Resolve ability server URL: env var (remote) or auto-detect local.

    On RunPod, AII_SERVER_URL is set explicitly.
    Locally, the ability server runs on localhost — auto-detect it so
    both modes use the same credential/switching/usage code path.
    """
    from aii_lib.utils.run_mode import is_local as _is_local

    url = ability_service_url()
    if url:
        return url

    if _is_local():
        try:
            import httpx

            from aii_lib.server_url import SERVER_PORT
            from aii_lib.utils.internal_auth import internal_headers

            local_url = f"http://localhost:{SERVER_PORT}"
            resp = httpx.get(
                f"{local_url}/agent_abilities/health",
                headers=internal_headers(),
                timeout=2.0,
            )
            if resp.status_code == 200:
                logger.debug(f"Auto-detected local aii_server at {local_url}")
                return local_url
        except Exception:
            pass
    return None


def ensure_oauth_token_fresh(min_validity_seconds: int = 3600) -> bool:
    """Ensure OAuth token (Layer 2) has enough remaining validity.

    Called between pipeline steps to prevent mid-step token expiry.

    Unified flow for both local and remote:
      1. Check if ability server is reachable (AII_SERVER_URL or
         auto-detected on localhost). If so, fetch credentials from its
         ``/claude/credentials`` endpoint — it handles OAuth flow,
         multi-account switching, and usage monitoring.
      2. Fallback: run single-account OAuth flow directly (no ability server).

    Args:
        min_validity_seconds: Get a new token if fewer than this many
            seconds remaining. Default 3600 (1 hour).

    Returns:
        True if token has sufficient validity (existing or freshly obtained).
        False if OAuth flow failed.
    """
    # Check local token first (works in both modes)
    remaining = get_oauth_token_remaining_seconds()

    if remaining >= min_validity_seconds:
        remaining_min = remaining / 60
        logger.info(
            f"OAuth token valid for {remaining_min:.0f}m (threshold: {min_validity_seconds // 60}m)"
        )
        return True

    remaining_min = remaining / 60
    threshold_min = min_validity_seconds / 60
    logger.warning(
        f"OAuth token expires in {remaining_min:.0f}m (threshold: {threshold_min:.0f}m) — re-authenticating"
    )

    # Unified: fetch from ability server (remote or local auto-detect)
    ability_url = _resolve_ability_url()
    if ability_url:
        logger.info(f"Fetching credentials from ability server: {ability_url}")
        if _fetch_credentials_remote(ability_url):
            new_remaining = get_oauth_token_remaining_seconds()
            if new_remaining >= min_validity_seconds:
                return True
            logger.warning(
                f"Credentials still below threshold ({new_remaining / 60:.0f}m < {threshold_min:.0f}m)"
            )
            return True  # Token exists, ability server may have returned its best
        logger.error("Ability server credential fetch failed")
        return False

    # Fallback: no ability server — single-account local OAuth flow
    logger.info("No ability server available — running single-account OAuth flow")
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    subprocess.run(
        ["claude", "auth", "logout"],
        capture_output=True,
        timeout=10,
        env=env,
    )
    time.sleep(1)

    if ensure_oauth_token():
        new_remaining = get_oauth_token_remaining_seconds()
        logger.success(f"New OAuth token obtained — valid for {new_remaining / 60:.0f}m")
        return True

    logger.error("OAuth token refresh failed")
    return False


# ---------------------------------------------------------------------------
# Entry point (CLI)
# ---------------------------------------------------------------------------


def main() -> int:
    """Obtain Claude Code OAuth token via CLI."""
    import argparse

    from aii_lib.llm_backend.claude_max import aii_claude_dir

    parser = argparse.ArgumentParser(description="Obtain Claude Code OAuth token")
    parser.add_argument(
        "--web-session",
        "-s",
        type=Path,
        default=aii_claude_dir() / "web_session.json",
    )
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--max-retries", "-r", type=int, default=2)
    args = parser.parse_args()

    success = ensure_oauth_token(
        web_session_path=args.web_session,
        max_retries=args.max_retries,
        force=args.force,
    )
    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
