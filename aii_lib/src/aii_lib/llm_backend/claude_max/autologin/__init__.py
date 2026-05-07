"""Automated Claude Code OAuth token management.

Two auth layers:
    Layer 1 — Web session: claude.ai browser cookies (web_session.json)
    Layer 2 — OAuth token: Claude Code access token (.credentials.json)

Multi-account support:
    AccountManager loads numbered accounts from env vars and handles
    automatic switching when usage thresholds are exceeded.

Usage:
    from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token
    success = ensure_oauth_token()

    from aii_lib.llm_backend.claude_max.autologin import Account, AccountManager
    mgr = AccountManager()

CLI:
    python -m aii_lib.llm_backend.claude_max.autologin
    python -m aii_lib.llm_backend.claude_max.autologin --force
"""

from .accounts import Account, AccountManager
from .autologin import (
    _fetch_credentials_remote,
    check_oauth_token_expired,
    check_oauth_token_valid,
    ensure_deps,
    ensure_oauth_token,
    ensure_oauth_token_for_account,
    ensure_oauth_token_fresh,
    get_oauth_token_remaining_seconds,
    run_oauth_flow,
)

__all__ = [
    "Account",
    "AccountManager",
    "_fetch_credentials_remote",
    "check_oauth_token_expired",
    "check_oauth_token_valid",
    "ensure_deps",
    "ensure_oauth_token",
    "ensure_oauth_token_for_account",
    "ensure_oauth_token_fresh",
    "get_oauth_token_remaining_seconds",
    "run_oauth_flow",
]
