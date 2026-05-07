"""Multi-account management for Claude Code autologin.

Accounts come from ``claude_max.account_priority`` in
``aii_config/pipeline/harness/llm_backend.yaml``, with ``llm_backend.private.yaml``
(gitignored sibling) overriding any keys it sets. The tracked file ships
with an empty list, so accounts are local-machine-specific by default.
Order determines priority: first = primary, rest = fallbacks.

Each account gets its own directory under ``<claude-dir>/accounts/<key>/``
with independent ``web_session.json`` and ``.credentials.json`` files.
``<key>`` is the per-account ``cookie_dir`` from the YAML if set, else
the account's email (``@`` becomes ``_at_`` so it survives shell tooling).
Email-keyed so reordering or commenting accounts in the config never
re-points an entry at another account's stored creds.

The AccountManager tracks which account is active and handles switching
when the ability server detects usage threshold exceeded.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class Account:
    """A Claude account with its credentials and per-account storage paths."""

    index: int
    claude_email: str
    chrome_profile: str = ""
    # Optional explicit subdir name under ``<claude-dir>/accounts/`` for this
    # account's stored ``web_session.json`` + ``.credentials.json``. When unset
    # we derive it from ``claude_email`` so reordering/commenting accounts
    # in the YAML can't re-point an entry at another account's stored creds.
    cookie_dir: str = ""

    @property
    def base_dir(self) -> Path:
        """Per-account directory under ``<claude-dir>/accounts/``.

        Key = explicit ``cookie_dir`` if set, else ``claude_email`` with
        ``@`` swapped to ``_at_`` (filesystem-safe, still readable).
        """
        from aii_lib.llm_backend.claude_max import aii_claude_dir

        sub = self.cookie_dir or self.claude_email.replace("@", "_at_")
        return aii_claude_dir() / "accounts" / sub

    @property
    def web_session_path(self) -> Path:
        """Per-account web session cookies (Layer 1)."""
        return self.base_dir / "web_session.json"

    @property
    def credentials_path(self) -> Path:
        """Per-account OAuth credentials (Layer 2)."""
        return self.base_dir / ".credentials.json"

    def ensure_dirs(self) -> None:
        """Create account directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_oauth_remaining_seconds(self) -> float:
        """Get seconds until this account's OAuth token expires."""
        if not self.credentials_path.exists():
            return 0.0
        try:
            creds = json.loads(self.credentials_path.read_text())
            oauth = creds.get("claudeAiOauth", {})
            expires_at_ms = oauth.get("expiresAt", 0)
            if not expires_at_ms or not oauth.get("accessToken"):
                return 0.0
            now_ms = int(time.time() * 1000)
            return max(0.0, (expires_at_ms - now_ms) / 1000.0)
        except Exception:
            return 0.0

    def __str__(self) -> str:
        return f"Account#{self.index}({self.claude_email})"


# Standard paths the claude CLI reads/writes — resolved lazily so the env
# var is honoured even when imported before aii_launcher sets CLAUDE_CONFIG_DIR.
def _std_creds_path() -> Path:
    from aii_lib.llm_backend.claude_max import aii_claude_dir

    return aii_claude_dir() / ".credentials.json"


def _std_web_session_path() -> Path:
    from aii_lib.llm_backend.claude_max import aii_claude_dir

    return aii_claude_dir() / "web_session.json"


class AccountManager:
    """Manages multiple Claude accounts and switching between them.

    Loads accounts from numbered env vars at init time.
    Tracks the active account index and handles:
    - Activating an account (copying its creds to the standard path)
    - Switching to the next account when usage threshold exceeded
    - Saving OAuth results back to account-specific paths
    """

    def __init__(self) -> None:
        self.accounts: list[Account] = self._load_accounts()
        self._active_index: int = 0
        self._lock = threading.Lock()
        self._switch_count: int = 0

        if not self.accounts:
            logger.warning(
                "No Claude accounts configured "
                "(aii_config/pipeline/harness/llm_backend.private.yaml)"
            )
        else:
            logger.info(
                f"Loaded {len(self.accounts)} account(s): "
                + ", ".join(str(a) for a in self.accounts)
            )
            # Ensure directories exist
            for account in self.accounts:
                account.ensure_dirs()

    @staticmethod
    def _load_accounts() -> list[Account]:
        """Load account priority from harness/llm_backend.yaml.

        Reads ``claude_max.account_priority`` from
        ``aii_config/pipeline/harness/llm_backend.yaml``, with
        ``llm_backend.private.yaml`` (gitignored sibling) deep-merged on top.

        Each entry can be either a plain email string or a dict with
        ``email`` + optional ``chrome_profile`` + optional ``cookie_dir``
        (explicit subdir under ``<claude-dir>/accounts/``).
        """
        try:
            from aii_lib.utils.config_overrides import load_config_with_overrides

            # Walk up to the repo root (marker: ``aii_config/`` dir).
            def _find_repo_root() -> Path | None:
                here = Path(__file__).resolve()
                for parent in here.parents:
                    if (parent / "aii_config").is_dir():
                        return parent
                return None

            roots = [Path("/ai-inventor")]
            if (r := _find_repo_root()) is not None:
                roots.append(r)

            entries: list = []
            for root in roots:
                cfg = load_config_with_overrides(
                    root / "aii_config" / "pipeline" / "harness" / "llm_backend.yaml"
                )
                cand = cfg.get("claude_max", {}).get("account_priority")
                if cand:
                    entries = cand
                    break

            return [
                Account(
                    index=i + 1,
                    claude_email=e["email"] if isinstance(e, dict) else e,
                    chrome_profile=e.get("chrome_profile", "") if isinstance(e, dict) else "",
                    cookie_dir=e.get("cookie_dir", "") if isinstance(e, dict) else "",
                )
                for i, e in enumerate(entries)
            ]
        except Exception as e:
            logger.debug(f"Could not load Claude account list: {e}")
            return []

    @property
    def active(self) -> Account | None:
        """Currently active account."""
        if not self.accounts:
            return None
        return self.accounts[self._active_index]

    @property
    def has_fallback(self) -> bool:
        """Always true with 2+ accounts (round-robin)."""
        return len(self.accounts) > 1

    def activate(self, account: Account) -> None:
        """Make this account active: copy its credentials to the standard path.

        The claude CLI always reads from ~/.claude/.credentials.json,
        so we copy the account's creds there.
        """
        std_creds = _std_creds_path()
        std_creds.parent.mkdir(parents=True, exist_ok=True)

        if account.credentials_path.exists():
            shutil.copy2(account.credentials_path, std_creds)
            remaining = account.get_oauth_remaining_seconds()
            logger.info(f"Activated {account} — token valid for {remaining / 60:.0f}m")
        else:
            # No per-account creds — clear standard path so credential
            # endpoint forces a fresh OAuth for this account.
            if std_creds.exists():
                std_creds.unlink()
            logger.info(f"Activated {account} — no existing credentials (needs autologin)")

        # Also copy web session if it exists
        if account.web_session_path.exists():
            shutil.copy2(account.web_session_path, _std_web_session_path())

    def save_credentials_for_active(self) -> None:
        """Save current standard credentials back to the active account's directory.

        Called after an OAuth flow writes to ~/.claude/.credentials.json —
        we copy it to the active account's per-account path for later reuse.
        """
        account = self.active
        if not account:
            return
        std_creds = _std_creds_path()
        std_web = _std_web_session_path()
        if std_creds.exists():
            shutil.copy2(std_creds, account.credentials_path)
        if std_web.exists():
            shutil.copy2(std_web, account.web_session_path)

    def switch_to_next(self) -> Account | None:
        """Switch to the next account (round-robin). Returns new active, or None if only one account.

        Thread-safe. Kills the persistent usage tmux session so it restarts
        with the new account's credentials on next usage check.
        """
        with self._lock:
            if not self.has_fallback:
                logger.warning("Only one account configured — cannot switch")
                return None

            old = self.active
            self._active_index = (self._active_index + 1) % len(self.accounts)
            new = self.active
            self._switch_count += 1

            logger.warning(f"Switching account: {old} -> {new} (switch #{self._switch_count})")

            # Activate new account (copy creds to standard path)
            self.activate(new)

            # Kill persistent usage tmux session so it restarts with new creds
            self._kill_usage_session()

            return new

    @staticmethod
    def _kill_usage_session() -> None:
        """Kill the persistent usage tmux session so it restarts with new creds."""
        from aii_lib.utils.tmux import kill_session

        kill_session("claude_usage_persistent")
        logger.debug("Killed persistent usage session for account switch")
