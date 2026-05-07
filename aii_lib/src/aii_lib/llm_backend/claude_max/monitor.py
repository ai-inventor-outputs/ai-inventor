"""Background usage monitor for Claude Code rate limiting.

Two modes:
  - **Remote** (AII_SERVER_URL set): fetches usage from the ability
    server's ``/claude/usage`` endpoint.  No local Chrome/tmux needed.
  - **Local** (default): runs ``get_claude_usage()`` which interacts with
    the ``claude`` CLI in a tmux session.
"""

import asyncio
import json
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from aii_lib.llm_backend.claude_max.usage import ClaudeUsage, get_claude_usage

# Config file paths (priority order: global config > local config > defaults)
PIPELINE_CONFIG_PATH = (
    Path(__file__).parents[6] / "aii_config" / "pipeline" / "harness" / "llm_backend.yaml"
)
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _short_err(e: Exception) -> str:
    """One-liner for hot-path warning messages — no traceback dump.

    Same intent as ``workflows.summarize._summarize_provider_error``
    but local to the monitor so we don't pull a workflow import here.
    """
    s = str(e).strip()
    cls = type(e).__name__
    if cls in (
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
    ):
        return f"timed out ({cls})"
    if "429" in s:
        return "rate-limited (429)"
    if "Connection refused" in s:
        return "connection refused (server down?)"
    first_line = s.split("\n", 1)[0][:140]
    return first_line if first_line else cls


class UsageMonitor:
    """
    Background monitor for Claude Code usage.

    Checks usage periodically and blocks calls when usage exceeds threshold.
    """

    _instance: "UsageMonitor | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "UsageMonitor":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self._config = self._load_config()
        self._latest_usage: ClaudeUsage | None = None
        self._is_rate_limited = threading.Event()
        self._rate_limit_start: float | None = None
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._telemetry_file: Path | None = None
        self._consecutive_failures: int = 0

        # Remote mode: fetch from ability server instead of local tmux.
        # Deferred import: server_url.py partially-initializes when imported
        # via aii_lib.utils.config_overrides → utils/__init__ → agent_to_llm
        # → agent_backend.claude.agent → here. Hoisting this to module-level
        # re-enters server_url before ability_service_url is bound.
        from aii_lib.server_url import ability_service_url

        self._remote_url: str = ability_service_url()
        self._server_claude_account: str | None = None  # track active account on server

        if self._config["telemetry"]["enabled"]:
            self._telemetry_file = Path(self._config["telemetry"]["log_file"])

    def _load_config(self) -> dict:
        """Load configuration from yaml file.

        Priority: aii_config/pipeline/harness/llm_backend.yaml > local config.yaml > defaults
        Reads ``claude_max`` (the only llm_backend that has Claude Max usage
        tracking + telemetry — openrouter doesn't gate on Anthropic plan quotas).
        """
        # Try pipeline config first.
        # Use the deep-merge loader so ``llm_backend.private.yaml`` overrides
        # land — otherwise threshold tweaks in the private file get ignored.
        from aii_lib.utils.config_overrides import load_config_with_overrides

        if PIPELINE_CONFIG_PATH.exists():
            pipeline_config = load_config_with_overrides(PIPELINE_CONFIG_PATH)
            section = pipeline_config.get("claude_max")
            if isinstance(section, dict):
                return section

        # Fall back to local config
        if LOCAL_CONFIG_PATH.exists():
            return load_config_with_overrides(LOCAL_CONFIG_PATH)

        # Default config
        return {
            "usage_tracking": {
                "enabled": True,
                "check_interval_seconds": 660,
                "thresholds": {
                    "current_session": 70,
                    "current_week_all_models": 90,
                    "current_week_sonnet": 90,
                },
            },
            "telemetry": {
                "enabled": True,
                "log_file": "/tmp/claude_usage_telemetry.jsonl",
            },
        }

    def _log_telemetry(self, usage: ClaudeUsage, is_rate_limited: bool) -> None:
        """Log usage data to telemetry file."""
        if not self._telemetry_file:
            return

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "usage": asdict(usage),
            "is_rate_limited": is_rate_limited,
            "thresholds": self._config["usage_tracking"]["thresholds"],
        }

        with open(self._telemetry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _check_threshold(self, usage: ClaudeUsage) -> bool:
        """Check if any monitored metric exceeds its threshold (no logging here).

        Each metric has its own threshold. Set to null in config to disable.
        """
        thresholds = self._config["usage_tracking"]["thresholds"]

        session_threshold = thresholds.get("current_session")
        if session_threshold is not None and usage.current_session is not None:
            if usage.current_session >= session_threshold:
                return True

        all_models_threshold = thresholds.get("current_week_all_models")
        if all_models_threshold is not None and usage.current_week_all_models is not None:
            if usage.current_week_all_models >= all_models_threshold:
                return True

        sonnet_threshold = thresholds.get("current_week_sonnet")
        if sonnet_threshold is not None and usage.current_week_sonnet is not None:
            if usage.current_week_sonnet >= sonnet_threshold:
                return True

        return False

    def _fetch_usage_remote(self) -> tuple[ClaudeUsage, bool, dict, str | None]:
        """Fetch usage from ability server's /claude/usage endpoint.

        Retries on transient errors (connection drops, proxy hiccups).

        Returns:
            (usage, over_threshold, thresholds, active_account) — server is
            the authority for threshold decisions and account switching.
        """
        import httpx
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        from aii_lib.utils.retry import make_retry_log

        @retry(
            retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=15),
            before_sleep=make_retry_log(label="claude monitor"),
            reraise=True,
        )
        def _fetch() -> dict:
            from aii_lib.utils.internal_auth import internal_headers

            resp = httpx.get(
                f"{self._remote_url}/agent_abilities/claude/usage",
                headers=internal_headers(),
                # 120s matches api_get_usage's max wait for cache population.
                # After an account switch the cache is invalidated, and the
                # next scrape can take ~45s to rebuild the persistent claude
                # tmux session. 60s wasn't enough — re-check timed out every
                # time and triggered the assume-success fallback.
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()

        data = _fetch()
        if not data.get("success"):
            # Rate-limited: the usage API itself is unavailable — not an error worth
            # counting as a failure. Return unknown usage (won't trigger threshold).
            if data.get("rate_limited"):
                logger.debug("Usage API rate-limited — treating as unknown (no threshold action)")
                return ClaudeUsage(), False, {}, None
            raise RuntimeError(data.get("error", "Unknown error from ability server"))

        u = data["usage"]
        usage = ClaudeUsage(
            current_session=u.get("current_session"),
            current_week_all_models=u.get("current_week_all_models"),
            current_week_sonnet=u.get("current_week_sonnet"),
        )
        return (
            usage,
            data.get("over_threshold", False),
            data.get("thresholds", {}),
            data.get("active_account"),
        )

    def _refresh_credentials_remote(self) -> None:
        """Fetch fresh credentials from ability server (e.g. after account switch)."""
        try:
            from aii_lib.llm_backend.claude_max.autologin import _fetch_credentials_remote

            _fetch_credentials_remote(self._remote_url)
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {_short_err(e)}")

    def _request_account_switch_remote(self) -> bool:
        """Ask the ability server to switch to the next account.

        Returns True if the switch succeeded and new credentials were applied.
        Timeout is generous (5min) because the server runs a full OAuth flow
        (Xvfb + browser + Cloudflare + button click) for the new account.

        Retries on transient transport failures (5xx from the RunPod proxy,
        connect/read timeouts) so a single proxy hiccup doesn't collapse
        rotation into the false "no more accounts available" path. The
        transient-status set mirrors ``ability_client._TRANSIENT_STATUS_CODES``.
        """
        if not self._remote_url:
            return False
        import httpx

        from aii_lib.utils.internal_auth import internal_headers

        url = f"{self._remote_url}/agent_abilities/claude/credentials"
        # Three tries, ~6s of total backoff on the transport — well under
        # the 300s server-side OAuth budget so retries can't outlast the
        # actual flow time. 500 is included because in practice it fires
        # when an aii-server uvicorn worker crashes mid-OAuth-flow (Xvfb
        # / browser / Cloudflare hiccup) and respawns — retrying succeeds.
        # Real permanent server bugs would still chew the full 3 attempts
        # but they're rare and the alternative (collapse rotation on first
        # 500) is strictly worse.
        transient_status = {429, 500, 502, 503, 504, 524}
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = httpx.get(
                    url,
                    params={"reason": "rate_limited"},
                    headers=internal_headers(),
                    timeout=300.0,
                )
                if resp.status_code in transient_status:
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                data = resp.json()
                if data.get("switched"):
                    new_account = data.get("active_account", "?")
                    logger.info(f"Account switched to {new_account} (proactive threshold switch)")
                    return True
                logger.warning(
                    "Account switch requested but server did not switch (no fallback available?)"
                )
                return False
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_err = e
            except httpx.HTTPStatusError as e:
                if e.response.status_code in transient_status:
                    last_err = e
                else:
                    logger.error(f"Failed to request account switch: {_short_err(e)}")
                    return False
            except Exception as e:
                logger.error(f"Failed to request account switch: {_short_err(e)}")
                return False
            if attempt < 3:
                time.sleep(2 * attempt)  # 2s, 4s
        logger.error(
            f"Failed to request account switch after 3 attempts (transient): {_short_err(last_err) if last_err else 'unknown'}"
        )
        return False

    def _fmt_usage(self, usage: ClaudeUsage, thresholds: dict) -> str:
        """Format usage as ``session: 75%/70% (OVER) | all_models: 8%/90% | sonnet: 0%/95%``.

        Server returns a single ``is_over_threshold`` bool — without this
        annotation the logs don't show *which* metric tripped (e.g.
        ``all_models: 7%`` looks fine but ``session: 75%`` was the
        actual trigger). The ``(OVER)`` marker is appended after each
        metric whose value meets/exceeds its threshold.
        """
        parts = []
        for label, val, thr_key in (
            ("session", usage.current_session, "current_session"),
            ("all_models", usage.current_week_all_models, "current_week_all_models"),
            ("sonnet", usage.current_week_sonnet, "current_week_sonnet"),
        ):
            thr = thresholds.get(thr_key)
            val_str = f"{val}%" if val is not None else "?%"
            thr_str = f"{thr}%" if thr is not None else "-"
            marker = " (OVER)" if val is not None and thr is not None and val >= thr else ""
            parts.append(f"{label}: {val_str}/{thr_str}{marker}")
        return " | ".join(parts)

    def _monitor_loop(self) -> None:
        """Background monitoring loop.

        Remote mode: ability server is the single source of truth for
        thresholds and account switching. We just read ``over_threshold``
        from the response and display the server-provided thresholds.

        Local mode: uses local config thresholds (no account switching).
        """
        interval = self._config["usage_tracking"]["check_interval_seconds"]
        local_thresholds = self._config["usage_tracking"]["thresholds"]

        if self._remote_url:
            logger.info(
                f"Usage monitor: remote mode ({self._remote_url}/agent_abilities/claude/usage)"
            )
        else:
            logger.info("Usage monitor: local mode (tmux claude CLI)")

        while not self._stop_event.is_set():
            try:
                # Fetch usage — remote returns server-side threshold decision
                if self._remote_url:
                    usage, is_over_threshold, thresholds, active_account = (
                        self._fetch_usage_remote()
                    )

                    # Detect account switch by another worker's monitor,
                    # or sync on first poll (server may have switched before
                    # this worker started).
                    if active_account:
                        if self._server_claude_account is None:
                            # First poll — sync credentials to match server
                            logger.info(
                                f"Initial credential sync — server account: {active_account}"
                            )
                            self._refresh_credentials_remote()
                        elif active_account != self._server_claude_account:
                            logger.info(
                                f"Server account changed: {self._server_claude_account} → {active_account} "
                                f"— syncing credentials"
                            )
                            self._refresh_credentials_remote()
                        self._server_claude_account = active_account
                else:
                    usage = get_claude_usage()
                    is_over_threshold = self._check_threshold(usage)
                    thresholds = local_thresholds

                self._latest_usage = usage
                self._consecutive_failures = 0
                was_rate_limited = self._is_rate_limited.is_set()

                self._log_telemetry(usage, is_over_threshold)

                if is_over_threshold:
                    if not was_rate_limited:
                        self._is_rate_limited.set()
                        self._rate_limit_start = time.time()

                    # Proactive account switch: keep rotating until we find
                    # an account below threshold or exhaust all accounts.
                    # Runs on every poll (not just first detection) since
                    # account usage may have decreased since last check.
                    if self._remote_url:
                        logger.info(
                            f"Usage over threshold — switching from "
                            f"{self._server_claude_account or '?'} | "
                            + self._fmt_usage(usage, thresholds)
                        )
                        first_account = self._server_claude_account
                        found_below = False
                        attempt = 0
                        while not self._stop_event.is_set():
                            attempt += 1
                            switched = self._request_account_switch_remote()
                            if not switched:
                                # Either the server explicitly reported no
                                # fallback or the request failed transiently
                                # past its retry budget — the function logged
                                # the precise reason already. Log a generic
                                # message here, not "no more accounts" (the
                                # former wording masked transient failures).
                                logger.warning(
                                    "Account switch unavailable — falling back to wait-for-cooldown"
                                )
                                break
                            try:
                                usage, is_over_threshold, thresholds, active_account = (
                                    self._fetch_usage_remote()
                                )
                                self._server_claude_account = active_account
                                self._latest_usage = usage
                                self._log_telemetry(usage, is_over_threshold)
                                if not is_over_threshold:
                                    logger.success(
                                        f"Switched to {active_account} (try #{attempt}) — "
                                        f"below threshold, resuming | "
                                        + self._fmt_usage(usage, thresholds)
                                    )
                                    self._refresh_credentials_remote()
                                    self._is_rate_limited.clear()
                                    self._rate_limit_start = None
                                    found_below = True
                                    break
                                logger.info(
                                    f"Switched to {active_account} (try #{attempt}) — "
                                    f"still over threshold | " + self._fmt_usage(usage, thresholds)
                                )
                                # Cycled back to starting account — all exhausted
                                if active_account and active_account == first_account:
                                    next_check_min = max(1, int(interval / 60))
                                    logger.warning(
                                        f"All {attempt} accounts over threshold — "
                                        f"waiting for cooldown (next check in ~{next_check_min}min)"
                                    )
                                    break
                            except Exception as e:
                                logger.warning(f"Re-check after switch failed: {_short_err(e)}")
                                # Switch succeeded but re-check failed — assume new account is OK
                                logger.info(
                                    "Assuming new account is below threshold (re-check unavailable)"
                                )
                                self._refresh_credentials_remote()
                                self._is_rate_limited.clear()
                                self._rate_limit_start = None
                                found_below = True
                                break
                        if found_below:
                            self._stop_event.wait(interval)
                            continue

                    elapsed_min = (
                        int((time.time() - self._rate_limit_start) / 60)
                        if self._rate_limit_start
                        else 0
                    )
                    next_check_min = max(1, int(interval / 60))
                    logger.warning(
                        f"Usage over threshold — waited {elapsed_min}min, "
                        f"next check in ~{next_check_min}min | "
                        + self._fmt_usage(usage, thresholds)
                    )
                else:
                    if was_rate_limited:
                        elapsed_min = (
                            int((time.time() - self._rate_limit_start) / 60)
                            if self._rate_limit_start
                            else 0
                        )
                        logger.success(
                            f"Usage below thresholds — resuming (waited {elapsed_min}min)"
                        )
                        # Fetch fresh credentials (account may have switched)
                        if self._remote_url:
                            self._refresh_credentials_remote()
                        self._is_rate_limited.clear()
                        self._rate_limit_start = None

            except Exception as e:
                self._consecutive_failures += 1
                backoff = max(600, min(self._consecutive_failures * interval, 1800))
                err_msg = str(e).split("\n")[0]
                _pending_warn = (
                    f"Usage monitor: check failed ({self._consecutive_failures}): {err_msg}"
                    + (f" — backing off {backoff}s" if self._consecutive_failures > 1 else "")
                )
            else:
                _pending_warn = None

            # Emit OUTSIDE except block so _enrich_with_traceback won't append traceback
            if _pending_warn:
                logger.warning(_pending_warn)
                self._stop_event.wait(backoff)
                continue

            self._stop_event.wait(interval)

    def start(self) -> None:
        """Start the background monitor.

        Performs first usage check synchronously so wait_for_capacity
        never has to poll for the initial result.
        """
        if not self._config["usage_tracking"]["enabled"]:
            return

        if self._monitor_thread and self._monitor_thread.is_alive():
            return  # Already running, no need to log

        # First check synchronously — blocks but guarantees _latest_usage
        # is set (or clearly failed) before any task proceeds.
        if self._latest_usage is None:
            try:
                if self._remote_url:
                    usage, is_over, _, _ = self._fetch_usage_remote()
                else:
                    usage = get_claude_usage()
                    is_over = self._check_threshold(usage)
                self._latest_usage = usage
                self._log_telemetry(usage, is_over)
                if is_over:
                    self._is_rate_limited.set()
            except Exception as e:
                logger.warning(f"Usage monitor first check failed: {str(e).split(chr(10))[0]}")

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the background monitor."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def wait_for_capacity(self, timeout: float | None = None) -> bool:
        """
        Block until usage is below threshold.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if capacity available, False if timed out.
        """
        # First check runs in start(), so _latest_usage should already be set.
        # Brief wait as safety net in case background thread hasn't persisted yet.
        if self._latest_usage is None:
            for _ in range(10):
                if self._latest_usage is not None:
                    break
                time.sleep(1)
            if self._latest_usage is None:
                logger.warning("Usage monitor has no data — continuing without monitoring")
                return True

        if not self._is_rate_limited.is_set():
            return True

        # Wait for rate limit to clear (monitor loop logs status)
        start = time.time()
        while self._is_rate_limited.is_set():
            if timeout is not None and (time.time() - start) > timeout:
                logger.error(f"Timeout waiting for Claude capacity after {timeout}s")
                return False
            time.sleep(1)
        return True

    async def async_wait_for_capacity(self, timeout: float | None = None) -> bool:
        """
        Async version of wait_for_capacity. Uses asyncio.sleep to not block event loop.

        This allows asyncio.wait_for (agent_timeout) to cancel the wait,
        and lets other async tasks run during the wait.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if capacity available, False if timed out.
        """
        # First check runs in start(), so _latest_usage should already be set.
        # Brief wait as safety net in case background thread hasn't persisted yet.
        if self._latest_usage is None:
            for _ in range(10):
                if self._latest_usage is not None:
                    break
                await asyncio.sleep(1)
            if self._latest_usage is None:
                logger.warning("Usage monitor has no data — continuing without monitoring")
                return True

        if not self._is_rate_limited.is_set():
            return True

        # Wait for rate limit to clear (monitor loop logs status)
        start = time.time()
        while self._is_rate_limited.is_set():
            if timeout is not None and (time.time() - start) > timeout:
                logger.error(f"Timeout waiting for Claude capacity after {timeout}s")
                return False
            await asyncio.sleep(1)
        return True

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return self._is_rate_limited.is_set()


# Global monitor instance
_monitor: UsageMonitor | None = None


def get_monitor() -> UsageMonitor:
    """Get the global usage monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = UsageMonitor()
    return _monitor


def require_capacity(timeout: float | None = None) -> bool:
    """
    Sync helper - wait for capacity before proceeding.

    Use this at the start of any sync function that calls Claude.
    For async code, use async_require_capacity instead.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if capacity available, False if timed out.
    """
    monitor = get_monitor()
    return monitor.wait_for_capacity(timeout=timeout)


async def async_require_capacity(timeout: float | None = None) -> bool:
    """
    Async helper - wait for capacity before proceeding.

    Use this at the start of any async function that calls Claude.
    Uses asyncio.sleep so it doesn't block the event loop.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if capacity available, False if timed out.
    """
    monitor = get_monitor()
    return await monitor.async_wait_for_capacity(timeout=timeout)
