"""Fetch Claude Code usage statistics programmatically.

Uses tmux to interact with the claude CLI and capture /usage output.
Requires: tmux, claude CLI.

In Docker containers, claude always shows the onboarding flow (theme selector →
login method) on every launch. This module maintains a persistent tmux session
to avoid repeated onboarding. On first use, it completes the full login flow
(via autologin_claude) if needed, then keeps the session alive for subsequent
/usage checks.

Capture strategy (v2.1.63+):
The /usage command opens a TUI settings dialog that capture-pane can't always
see (modal overlay). We use two parallel capture methods:
  1. capture-pane: standard tmux screen buffer (works when dialog persists)
  2. pipe-pane: raw terminal byte stream capture (catches dialog content even
     if it's briefly rendered or dismissed before capture-pane sees it)
"""

import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aii_lib.utils.retry import make_retry_log
from aii_lib.utils.tmux import (
    capture_pane,
    kill_session,
    launch_in_tmux,
    pipe_pane,
    resize_window,
    send_keys,
    send_text,
    session_exists,
)

# Regex to strip ANSI escape sequences from raw pipe-pane output
_ANSI_RE = re.compile(
    r"\x1b"  # ESC
    r"(?:"
    r"\[[0-9;?]*[A-Za-z]"  # CSI sequences: \x1b[...X
    r"|\][^\x07]*\x07"  # OSC sequences: \x1b]...\x07
    r"|\([AB012]"  # Charset selection: \x1b(A etc.
    r"|[NOcDEHMZ78>=]"  # SS2, SS3, misc single-char
    r")"
)


@dataclass
class ClaudeUsage:
    """Claude Code usage statistics."""

    current_session: int | None = None
    current_week_all_models: int | None = None
    current_week_sonnet: int | None = None

    def __str__(self) -> str:
        return (
            f"current_session: {self.current_session}%\n"
            f"current_week_all_models: {self.current_week_all_models}%\n"
            f"current_week_sonnet: {self.current_week_sonnet}%"
        )


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from raw terminal output."""
    return _ANSI_RE.sub("", text)


class UsageRateLimitedError(RuntimeError):
    """Raised when the Claude usage API itself is rate-limited."""


def _parse_usage(raw_output: str) -> ClaudeUsage:
    """Parse usage percentages from tmux output (capture-pane or pipe-pane).

    Finds all N% occurrences in order and assigns them positionally:
    1st = current_session, 2nd = current_week_all_models, 3rd = current_week_sonnet.

    Handles both clean capture-pane output and raw pipe-pane output with
    interleaved ANSI escape sequences.

    Raises UsageRateLimitedError if the dialog shows a rate limit error
    instead of usage data.
    """
    cleaned = _strip_ansi(raw_output)

    # Detect rate limit error from the /usage dialog itself
    if "rate_limit_error" in cleaned.lower() or "rate limited" in cleaned.lower():
        raise UsageRateLimitedError("Claude usage API is rate-limited")

    # Match "N% used" with possible invisible/whitespace chars between them
    percentages = re.findall(r"(\d+)\s*%\s*u\s*s\s*e\s*d", cleaned)
    return ClaudeUsage(
        current_session=int(percentages[0]) if percentages else None,
        current_week_all_models=int(percentages[1]) if len(percentages) > 1 else None,
        current_week_sonnet=int(percentages[2]) if len(percentages) > 2 else None,
    )


def _check_prerequisites() -> None:
    """Verify tmux and claude are available. Raises RuntimeError if not."""
    if not shutil.which("tmux"):
        raise RuntimeError("tmux not found on PATH")
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found on PATH")


# ---------------------------------------------------------------------------
# Composite helper (usage-specific)
# ---------------------------------------------------------------------------


def _clear_input(session: str) -> None:
    """Clear any stale text from the Claude CLI input field.

    Uses Ctrl+U to clear the line, then backspaces as fallback.
    Does NOT send Escape — at the main prompt, Escape opens the Rewind
    panel which breaks /usage. Dialog cleanup is handled by
    _ensure_persistent_session instead.
    """
    send_keys(session, "C-u")
    time.sleep(0.1)
    send_keys(session, *(["BSpace"] * 10))
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# Persistent session management
# ---------------------------------------------------------------------------

_PERSISTENT_SESSION = "claude_usage_persistent"


def _is_main_prompt(output: str) -> bool:
    """Check if the Claude CLI is at its main input prompt (not a dialog).

    Looks for footer markers that only appear at the main prompt and not
    on onboarding/trust/login dialogs. Using just "❯" is ambiguous because
    it also appears in the workspace trust selector.

    Includes markers from multiple CLI versions:
      - "for shortcuts" — v2.1.x ("? for shortcuts" footer)
      - "tokens" / "permissions" — older versions (status bar)
    """
    return "for shortcuts" in output or "tokens" in output or "permissions" in output


def _ensure_persistent_session() -> str:
    """Ensure a persistent claude tmux session exists and is at the main prompt.

    On first call (or after session dies), starts claude, navigates through
    onboarding/trust/login, and leaves the session at the main input prompt.
    On subsequent calls, reuses the existing session.

    Returns the session name.
    """
    if session_exists(_PERSISTENT_SESSION):
        # Session exists — verify Claude CLI is still running inside it
        output = capture_pane(_PERSISTENT_SESSION)
        # If /usage dialog is still open from last check, close it
        if "% used" in output:
            logger.debug("Usage dialog still open — closing")
            send_keys(_PERSISTENT_SESSION, "Escape")
            time.sleep(1)
            return _PERSISTENT_SESSION
        # CLI exited — shell prompt visible (no Claude TUI markers)
        if "command not found" in output or (
            "$" in output and "Claude" not in output and "❯" not in output
        ):
            logger.warning("Claude CLI exited — recreating session")
            kill_session(_PERSISTENT_SESSION)
        # Session stuck at onboarding/login — kill and recreate
        elif "Select login method" in output or "Dark mode" in output:
            logger.warning("Session stuck at onboarding — recreating")
            kill_session(_PERSISTENT_SESSION)
        elif "Skill(usage)" in output or "/usage is a built-in" in output:
            # Previous /usage was processed as a user message (not a CLI command)
            # due to input doubling — session now has a conversation. Kill and
            # recreate for a clean state.
            logger.warning("Previous /usage sent as user message — recreating session")
            kill_session(_PERSISTENT_SESSION)
        # Stuck at workspace trust dialog — confirm and wait for main prompt
        elif "trust this folder" in output or "safety check" in output:
            logger.debug("Workspace trust dialog — confirming")
            send_keys(_PERSISTENT_SESSION, "Enter")
            for _ in range(15):
                time.sleep(1)
                output = capture_pane(_PERSISTENT_SESSION)
                if _is_main_prompt(output):
                    return _PERSISTENT_SESSION
            logger.warning("Trust dialog did not resolve — recreating session")
            kill_session(_PERSISTENT_SESSION)
        elif _is_main_prompt(output):
            # Main TUI active — reuse
            return _PERSISTENT_SESSION
        else:
            # Unknown state — log and recreate for safety
            non_blank = [ln.strip() for ln in output.splitlines() if ln.strip()]
            snippet = " | ".join(non_blank[-3:]) if non_blank else "(empty)"
            logger.warning(f"Session in unknown state — recreating | {snippet}")
            kill_session(_PERSISTENT_SESSION)

    # Create new session in /tmp to avoid loading project context (CLAUDE.md,
    # memory, skills) and polluting the project's conversation cache with
    # usage-check sessions.
    logger.debug("Creating persistent claude session for usage checks")
    launch_in_tmux(session=_PERSISTENT_SESSION, cmd="claude", cwd="/tmp")

    # Wait for CLI to fully start — use _is_main_prompt to distinguish the
    # actual prompt from intermediate dialogs (trust, onboarding) that also
    # contain "❯".
    output = ""
    for _ in range(45):  # up to 45s (trust dialog adds time)
        time.sleep(1)
        output = capture_pane(_PERSISTENT_SESSION)
        if _is_main_prompt(output):
            break
        # Navigate trust dialog mid-wait
        if "trust this folder" in output or "safety check" in output:
            logger.debug("Navigating workspace trust dialog")
            send_keys(_PERSISTENT_SESSION, "Enter")

    # Navigate theme selector
    if "Dark mode" in output:
        logger.debug("Navigating theme selector")
        send_keys(_PERSISTENT_SESSION, "Enter")
        time.sleep(3)
        output = capture_pane(_PERSISTENT_SESSION)

    # Handle "Not logged in" screen
    if "Not logged in" in output or ("/login" in output and "Select login method" not in output):
        logger.warning("Usage session not logged in — triggering /login")
        send_text(_PERSISTENT_SESSION, "/login")
        time.sleep(0.5)
        send_keys(_PERSISTENT_SESSION, "Enter")
        time.sleep(3)
        output = capture_pane(_PERSISTENT_SESSION)

    # If CLI needs login, don't run OAuth here — let /claude/credentials handle it.
    # Running OAuth from the usage scraper races with the credentials endpoint.
    if "Select login method" in output or "Paste code here" in output:
        kill_session(_PERSISTENT_SESSION)
        raise UsageRateLimitedError(
            "OAuth token not ready — waiting for /claude/credentials to authenticate"
        )

    # Final check: if we're still not at the main prompt, something went wrong
    if not _is_main_prompt(output):
        non_blank = [ln.strip() for ln in output.splitlines() if ln.strip()]
        snippet = " | ".join(non_blank[-3:]) if non_blank else "(empty)"
        logger.warning(f"Session did not reach main prompt — {snippet}")

    return _PERSISTENT_SESSION


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=30, min=30, max=30),
    retry=retry_if_not_exception_type(UsageRateLimitedError),
    before_sleep=make_retry_log(label="claude usage"),
    reraise=True,
)
def get_claude_usage(timeout_seconds: int = 30) -> ClaudeUsage:
    """
    Fetch Claude Code usage statistics.

    Uses a persistent tmux session to avoid repeated onboarding in Docker.
    Sends /usage, then uses both capture-pane and pipe-pane (raw terminal
    byte stream) to find the usage percentages, polling rapidly instead of
    sleeping for a fixed duration.

    Args:
        timeout_seconds: Max seconds to wait for usage data to appear.

    Returns:
        ClaudeUsage dataclass with usage percentages.

    Raises:
        RuntimeError: If unable to fetch or parse usage data.
    """
    _check_prerequisites()

    # Kill any existing session to avoid stale pane buffer (rate limit errors,
    # old usage dialogs) from previous checks
    try:
        kill_session(_PERSISTENT_SESSION)
    except Exception:
        pass

    session_name = _ensure_persistent_session()

    # Ensure pane is large enough for the dialog to render.
    # Resize ONCE up front then let the CLI settle — SIGWINCH from resize
    # can dismiss the /usage dialog if it arrives mid-render.
    resize_window(session_name, 200, 50)
    time.sleep(1)

    # Start raw terminal output capture (catches dialog even if transient)
    pipe_file = Path(f"/tmp/claude_usage_pipe_{uuid.uuid4().hex[:8]}.raw")
    pipe_file.write_text("")

    try:
        pipe_pane(session_name, f"cat >> {pipe_file}")

        # Send /usage once, then rapidly capture the screen — the dialog may
        # flash briefly then auto-dismiss, so we accumulate ALL captures and
        # search through them at the end.
        all_captures: list[str] = []
        usage = ClaudeUsage()
        rate_limited = False

        _clear_input(session_name)
        send_text(session_name, "/usage")
        time.sleep(0.5)
        send_keys(session_name, "Enter")

        # Rapid poll for the full timeout — capture as often as possible
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            time.sleep(0.3)
            try:
                raw_pane = capture_pane(session_name)
                all_captures.append(raw_pane)
                usage = _parse_usage(raw_pane)
                if usage.current_week_all_models is not None:
                    break

                if pipe_file.exists() and pipe_file.stat().st_size > 0:
                    raw_pipe = pipe_file.read_text(errors="replace")
                    all_captures.append(raw_pipe)
                    usage = _parse_usage(raw_pipe)
                    if usage.current_week_all_models is not None:
                        break
            except UsageRateLimitedError:
                rate_limited = True
                break

        # Stop pipe capture
        pipe_pane(session_name)

        # If live polling didn't find data, search ALL accumulated captures
        if usage.current_week_all_models is None and not rate_limited:
            for cap in all_captures:
                try:
                    usage = _parse_usage(cap)
                    if usage.current_week_all_models is not None:
                        logger.debug("Found usage data in accumulated captures")
                        break
                except UsageRateLimitedError:
                    rate_limited = True
                    break

        # Close dialog — send Escape once, then check if we're back at the
        # main prompt. A second Escape at the main prompt opens the Rewind
        # panel, which breaks subsequent /usage calls.
        send_keys(session_name, "Escape")
        time.sleep(0.5)
        post_close = capture_pane(session_name)
        if "% used" in post_close or "Esc to cancel" in post_close:
            send_keys(session_name, "Escape")
            time.sleep(0.5)

        if rate_limited:
            raise UsageRateLimitedError("Claude usage API is rate-limited")

        if usage.current_week_all_models is None:
            # Detect "Loading usage data…" plateau — the TUI sometimes hangs
            # on this state indefinitely (no usage rendering, no error).
            # Bail without writing a debug capture, since the captures are
            # all the same loading frame and not useful.
            stuck_loading = any("Loading usage data" in cap for cap in all_captures[-3:])
            if stuck_loading:
                raise RuntimeError("Claude /usage TUI stuck on 'Loading usage data…' — skipping")
            # Stable filename per session — overwritten on each failure
            # rather than uuid-suffixed (which accumulated 144KB/hour of
            # forgettable debug captures with no cleanup, per the errors
            # doc). One file per session is enough to inspect the most
            # recent failure.
            debug_file = Path(f"/tmp/claude_usage_raw_{session_name}.txt")
            debug_parts = []
            for i, cap in enumerate(all_captures[-6:]):  # last 6 captures
                debug_parts.append(f"=== capture {i} ===\n{cap}")
            if pipe_file.exists():
                debug_parts.append(f"=== pipe-pane ===\n{pipe_file.read_text(errors='replace')}")
            debug_file.write_text("\n\n".join(debug_parts))
            raise RuntimeError(
                f"Failed to parse usage data from {len(all_captures)} captures "
                f"(saved to {debug_file})"
            )

        return usage
    finally:
        pipe_file.unlink(missing_ok=True)


def cleanup_persistent_session() -> None:
    """Kill the persistent usage session. Call on shutdown."""
    if session_exists(_PERSISTENT_SESSION):
        send_keys(_PERSISTENT_SESSION, "Escape")
        time.sleep(0.5)
        send_text(_PERSISTENT_SESSION, "/exit")
        send_keys(_PERSISTENT_SESSION, "Enter")
        time.sleep(1)
        kill_session(_PERSISTENT_SESSION)


if __name__ == "__main__":
    try:
        usage = get_claude_usage()
        logger.info(usage)
    finally:
        cleanup_persistent_session()
