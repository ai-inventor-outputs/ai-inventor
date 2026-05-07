"""Tmux session management — single source of truth.

Every "start a tmux session" in this repo (aii_launcher, aii_server, the
Claude usage scraper, oauth bootstrap, the db_backup supervisor) routes
through :func:`launch_in_tmux` here. Don't add another tmux launch site
elsewhere — extend this module instead.

Usage:
    from aii_lib.utils.tmux import launch_in_tmux, session_exists, kill_session

    launch_in_tmux(
        session="aii-server",
        cmd="python aii_server.py",
        log_file="logs/server.log",
        cwd="/repo/aii_server",
    )
"""

import os
import shlex
import subprocess
import time
from pathlib import Path


def session_exists(name: str) -> bool:
    """Check if a tmux session with this name exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def kill_session(name: str) -> None:
    """Kill a tmux session by name. No-op if it doesn't exist."""
    subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)


def rename_session(old: str, new: str) -> bool:
    """Rename a tmux session. Returns True on success.

    Returns False if ``old`` doesn't exist or the rename failed (e.g. a
    session named ``new`` already exists — tmux refuses to overwrite).
    """
    if not session_exists(old):
        return False
    result = subprocess.run(
        ["tmux", "rename-session", "-t", old, new],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def in_tmux() -> bool:
    """Check if we're currently running inside a tmux session."""
    return os.environ.get("TMUX_PANE") is not None


def current_session_name() -> str | None:
    """Return the tmux session name we're running inside, or ``None``.

    Used by the pipeline boot to rename its (placeholder-named) session
    to ``aii-<run_id>`` once the pipeline auto-generates its run id —
    fresh-run starts from the dashboard launch tmux with a uuid
    placeholder because the dashboard doesn't know the run id upfront.
    """
    if not in_tmux():
        return None
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#S"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def new_session_env_flags(extra: dict[str, str] | None = None) -> list[str]:
    """Build ``-e KEY=VAL`` args for ``tmux new-session``.

    ``tmux new-session`` ignores the ``env=`` passed to ``subprocess.run`` —
    the inner command inherits the *tmux server's* env, not the caller's.
    The ``-e`` flag is the only way to force per-session env vars that the
    inner command will actually see.

    Always forwards:
      - ``CLAUDE_CONFIG_DIR`` (so the inner ``claude`` writes creds to the
        configured dir, not the default ``~/.claude``)
      - ``CLAUDECODE=`` cleared (the marker the CLI sets when running
        inside itself)
      - ``PYTHONUNBUFFERED=1`` (default-on so Python child stdout flushes
        line-by-line into the tee'd log; harmless for non-Python commands)

    ``extra`` overrides any of the defaults if it sets the same key — tmux
    keeps the last ``-e`` for a given key.
    """
    flags: list[str] = []
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        flags += ["-e", f"CLAUDE_CONFIG_DIR={cfg}"]
    flags += ["-e", "CLAUDECODE="]
    flags += ["-e", "PYTHONUNBUFFERED=1"]
    for k, v in (extra or {}).items():
        flags += ["-e", f"{k}={v}"]
    return flags


# Static (non-run) tmux sessions launched by aii_launcher / aii_server. The
# pipeline-run prefix matcher excludes these so stop_all / orphan-detection
# don't accidentally target them as if they were run sessions.
_STATIC_AII_SESSIONS: frozenset[str] = frozenset(
    {
        "aii-pipeline",  # transient placeholder before cli.py renames
        "aii-server",
        "aii-dev-frontend",
        "aii-storybook",
        "aii-db-backup",
    }
)

# Prefix-based static-session matchers. Local tmux mirrors of remote RunPod
# pod logs (SSH+tail with retry) are spawned per-pod by aii_launcher --runpod
# with names like ``aii-runpod-server-<pod_id>`` / ``aii-runpod-orch-<pod_id>``
# — multiple deployments don't clobber each other. Treated as static services
# (not pipeline runs) so ``stop_local_runs`` doesn't sweep them up.
RUNPOD_MIRROR_SERVER_PREFIX = "aii-runpod-server-"
RUNPOD_MIRROR_ORCH_PREFIX = "aii-runpod-orch-"
_STATIC_AII_PREFIXES: tuple[str, ...] = (
    RUNPOD_MIRROR_SERVER_PREFIX,
    RUNPOD_MIRROR_ORCH_PREFIX,
)


def pipeline_session_name(run_id: str) -> str:
    """Derive a tmux session name from a pipeline run_id.

    Sanitizes characters that tmux doesn't allow in session names.
    Both aii_launcher and aii_server use this for consistent naming.
    """
    sanitized = run_id.replace(".", "-").replace(":", "-")
    return f"aii-{sanitized}"


def is_pipeline_run_session(name: str) -> bool:
    """True iff ``name`` is a pipeline-run tmux session.

    Pipeline runs use the ``aii-{run_id}`` shape (or ``aii-pending-{uuid}``
    while the pipeline hasn't yet renamed itself to the canonical form);
    static services like ``aii-server`` / ``aii-storybook`` and
    per-pod runpod log mirrors (``aii-runpod-{server,orch}-{pod_id}``)
    are excluded.
    """
    if not name.startswith("aii-"):
        return False
    if name in _STATIC_AII_SESSIONS:
        return False
    return not any(name.startswith(p) for p in _STATIC_AII_PREFIXES)


def list_sessions() -> list[str]:
    """Return the names of all currently running tmux sessions.

    Empty list when no sessions exist or the tmux server is down — both
    cases treated the same since callers only need "what's there now".
    """
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [n for n in result.stdout.strip().splitlines() if n]


def capture_pane(session: str) -> str:
    """Return the full scrollback buffer of ``session``'s active pane.

    ``-S -`` walks back to the start of the buffer so callers see the
    whole TUI state, not just what's on-screen. Returns "" if the session
    is gone or tmux isn't reachable.
    """
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def send_keys(session: str, *keys: str) -> None:
    """Send raw key tokens to a tmux session via ``tmux send-keys``.

    Each token is a tmux key spec (e.g. ``"C-c"``, ``"Enter"``,
    ``"Escape"``, ``"BSpace"``) or empty string (terminator). For literal
    text input use :func:`send_text`.

    Silent on tmux/timeout errors — input ops are best-effort and the
    caller verifies via ``capture_pane``.
    """
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", session, *keys],
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def send_text(session: str, text: str) -> None:
    """Type ``text`` literally into a tmux session (``send-keys -l``).

    The ``-l`` flag treats the input as literal characters — no key-token
    interpretation, so braces / quotes / spaces all pass through unchanged.
    """
    try:
        subprocess.run(
            ["tmux", "send-keys", "-l", "-t", session, text],
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def resize_window(session: str, width: int, height: int) -> None:
    """Resize ``session``'s active window to ``width`` × ``height`` columns/rows.

    Used by the /usage scraper to force a wide-enough viewport that the
    Claude TUI's status row isn't truncated mid-token before capture.
    """
    subprocess.run(
        ["tmux", "resize-window", "-t", session, "-x", str(width), "-y", str(height)],
        capture_output=True,
        timeout=5,
    )


def pipe_pane(session: str, target_cmd: str | None = None) -> None:
    """Pipe (or unpipe) a tmux pane's raw byte stream to a shell command.

    Args:
        session: tmux session whose active pane to pipe.
        target_cmd: shell command to receive the byte stream (e.g.
            ``"cat >> /tmp/raw.log"``). Pass ``None`` to stop the active
            pipe — tmux's ``pipe-pane`` with no command toggles off.

    Why: ``capture-pane`` reads the rendered terminal buffer and can miss
    transient frames; ``pipe-pane`` captures every byte tmux sees so
    short-lived dialogs are recoverable from the raw log.
    """
    argv = ["tmux", "pipe-pane", "-t", session]
    if target_cmd is not None:
        argv += ["-o", target_cmd]
    subprocess.run(argv, capture_output=True, timeout=10)


def wait_for_text(session: str, text: str, timeout: int = 30) -> str | None:
    """Poll ``capture_pane`` until ``text`` appears or ``timeout`` elapses.

    Returns the full pane buffer when the text shows up, ``None`` on
    timeout. Used by interactive flows (autologin, /usage) to gate on TUI
    state transitions.
    """
    for _ in range(timeout):
        output = capture_pane(session)
        if text in output:
            return output
        time.sleep(1)
    return None


def get_pane_pid(session: str) -> int | None:
    """Get the PID of the process running inside a tmux session's pane."""
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def stop_session(name: str) -> bool:
    """Stop a tmux session by killing the process tree, then the session.

    Returns True if the session was running.
    """
    if not session_exists(name):
        return False

    # Grab the PID before we destroy the session
    pane_pid = get_pane_pid(name)

    # Kill the process tree (tmux kill-session alone doesn't kill child processes)
    if pane_pid:
        import signal

        try:
            os.killpg(os.getpgid(pane_pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pane_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    if session_exists(name):
        kill_session(name)
    return True


def pkill_orphans(pattern: str) -> None:
    """Kill orphan processes matching a distinctive cmdline substring.

    When a tmux session dies without cleanly shutting down its children
    (e.g. ``kill -9`` on tmux itself), the Python processes get reparented
    to init and keep holding their sockets. On the next deploy they prevent
    the new server from binding to its port and the wait-loop times out.

    Critically, this excludes the tmux daemon: ``tmux new-session -d <cmd>``
    spawns the daemon by forking from the new-session invocation, so the
    daemon inherits a cmdline like ``tmux new-session -d -s … <cmd>`` — i.e.
    it carries the args of the very command we're trying to clean up. A
    naive ``pkill -f <pattern>`` would match the daemon and kill it, taking
    down every tmux session including unrelated runpod log mirrors and
    in-flight pipeline runs. We filter on ``/proc/<pid>/comm`` to skip any
    process whose name is ``tmux: server`` (or any ``tmux:`` variant).

    Two-phase kill: SIGTERM first to let well-behaved children clean up,
    then SIGKILL to mop up survivors. Needed because aii_server spawns
    multiprocessing Manager subprocesses (one per registered ability) that
    install their own SIGTERM handler — under load that handler can hang
    long enough to outlive a single-pass pkill, leaving Apr-29-style
    long-lived zombies reparented to systemd-user.
    """
    import signal

    def _matching_pids(pattern: str) -> list[int]:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        pids: list[int] = []
        for tok in result.stdout.split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
            except OSError:
                continue
            if comm.startswith("tmux"):
                # tmux daemon inherited the new-session cmdline — skip it.
                continue
            pids.append(pid)
        return pids

    def _signal(pids: list[int], sig: int) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    _signal(_matching_pids(pattern), signal.SIGTERM)
    time.sleep(0.4)
    _signal(_matching_pids(pattern), signal.SIGKILL)
    time.sleep(0.2)


def launch_in_tmux(
    session: str,
    cmd: str,
    *,
    cwd: str | None = None,
    log_file: str | None = None,
    extra_env: dict[str, str] | None = None,
    orphan_pattern: str | None = None,
) -> None:
    """Launch a command in a named tmux session (detached).

    Args:
        session: tmux session name. Killed first if it already exists.
        cmd: shell command to run (evaluated by tmux's $SHELL).
        cwd: working directory; passed via tmux ``-c`` flag.
        log_file: if set, ``tee`` stdout+stderr to this file (parent dir
            created if missing).
        extra_env: extra env vars forwarded to the inner command via
            tmux ``-e`` flags. Overrides defaults of the same key.
        orphan_pattern: if set, ``pkill -f <pattern>`` before launch to
            clear stale processes reparented from a previous session.

    Always detached. Returns immediately. Use ``session_exists()`` to
    verify the session actually came up.
    """
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    kill_session(session)
    if orphan_pattern:
        pkill_orphans(orphan_pattern)

    inner = cmd
    if log_file:
        inner = f"{inner} 2>&1 | tee {shlex.quote(log_file)}"

    argv: list[str] = ["tmux", "new-session", "-d", "-s", session]
    if cwd:
        argv += ["-c", cwd]
    argv += new_session_env_flags(extra_env)
    argv += [inner]

    subprocess.run(argv, timeout=10)
