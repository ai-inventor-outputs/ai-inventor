#!/usr/bin/env python3
"""aii_launcher — Start aii_server + aii_pipeline (locally or on RunPod).

Usage:
    aii_launcher                          # local mode (default)
    aii_launcher --local                  # same as above
    aii_launcher --runpod                 # deploy to RunPod pods
    aii_launcher --resume <pod>           # resume streaming from existing pod
    aii_launcher --server-only            # just start aii_server
    aii_launcher --pipeline-only          # just start pipeline in tmux (no server)
    aii_launcher                          # dev Next.js frontend by default (local only)
    aii_launcher --dev-frontend           # explicit form of the default
    aii_launcher --prod-frontend          # opt into ``next build`` + ``next start``
    aii_launcher --no-frontend            # opt out of any frontend

aii_pipeline and aii_server are simple foreground processes.
aii_launcher handles tmux, health checks, and orchestration.
"""

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Tmux session names
SERVER_SESSION = "aii-server"
PIPELINE_SESSION = "aii-pipeline"

# Local tmux mirrors of remote RunPod pod logs use prefixes
# ``aii-runpod-server-<pod_id>`` / ``aii-runpod-orch-<pod_id>`` — see
# ``aii_runpod.deploy.remote._spawn_local_log_mirror``. They self-terminate
# within ~60 s of their remote pod being deleted, so ``stop_all`` leaves
# them alone — killing them only stops watching the pod's logs, the actual
# pods aren't affected anyway.

# Sessions owned by an aii_launcher --local invocation: the previous server,
# the previous immediate-CLI pipeline, and every static service the server
# spawns (aii-storybook + aii-db-backup are launched from aii_server.py /
# db_backup_supervisor.py at boot — they outlive aii-server itself otherwise).
# Pipeline-run sessions (``aii-<run_id>``) and runpod mirrors are NOT in this
# list so a local re-launch leaves dashboard-spawned runs alone.
_LOCAL_OWNED_SESSIONS: tuple[str, ...] = (
    PIPELINE_SESSION,
    SERVER_SESSION,
    "aii-dev-frontend",
    "aii-prod-frontend",
    "aii-storybook",
    "aii-db-backup",
    "claude_usage_persistent",
)


def _resolve_session_for_pid(pid: int) -> str | None:
    """Return the tmux session name whose pane PID matches ``pid``, or None.

    Single-shot lookup — used for refreshing the session name at detach
    time (in case ``_wait_for_renamed_session`` returned the fallback
    because rename hadn't completed yet).
    """
    from aii_lib.utils.tmux import get_pane_pid, list_sessions

    for name in list_sessions():
        if get_pane_pid(name) == pid:
            return name
    return None


def _wait_for_renamed_session(pid: int, timeout: float = 30.0) -> str:
    """Resolve the live tmux session name for the pipeline pane.

    aii_pipeline.cli renames the launched ``aii-pipeline`` session to
    ``aii-<run_id>`` once Run.gen_id() returns. Empirical timing varies
    a lot — first-time imports under ty/oxlint contention can push this
    past 5 s — so the timeout is generous; the rename normally lands
    within 2 s and the loop exits early.

    Polls list_sessions for a pipeline-run session (``aii-<id>``) whose
    pane PID matches ``pid``; falls back to whichever session matches,
    then PIPELINE_SESSION.
    """
    from aii_lib.utils.tmux import get_pane_pid, is_pipeline_run_session, list_sessions

    deadline = time.time() + timeout
    fallback: str | None = None
    while time.time() < deadline:
        for name in list_sessions():
            if get_pane_pid(name) != pid:
                continue
            if is_pipeline_run_session(name):
                return name
            fallback = name
        time.sleep(0.1)
    return fallback or PIPELINE_SESSION


# ---------------------------------------------------------------------------
# Local deployment
# ---------------------------------------------------------------------------


def _start_local_server_and_wait(
    *,
    logs_dir: Path,
    dev_frontend: bool = False,
    prod_frontend: bool = False,
    dev_autologin: bool = False,
    reuse_if_healthy: bool = False,
) -> int:
    """Start aii_server in a tmux session, wait for /agent_abilities/health.

    Used by both ``deploy_local`` (with dev autologin) and ``deploy_runpod``
    (without — the RunPod path uses the local aii_server purely as the
    proxy that holds RUNPOD_API_KEY and exposes the ``aii_runpod__*``
    abilities).

    With ``reuse_if_healthy=True`` (set by ``deploy_runpod``) the existing
    session is left alone if ``/agent_abilities/health`` already returns
    200 — important so a runpod deploy doesn't blow away a local server
    the user is actively using. Otherwise (default, used by
    ``deploy_local``) the existing session is killed and replaced.

    Returns 0 on healthy, 1 on health timeout.
    """
    from aii_lib.server_url import SERVER_PORT
    from aii_lib.utils.tmux import kill_session, launch_in_tmux

    if reuse_if_healthy and _server_already_healthy(SERVER_PORT):
        print(f"  aii_server already healthy on :{SERVER_PORT} — reusing existing session")
        return 0

    kill_session(SERVER_SESSION)

    # AII_DEV_AUTOLOGIN=1 opts into dashboard auto-login as admin — only
    # appropriate for single-user local dev. RunPod paths require real /auth login.
    server_cmd = (
        "AII_DEV_AUTOLOGIN=1 " if dev_autologin else ""
    ) + f"{_py()} {PROJECT_ROOT / 'aii_server' / 'aii_server.py'} "
    if dev_frontend:
        server_cmd += " --dev-frontend"
    if prod_frontend:
        server_cmd += " --prod-frontend"

    launch_in_tmux(
        session=SERVER_SESSION,
        cmd=server_cmd,
        log_file=str(logs_dir / "aii_server.log"),
        cwd=str(PROJECT_ROOT / "aii_server"),
        extra_env=_aii_pipeline_env(),
        orphan_pattern=str(PROJECT_ROOT / "aii_server" / "aii_server.py"),
    )
    print(f"  aii_server started in tmux '{SERVER_SESSION}'")

    print("  Waiting for aii_server health...")
    if not _wait_for_server(SERVER_PORT, timeout=180):
        print("  ERROR: aii_server not healthy after 180s")
        print(f"  Check: tmux attach -t {SERVER_SESSION}")
        return 1
    print("  aii_server healthy")
    return 0


def deploy_local(
    dev_frontend: bool = False,
    prod_frontend: bool = False,
    server_only: bool = False,
    pipeline_args: list[str] | None = None,
    stop_local_runs: bool = False,
) -> int:
    """Start aii_server + aii_pipeline in tmux sessions locally.

    Stops only the previous launcher's own sessions (``_LOCAL_OWNED_SESSIONS``)
    by default — pipeline-run sessions (``aii-<run_id>``) spawned by the
    dashboard are left alone so a local re-launch doesn't kill in-flight runs.
    Pass ``stop_local_runs=True`` to additionally kill every pipeline-run
    tmux session.

    Detached launches; the pipeline log is then streamed to the foreground
    via ``tail -f`` so Ctrl+C stops the stream while the tmux processes keep
    running.
    """
    from aii_lib.utils.paths import logs_dir as _logs_dir
    from aii_lib.utils.tmux import (
        get_pane_pid,
        is_pipeline_run_session,
        kill_session,
        launch_in_tmux,
        list_sessions,
    )

    logs_dir = _logs_dir("deploy")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Always kill the previous launcher's own sessions — but never pipeline
    # runs unless the user opted in via --stop-local-runs.
    for session in _LOCAL_OWNED_SESSIONS:
        if session == PIPELINE_SESSION and server_only:
            continue
        kill_session(session)

    if stop_local_runs:
        killed_runs: list[str] = []
        for name in list_sessions():
            if is_pipeline_run_session(name) and name not in _LOCAL_OWNED_SESSIONS:
                kill_session(name)
                killed_runs.append(name)
        if killed_runs:
            print(f"  --stop-local-runs: killed {len(killed_runs)} run session(s)")

    if rc := _start_local_server_and_wait(
        logs_dir=logs_dir,
        dev_frontend=dev_frontend,
        prod_frontend=prod_frontend,
        dev_autologin=True,
    ):
        return rc

    if server_only:
        print(f"\n  attach: tmux attach -t {SERVER_SESSION}")
        print(f"  log:    tail -f {logs_dir / 'aii_server.log'}")
        return 0

    # 4. Start aii_pipeline in tmux (detached). ``--execute-mode=local`` is
    # threaded down so the pipeline's ExecuteEnvConfig matches the launcher's
    # ``--local`` intent (artifacts run in-process on this host instead of
    # dispatching to RunPod worker pods).
    pipeline_cmd = f"{_py()} -m aii_pipeline.cli --execute-mode=local "
    if pipeline_args:
        pipeline_cmd += " " + " ".join(shlex.quote(a) for a in pipeline_args)

    # Note: no ``orphan_pattern`` — ``pkill -f aii_pipeline.cli`` would also
    # kill dashboard-spawned run sessions (``aii-<run_id>``) that share the
    # ``aii_pipeline.cli`` argv. The kill_session above is enough for the
    # static aii-pipeline session; reparented zombies from a previous
    # abnormal exit are handled by ``--stop-local-runs`` if needed.
    launch_in_tmux(
        session=PIPELINE_SESSION,
        cmd=pipeline_cmd,
        log_file=str(logs_dir / "pipeline.log"),
        cwd=str(PROJECT_ROOT),
        extra_env=_aii_pipeline_env(),
    )

    # Capture the pane PID IMMEDIATELY before the pipeline can rename
    # its tmux session — ``aii_pipeline.cli`` renames ``aii-pipeline`` to
    # the canonical ``aii-<run_id>`` shape as soon as ``Run.gen_id()``
    # returns (~1 s after spawn). A delayed ``session_exists(PIPELINE_SESSION)``
    # check would false-negative even when the pipeline is healthy.
    pipeline_pid = get_pane_pid(PIPELINE_SESSION)
    if pipeline_pid is None:
        print("  ERROR: pipeline tmux session failed to start")
        return 1
    print(f"  aii_pipeline started in tmux '{PIPELINE_SESSION}' (pid={pipeline_pid})")

    # Wait for the pipeline to rename its session to aii-<id> so the
    # attach hints below reference the live name, not the stale PIPELINE_SESSION.
    session_name = _wait_for_renamed_session(pipeline_pid)

    # 5. Stream pipeline log (Ctrl+C stops streaming, process keeps running).
    # Uses subprocess.run (not os.execvp) so the KeyboardInterrupt handler can
    # print the re-attach hint before returning.
    pipeline_log = logs_dir / "pipeline.log"
    print("\n  Streaming pipeline output (Ctrl+C to detach)...")
    print(f"  Re-attach: tmux attach -t {session_name}")
    print("  Stop all:  aii_launcher --stop-local")
    print()
    try:
        subprocess.run(["tail", "-f", str(pipeline_log)])
    except KeyboardInterrupt:
        pass
    # Re-resolve in case _wait_for_renamed_session timed out before the
    # rename to ``aii-<run_id>`` completed and we'd otherwise print the
    # stale ``aii-pipeline`` placeholder. By Ctrl+C time the rename has
    # almost always landed.
    final_session = _resolve_session_for_pid(pipeline_pid) or session_name
    print(f"\n  Detached. Pipeline still running in tmux '{final_session}'.")
    print(f"  attach: tmux attach -t {final_session}")
    return 0


# ---------------------------------------------------------------------------
# RunPod deployment
# ---------------------------------------------------------------------------


def _ensure_prod_frontend(logs_dir: Path) -> None:
    """Idempotently launch the production Next.js frontend in tmux.

    Used by ``deploy_runpod`` so the local web app is up regardless of
    whether ``_start_local_server_and_wait`` started a fresh server or
    reused a healthy one (aii_server only spawns its frontend on its own
    boot path, so the reuse branch needs us to spawn it ourselves).

    The dev frontend + Storybook are killed first since they both bind
    :3000 and would conflict with ``next start``.
    """
    from aii_lib.utils.tmux import kill_session, launch_in_tmux, session_exists

    kill_session("aii-dev-frontend")
    kill_session("aii-storybook")

    if session_exists("aii-prod-frontend"):
        print("  aii-prod-frontend already running — leaving alone")
        return

    frontend_dir = PROJECT_ROOT / "aii_frontend"
    if not (frontend_dir / "package.json").exists():
        print("  aii_frontend/package.json missing — skipping prod frontend")
        return

    launch_in_tmux(
        session="aii-prod-frontend",
        cmd="npm run build && npm run start",
        cwd=str(frontend_dir),
        log_file=str(logs_dir / "aii_frontend.log"),
    )
    print("  Next.js prod frontend launching in tmux 'aii-prod-frontend'")
    print("  (build + start; port 3000 reachable after build, ~1-2 min)")


async def deploy_runpod(
    pipeline_args: list[str] | None = None,  # noqa: ARG001 — accepted for CLI parity with deploy_local; reserved.
    server_only: bool = False,
    exec_mode: str | None = None,
    prod_frontend: bool = True,
) -> int:
    """Deploy server pod (and optionally pipeline pod) to RunPod.

    With ``server_only=True``, only the shared aii-server pod is created
    or reused — no orchestrator pod, no pipeline run, no streaming. Used
    to bring up the dashboard for app-deployment use without launching
    a pipeline.

    ``exec_mode`` overrides ``execution.mode`` in the shipped
    ``harness/execute_env.yaml`` so the pipeline running inside the
    orchestrator pod dispatches substep agents accordingly. The CLI
    always passes ``"runpod"`` from ``aii_launcher --runpod`` — there
    is no opt-out, since dispatching agents in-process inside a
    long-lived RunPod orchestrator pod defeats the purpose of running
    on RunPod at all.

    ``prod_frontend`` (default ``True``) launches the optimized Next.js
    web app locally so ``--runpod`` ships a working dashboard at :3000.
    The CLI flips it off via ``--no-frontend``.

    Local aii_server is started first as the proxy that holds
    ``RUNPOD_API_KEY`` and exposes the ``aii_runpod__*`` abilities; all
    RunPod operations from this point (template ensure inside
    ``deploy_and_run``, pod CRUD) route through it via :class:`RunPodClient`.
    """
    try:
        import aii_runpod  # noqa: F401  — presence probe for public builds
    except ImportError:
        print(
            "ERROR: --runpod requires the aii_runpod package, which is not\n"
            "       included in this build. Use --local (the default) instead."
        )
        return 1

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        return 1

    from aii_lib.utils.paths import logs_dir as _logs_dir

    logs_dir = _logs_dir("deploy")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Local aii_server — needed to call aii_runpod__* abilities below.
    #    Reuse the user's existing session if it's already healthy — a
    #    runpod deploy must not blow away whatever they're running locally.
    #    Frontend is spawned separately (step 2) so the reuse branch still
    #    gets it; passing ``prod_frontend`` here would double-launch.
    if rc := _start_local_server_and_wait(logs_dir=logs_dir, reuse_if_healthy=True):
        return rc

    # 2. Local production web app (port 3000) — opt out via --no-frontend.
    if prod_frontend:
        _ensure_prod_frontend(logs_dir)

    # 3. Hand off to remote-deploy. Template reconciliation lives inside
    #    ``deploy_and_run`` so the pipeline-driven entry point in
    #    ``aii_pipeline.cli`` gets it too.
    from aii_pipeline.utils.pipeline_config import PipelineConfig
    from aii_runpod.deploy import deploy_and_run

    config = PipelineConfig.from_yaml(PROJECT_ROOT / "aii_config" / "pipeline")
    if exec_mode in ("local", "runpod"):
        # In-memory override; deploy_and_run also rewrites the shipped
        # harness/execute_env.yaml so the orchestrator pod sees the same value.
        config.execute_env.mode = exec_mode
    return await deploy_and_run(config, server_only=server_only, exec_mode_override=exec_mode)


async def resume_stream(pod_name_or_id: str, after: str | None = None) -> int:
    """Resume streaming from an existing RunPod pod."""
    try:
        import aii_runpod  # noqa: F401  — presence probe for public builds
    except ImportError:
        print(
            "ERROR: --resume requires the aii_runpod package, which is not\n"
            "       included in this build."
        )
        return 1

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        return 1

    from aii_runpod.deploy import resume_stream as _resume

    return await _resume(
        pod_name_or_id=pod_name_or_id,
        api_key=api_key,
        after=after,
    )


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def stop_all() -> int:
    """Stop all *local* aii sessions. Leaves RunPod log mirrors alone.

    Sends SIGTERM (Ctrl-C) for a clean shutdown, then kills tmux for
    anything that didn't exit on its own.

    What gets stopped:
      * static launcher/server services (aii-server, aii-pipeline,
        aii-dev-frontend, aii-storybook, aii-db-backup, claude_usage_persistent)
      * pipeline-run sessions (``aii-<run_id>`` / ``aii-pending-<uuid>``)
        spawned by either the launcher or the dashboard

    What is **NOT** stopped:
      * RunPod log mirrors (``aii-runpod-{server,orch}-<pod_id>``) —
        these are local SSH-tail loops; killing them would just stop
        watching the remote pod's logs, the actual RunPod pods would
        keep running anyway. After this commit's mirror-script change,
        each mirror self-terminates within ~60 s of its remote pod
        being deleted, so leaving them alone is also self-cleaning.
        Use the runpod abilities (or the RunPod dashboard) to delete
        the actual pods.
    """
    from aii_lib.utils.tmux import (
        is_pipeline_run_session,
        kill_session,
        list_sessions,
        send_keys,
        session_exists,
    )

    # Fixed sessions. aii-storybook + aii-db-backup are spawned by aii_server
    # at startup (see aii_server.py / db_backup_supervisor.py), so stop_all
    # must clean them up too — they outlive aii-server itself otherwise.
    sessions = [
        PIPELINE_SESSION,
        SERVER_SESSION,
        "aii-dev-frontend",
        "claude_usage_persistent",
        "aii-storybook",
        "aii-db-backup",
    ]

    # Append pipeline-run sessions (``aii-<run_id>`` / ``aii-pending-<uuid>``)
    # started by either the launcher or the dashboard. Names are dynamic so
    # we can't enumerate them at module-import time — list_sessions is the
    # only source of truth. RunPod log mirrors are intentionally excluded
    # (see docstring).
    for name in list_sessions():
        if name in sessions:
            continue
        if is_pipeline_run_session(name):
            sessions.append(name)

    stopped = []
    for session in sessions:
        if not session_exists(session):
            continue
        # Send Ctrl+C (SIGINT) to process inside tmux → triggers cleanup handlers
        send_keys(session, "C-c", "")
        stopped.append(session)

    if not stopped:
        print("Nothing running")
        return 0

    # Wait briefly for processes to clean up
    time.sleep(3)

    # Kill any sessions that survived
    for session in stopped:
        if session_exists(session):
            kill_session(session)

    print(f"Stopped: {', '.join(stopped)}")
    return 0


async def stop_all_runpod() -> int:
    """Terminate every RunPod pod on the account, regardless of status.

    Talks to RunPod's REST API directly (no aii_server hop) so this works
    even if the local server isn't running. ``terminate_pod`` is the
    permanent delete — running, stopped, paused, exited pods all go.

    Local tmux mirrors for the deleted pods are NOT killed here — the
    mirror script self-terminates within ~60 s of its remote pod
    disappearing (see ``_spawn_local_log_mirror``'s post-connect failure
    budget).
    """
    try:
        import aii_runpod  # noqa: F401 — presence probe for public builds
    except ImportError:
        print(
            "ERROR: --stop-runpod-all requires the aii_runpod package, which is not\n"
            "       included in this build."
        )
        return 1

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        return 1

    from aii_runpod.deploy.runpod_api import RunPodAPI

    api = RunPodAPI(api_key)
    pods = await api.list_pods()
    if not pods:
        print("No RunPod pods to terminate")
        return 0

    print(f"Terminating {len(pods)} pod(s):")
    failures: list[str] = []
    for p in pods:
        pid = p.get("id", "?")
        name = p.get("name", "?")
        status = p.get("desiredStatus") or p.get("status") or "?"
        try:
            await api.terminate_pod(pid)
            print(f"  ✓ {pid}  {name}  (was {status})")
        except Exception as e:
            failures.append(pid)
            print(f"  ✗ {pid}  {name}  (was {status}) — {e}")

    if failures:
        print(f"FAILED: {len(failures)} pod(s) — {', '.join(failures)}")
        return 1
    print(f"Done — {len(pods)} pod(s) terminated")
    return 0


# ---------------------------------------------------------------------------
# Pipeline-only mode (used by aii_server to launch runs in tmux)
# ---------------------------------------------------------------------------


def deploy_pipeline_only(
    pipeline_args: list[str] | None = None,
    session_name: str | None = None,
) -> int:
    """Start just the pipeline in a tmux session. No server, no streaming.

    Used by aii_server's run API to launch pipeline runs in tmux sessions
    so they can be attached/monitored. Returns immediately after launch.
    """
    from aii_lib.utils.paths import logs_dir as _logs_dir
    from aii_lib.utils.tmux import (
        get_pane_pid,
        launch_in_tmux,
        pipeline_session_name,
    )

    # Per-run pipeline logs land under the persistent volume so they
    # survive pod restarts and sit beside the rest of the run's data.
    logs_dir = _logs_dir("runs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Derive run id (for orphan pattern + session-name fallback) from
    # pipeline args. Fork spawns supply ``--fork-new-run-id=<id>`` (id
    # pre-picked server-side via Run.gen_id); resume spawns supply
    # ``--resume-run-id=<id>``. Fresh-run spawns pass ``--session-name``
    # directly so this loop finds nothing and we fall back to it.
    run_id_for_orphan: str | None = None
    if pipeline_args:
        for i, arg in enumerate(pipeline_args):
            for prefix in ("--fork-new-run-id=", "--resume-run-id="):
                if arg.startswith(prefix):
                    run_id_for_orphan = arg.split("=", 1)[1]
                    break
                if arg == prefix.rstrip("=") and i + 1 < len(pipeline_args):
                    run_id_for_orphan = pipeline_args[i + 1]
                    break
            if run_id_for_orphan:
                break
    if not session_name and run_id_for_orphan:
        session_name = pipeline_session_name(run_id_for_orphan)
    if not session_name:
        session_name = PIPELINE_SESSION  # fallback

    # Build log file path from session name
    log_file = str(logs_dir / f"{session_name}.log")

    # ``--execute-mode=local`` is threaded down to ExecuteEnvConfig so this
    # pipeline runs artifacts in-process. ``deploy_pipeline_only`` is the
    # local-only path (aii_server's dashboard run API + ``aii_launcher
    # --pipeline-only``) — runpod orchs use ``scripts/runpod/run_pipeline.sh``
    # which hardcodes ``--execute-mode=runpod``.
    pipeline_cmd = f"{_py()} -m aii_pipeline.cli --execute-mode=local"
    if pipeline_args:
        pipeline_cmd += " " + " ".join(shlex.quote(a) for a in pipeline_args)

    # Scope orphan kill to *this run's* pipeline.cli process only.
    # Two levels of specificity:
    #   - "aii_pipeline.cli" component avoids matching the calling aii_launcher
    #     process (whose argv also contains the run-id flag) so we don't
    #     kill ourselves before tmux launches.
    #   - run-id component avoids killing concurrent forks for OTHER
    #     run ids that may be starting up at the same time.
    orphan_pat = f"aii_pipeline.cli.*-id={run_id_for_orphan}" if run_id_for_orphan else None
    launch_in_tmux(
        session=session_name,
        cmd=pipeline_cmd,
        log_file=log_file,
        cwd=str(PROJECT_ROOT),
        extra_env=_aii_pipeline_env(),
        orphan_pattern=orphan_pat,
    )

    # Capture the pane PID IMMEDIATELY before the pipeline can rename
    # its tmux session (cli.py renames ``aii-pending-<uuid>`` to the
    # canonical ``aii-<run_id>`` shape once Run.gen_id() returns).
    # If we waited for ``session_exists(session_name)`` after a delay,
    # the rename would race and we'd false-negative even when the
    # pipeline is healthy. Validating by PID-alive sidesteps the rename.
    pid = get_pane_pid(session_name)
    if pid is None:
        print(f"ERROR: pipeline session '{session_name}' failed to start", file=sys.stderr)
        return 1

    time.sleep(1)
    try:
        os.kill(pid, 0)  # signal 0 = "is the process alive?"
    except (OSError, ProcessLookupError):
        print(
            f"ERROR: pipeline pid {pid} died during boot (session '{session_name}')",
            file=sys.stderr,
        )
        return 1

    print(f"session={session_name} pid={pid} log={log_file}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _py() -> str:
    """Python executable path."""
    venv = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _aii_pipeline_env() -> dict[str, str]:
    """Env defaults shared by every aii server/pipeline tmux launch.

    Pipeline + server pin ``CLAUDE_CONFIG_DIR`` to a repo-local dir so AII
    state is independent of the user's personal ``~/.claude`` and so
    RunPod's NFS-mounted ``aii_data/`` shares sessions across pods.
    Personal terminal use without this env var stays in ``~/.claude``.

    ``AII_BUFFER_TRACE=1`` enables per-message lifecycle logging in
    ``_SummaryBuffer`` (writes JSONL to ``aii_data/logs/buffer_trace/<pid>.jsonl``).
    Diagnostic — leave on while investigating drain stalls.
    """
    return {
        "CLAUDE_CONFIG_DIR": str(PROJECT_ROOT / "aii_data" / ".claude"),
        "AII_BUFFER_TRACE": "1",
    }


def _server_already_healthy(port: int) -> bool:
    """One-shot probe: True if /agent_abilities/health returns 200 right now."""
    import httpx
    from aii_lib.utils.internal_auth import internal_headers

    try:
        r = httpx.get(
            f"http://localhost:{port}/agent_abilities/health",
            headers=internal_headers(),
            timeout=2,
        )
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_server(port: int, timeout: int = 180) -> bool:
    """Wait for aii_server health endpoint."""
    import httpx
    from aii_lib.utils.internal_auth import internal_headers
    from aii_lib.utils.tmux import session_exists

    for i in range(1, timeout + 1):
        try:
            r = httpx.get(
                f"http://localhost:{port}/agent_abilities/health",
                headers=internal_headers(),
                timeout=2,
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # Check tmux session still alive
        if not session_exists(SERVER_SESSION):
            print("  ERROR: aii_server tmux session died")
            return False
        if i % 15 == 0:
            print(f"  Still waiting... ({i}s)")
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for aii_launcher."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Deploy aii_server + aii_pipeline (locally or on RunPod)",
        prog="aii_launcher",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Deploy locally in tmux (default)",
    )
    group.add_argument("--runpod", action="store_true", help="Deploy to RunPod pods")
    group.add_argument("--resume", metavar="POD", help="Resume streaming from existing RunPod pod")
    group.add_argument(
        "--stop-local",
        action="store_true",
        help="Stop all local aii sessions (server, FE, pipeline runs). "
        "Leaves RunPod mirrors and actual RunPod pods alone.",
    )
    group.add_argument(
        "--stop-runpod-all",
        action="store_true",
        help="Terminate every RunPod pod on the account "
        "(running, stopped, paused — all). Does not touch local sessions.",
    )

    fe_group = parser.add_mutually_exclusive_group()
    fe_group.add_argument(
        "--dev-frontend",
        action="store_true",
        help=(
            "Explicitly select the dev Next.js frontend (Turbopack + HMR + "
            "Storybook). This is already the local-mode default; the flag "
            "exists so callers can be explicit and so the choice survives "
            "future default changes."
        ),
    )
    fe_group.add_argument(
        "--prod-frontend",
        action="store_true",
        help=(
            "Local-mode only: opt into the production Next.js frontend "
            "(``next build`` + ``next start``) instead of the dev server. "
            "``--runpod`` already defaults to the prod frontend, so this "
            "flag is a no-op there. First boot blocks ~1-2 min on the build."
        ),
    )
    fe_group.add_argument(
        "--no-frontend",
        action="store_true",
        help="Skip starting any Next.js frontend (server + pipeline only).",
    )
    parser.add_argument(
        "--server-only", action="store_true", help="Only start aii_server, not pipeline"
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Only start pipeline in tmux (no server)",
    )
    parser.add_argument(
        "--session-name", metavar="NAME", help="Tmux session name for --pipeline-only"
    )
    parser.add_argument(
        "--after", metavar="TEXT", help="With --resume: skip output before this text"
    )
    parser.add_argument(
        "--stop-local-runs",
        action="store_true",
        help=(
            "With --local: also kill all pipeline-run tmux sessions "
            "(aii-<run_id>) before launching. Off by default — local "
            "re-launches normally leave dashboard-spawned runs alone."
        ),
    )
    args, unknown = parser.parse_known_args()

    if args.runpod and args.pipeline_only:
        parser.error("--pipeline-only is not supported with --runpod (local mode only)")
    if args.stop_local_runs and (
        args.runpod or args.resume or args.stop_local or args.stop_runpod_all or args.pipeline_only
    ):
        parser.error("--stop-local-runs only applies to --local mode")

    if args.stop_local:
        sys.exit(stop_all())

    if args.stop_runpod_all:
        sys.exit(asyncio.run(stop_all_runpod()))

    if args.resume:
        sys.exit(asyncio.run(resume_stream(args.resume, args.after)))

    if args.runpod:
        # --runpod always means runpod end-to-end: orchestrator pod +
        # worker pods per substep. The shipped harness/execute_env.yaml is
        # rewritten to mode: runpod regardless of its on-disk value.
        # Local web app (prod build) is on by default so the dashboard works
        # alongside the remote pods; --no-frontend opts out.
        sys.exit(
            asyncio.run(
                deploy_runpod(
                    unknown or None,
                    server_only=args.server_only,
                    exec_mode="runpod",
                    prod_frontend=not args.no_frontend,
                )
            )
        )

    if args.pipeline_only:
        sys.exit(
            deploy_pipeline_only(
                pipeline_args=unknown or None,
                session_name=args.session_name,
            )
        )

    # Default: local. Dev frontend (Turbopack + HMR + Storybook) is the
    # default for fast iteration; --dev-frontend is the explicit form,
    # --prod-frontend opts into the optimized ``next build``+``next start``
    # boot, --no-frontend opts out entirely.
    dev_frontend = args.dev_frontend or (not args.prod_frontend and not args.no_frontend)
    sys.exit(
        deploy_local(
            dev_frontend=dev_frontend,
            prod_frontend=args.prod_frontend,
            server_only=args.server_only,
            pipeline_args=unknown or None,
            stop_local_runs=args.stop_local_runs,
        )
    )


if __name__ == "__main__":
    main()
