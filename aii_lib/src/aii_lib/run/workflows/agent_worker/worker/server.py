"""Worker pod aiohttp server — the in-pod runtime for ``agent_worker``.

Receives one ``POST /job`` payload, runs an agent, streams its events
out via ``GET /telemetry`` SSE, returns the AgentResponse via
``GET /result`` once finished. Two channels by design — see the
parent package docstring.

Routes (matching the long-standing project convention so existing
orchestrator clients keep working unchanged):

  - ``GET  /http_health``   — liveness probe (cheap)
  - ``GET  /system_health`` — cached OOM-detector state
  - ``POST /job``           — receive job envelope, kick off agent
  - ``GET  /result``        — poll for AgentResponse (204 if pending)
  - ``GET  /error``         — poll for error envelope (204 if none)
  - ``GET  /telemetry``     — SSE stream of typed Run events
  - ``GET  /debug``         — diagnostic dump (last events, traceback, …)
  - ``POST /cancel``        — graceful cancel + shutdown

Internals: each ``POST /job`` constructs a fresh :class:`aii_lib.run.run.Run`
keyed by ``task_id``, subscribes a private buffer sink, and runs the
agent against it (events flow via the ambient ``current_run()``).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from ....context import set_current_run
from ....run import Run

if TYPE_CHECKING:
    from ....messages import BaseMessage


# Port the in-pod aiohttp listens on. Matches the port the docker
# template publishes via the RunPod proxy URL scheme
# ``{pod_id}-{port}.proxy.runpod.net``.
WORKER_PORT = 8080


# Max SSE data payload (bytes). aiohttp's StreamReader default limit
# is 65536; keep well under to avoid "Chunk too big" on the
# orchestrator's SSE reader.
MAX_SSE_DATA = 48_000


# =====================================================================
# Internal: per-job event buffer + sink
# =====================================================================


def _event_to_wire(event: BaseMessage) -> dict:
    """Project a typed Run event into the wire dict shape consumers expect."""
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    if hasattr(event, "to_dict"):
        return event.to_dict()
    if isinstance(event, dict):
        return event
    return {"type": getattr(event, "type", "unknown"), "raw": str(event)}


class _EventBuffer:
    """Thread-safe append-only buffer of wire dicts for SSE serving.

    Holds every typed event the host Run emits. Append-only by design;
    consumers replay the full list on first connect and resume via
    ``?since=N`` on reconnect.
    """

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def append(self, event_dict: dict) -> None:
        with self._lock:
            self._events.append(event_dict)

    def since(self, index: int) -> tuple[list[dict], int]:
        with self._lock:
            return self._events[index:], len(self._events)


class _BufferSink:
    """Minimal RunSink that pushes every event onto an :class:`_EventBuffer`."""

    def __init__(self, buffer: _EventBuffer) -> None:
        self._buffer = buffer

    def flush(self, event: BaseMessage) -> None:
        self._buffer.append(_event_to_wire(event))

    def close(self) -> None:
        return None

    def map(self, event: Any) -> Any:
        return event


def _encode_msg(msg: dict) -> bytes:
    """Serialize one event as an SSE frame, hard-truncating if oversized."""
    data = json.dumps(msg, default=str)
    if len(data) > MAX_SSE_DATA:
        truncated = dict(msg)
        text = truncated.get("text", "")
        if isinstance(text, str) and len(text) > 500:
            truncated["text"] = text[:500] + f"... [truncated, {len(text)} chars total]"
            data = json.dumps(truncated, default=str)
        if len(data) > MAX_SSE_DATA:
            data = data[: MAX_SSE_DATA - 50] + ',"_truncated":true}'
    return f"event: telemetry\ndata: {data}\n\n".encode()


# =====================================================================
# Routes
# =====================================================================


async def handle_http_health(request: web.Request) -> web.Response:
    """Lightweight readiness probe — only proves aiohttp is alive."""
    return web.json_response({"status": "ok"})


async def handle_system_health(request: web.Request) -> web.Response:
    """Return cached system health state (OOM detector).

    The actual check runs in a background loop (``_system_health_loop``)
    so this endpoint is instant.
    """
    healthy = request.app.get("system_healthy")
    if healthy is None:
        return web.json_response({"status": "pending"})
    if healthy:
        return web.json_response({"status": "ok"})
    return web.json_response({"status": "unhealthy"}, status=503)


async def _run_system_check() -> bool:
    """Spawn a trivial subprocess. True if the system is healthy.

    Under OOM the kernel can't fork — ``create_subprocess_exec`` either
    raises or hangs until the wait_for timeout fires.
    """
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "python3",
                "-c",
                "x=bytearray(10*1024*1024);print('ok')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=10,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


async def _system_health_loop(app: web.Application) -> None:
    """Periodically verify the system can spawn processes; surface OOM."""
    try:
        await asyncio.sleep(30)
        while not app.get("job_done"):
            healthy = await _run_system_check()
            app["system_healthy"] = healthy
            if not healthy:
                if (
                    not app.get("error")
                    and not app.get("result")
                    and not app.get("_result_staging")
                ):
                    app["error"] = {
                        "error_type": "SystemHealthCheckFailed",
                        "error_message": (
                            "System health check failed: unable to spawn subprocess. "
                            "The system is likely in an OOM state."
                        ),
                        "traceback": "",
                    }
                    app["job_done"] = True
                return
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass


async def _fetch_credentials_from_server() -> None:
    """Fetch Claude OAuth credentials from the ability server.

    For RunPod worker pods: the server pod hosts the ability server
    (``AII_SERVER_URL``) and serves the OAuth tokens via HTTP. The
    worker pulls them at job start so the Claude SDK can authenticate.
    For local testing this is a no-op — set ``CLAUDE_CONFIG_DIR`` to
    a directory with valid creds instead.
    """
    try:
        from aii_lib.server_url import ability_service_url
    except ImportError:
        return
    ability_url = ability_service_url()
    if not ability_url:
        return  # local mode — caller should have set CLAUDE_CONFIG_DIR
    try:
        from aii_lib.llm_backend.claude_max.autologin import _fetch_credentials_remote

        _fetch_credentials_remote(ability_url)
    except ImportError:
        pass  # autologin not available — agent will fall back to env


async def _run_agent(
    job: dict,
    workspace_dir: Path,
    buffer: _EventBuffer,
) -> dict:
    """Run the agent job; return the AgentResponse as a dict."""
    from aii_lib.agent_backend import Agent, AgentOptions

    # RunPod path: pull OAuth creds from the ability server before the
    # SDK tries to authenticate. Local path: no-op (CLAUDE_CONFIG_DIR
    # already points at local creds).
    await _fetch_credentials_from_server()

    # Per-job Run: the worker's event scope. Subscribe the buffer sink
    # so every event the agent emits via ``current_run()`` lands in
    # the buffer that ``handle_telemetry`` serves over SSE.
    run = Run(node_id=job["task_id"])
    run.subscribe_sink(_BufferSink(buffer))
    set_current_run(run)

    options_dict = job["agent_options"]
    options_dict["cwd"] = str(workspace_dir)
    options_dict["run_id"] = job["task_id"]
    options_dict["agent_context"] = job["task_name"]

    # Coerce session_type enum if present
    if "session_type" in options_dict and isinstance(options_dict["session_type"], str):
        from aii_lib.agent_backend.claude_agent_sdk.models.enums import SessionType

        options_dict["session_type"] = SessionType(options_dict["session_type"])

    # Optional post-validate hook
    validation = job.get("validation")
    if validation:
        import importlib

        mod = importlib.import_module(validation["module"])
        make_validator = mod.make_post_validator
        options_dict["post_validate"] = make_validator(
            artifact_type=validation["artifact_type"],
            workspace_dir=str(workspace_dir),
            min_examples=validation.get("min_examples", 3),
            max_file_size_mb=validation.get("max_file_size_mb", 100),
        )
        options_dict["post_validate_retries"] = validation.get("schema_retries", 2)

    options = AgentOptions(**options_dict)
    response = await Agent(options).run(job["prompts"])
    return response.to_dict() if hasattr(response, "to_dict") else dict(response)


async def handle_job(request: web.Request) -> web.Response:
    """Receive a job and kick off the agent in the background."""
    app = request.app
    if app["job_running"]:
        return web.json_response(
            {"error": "A job is already running"},
            status=409,
        )

    # Cross-process OTel context propagation
    try:
        from opentelemetry import context as otel_context
        from opentelemetry.propagate import extract

        otel_context.attach(extract(dict(request.headers)))
    except ImportError:
        pass

    job_data = await request.json()
    app["job_running"] = True
    app["telemetry_buffer"] = _EventBuffer()

    workspace_rel = job_data.get("workspace_rel")
    task_id = job_data.get("task_id")
    if workspace_rel:
        workspace_dir = Path(app["workspace_root"]) / workspace_rel
    elif task_id:
        workspace_dir = Path(app["workspace_root"]) / "runs" / task_id
    else:
        workspace_dir = Path(app["workspace_root"]) / "runs" / "default"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    async def _drive() -> None:
        try:
            result = await _run_agent(job_data, workspace_dir, app["telemetry_buffer"])
            app["_result_staging"] = result
        except Exception as exc:
            app["error"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            app["job_running"] = False
            app["job_done"] = True

    app["agent_task"] = asyncio.create_task(_drive())
    app["system_health_task"] = asyncio.create_task(_system_health_loop(app))
    return web.json_response({"status": "started"})


async def handle_result(request: web.Request) -> web.Response:
    """Return the AgentResponse JSON (waits for SSE to flush first)."""
    app = request.app
    if not app.get("job_done"):
        return web.Response(status=204)

    # Wait for SSE handler to drain the buffer + emit `event: done`
    try:
        await asyncio.wait_for(app["sse_flushed"].wait(), timeout=10.0)
    except TimeoutError:
        # SSE never connected or stalled — promote the staged result anyway
        if app.get("_result_staging") is not None:
            app["result"] = app.pop("_result_staging")
        app["sse_flushed"].set()

    result = app.get("result")
    if result is None:
        return web.Response(status=204)
    return web.json_response(result)


async def handle_error(request: web.Request) -> web.Response:
    """Return the error envelope (204 if none)."""
    error = request.app.get("error")
    if error is None:
        return web.Response(status=204)
    return web.json_response(error)


async def handle_telemetry(request: web.Request) -> web.StreamResponse:
    r"""SSE stream of typed Run events.

    Wire format:
      - ``event: telemetry\\ndata: <json>\\n\\n`` per event
      - ``event: heartbeat\\ndata: {}\\n\\n`` keep-alive every 5s
      - ``event: done\\ndata: {}\\n\\n`` once the agent finishes + buffer drains

    Resume: ``?since=N`` query param.
    """
    app = request.app
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    try:
        index = int(request.query.get("since", 0))
    except (ValueError, TypeError):
        index = 0
    heartbeat_interval = 5.0
    poll_interval = 0.3
    elapsed = 0.0

    try:
        while True:
            buf: _EventBuffer | None = app.get("telemetry_buffer")
            if buf is not None:
                pending, new_index = buf.since(index)
                for msg in pending:
                    await response.write(_encode_msg(msg))
                index = new_index

            elapsed += poll_interval
            if elapsed >= heartbeat_interval:
                elapsed = 0.0
                await response.write(b"event: heartbeat\ndata: {}\n\n")

            if app.get("job_done"):
                # Flush remaining events, write done, promote staged result.
                if buf is not None:
                    pending, _ = buf.since(index)
                    for msg in pending:
                        await response.write(_encode_msg(msg))
                await response.write(b"event: done\ndata: {}\n\n")
                if app.get("_result_staging") is not None:
                    app["result"] = app.pop("_result_staging")
                app["sse_flushed"].set()
                break

            await asyncio.sleep(poll_interval)
    except (ConnectionResetError, ConnectionAbortedError, asyncio.CancelledError):
        # Consumer disconnected — release any /result waiter
        pass
    finally:
        if app.get("_result_staging") is not None:
            app["result"] = app.pop("_result_staging")
        app["sse_flushed"].set()

    return response


async def handle_debug(request: web.Request) -> web.Response:
    """Diagnostic dump — last events, traceback, system info."""
    app = request.app
    buf: _EventBuffer | None = app.get("telemetry_buffer")
    last_events: list[dict] = []
    if buf is not None:
        all_events, _ = buf.since(0)
        last_events = all_events[-20:]
    return web.json_response(
        {
            "job_running": app.get("job_running", False),
            "job_done": app.get("job_done", False),
            "has_result": app.get("result") is not None or app.get("_result_staging") is not None,
            "has_error": app.get("error") is not None,
            "last_events": last_events,
            "system_healthy": app.get("system_healthy"),
        }
    )


async def handle_cancel(request: web.Request) -> web.Response:
    """Graceful shutdown — cancel the agent, schedule SIGTERM."""
    app = request.app
    if app.get("result") or app.get("_result_staging"):
        return web.json_response({"cancelled": False, "reason": "job already completed"})

    task = app.get("agent_task")
    if task and not task.done():
        task.cancel()

    app["error"] = {
        "error_type": "Cancelled",
        "error_message": "Job cancelled by orchestrator",
        "traceback": "",
    }
    app["job_done"] = True

    asyncio.get_running_loop().call_later(
        1.0,
        lambda: os.kill(os.getpid(), signal.SIGTERM),
    )
    return web.json_response({"cancelled": True})


# =====================================================================
# App factory
# =====================================================================


def create_app(workspace_root: str = "/tmp/aii-worker") -> web.Application:
    """Build the in-pod aiohttp application with all routes + state.

    ``workspace_root`` is the base path where the worker resolves
    ``workspace_rel`` from each job envelope. On RunPod this is the
    network-volume mount; locally any tmpdir works.
    """
    app = web.Application()

    # Server state
    app["workspace_root"] = workspace_root
    app["job_running"] = False
    app["job_done"] = False
    app["result"] = None
    app["_result_staging"] = None
    app["error"] = None
    app["telemetry_buffer"] = None
    app["agent_task"] = None
    app["system_health_task"] = None
    app["system_healthy"] = None
    app["sse_flushed"] = asyncio.Event()

    app.router.add_get("/http_health", handle_http_health)
    app.router.add_get("/system_health", handle_system_health)
    app.router.add_post("/job", handle_job)
    app.router.add_get("/result", handle_result)
    app.router.add_get("/error", handle_error)
    app.router.add_get("/telemetry", handle_telemetry)
    app.router.add_get("/debug", handle_debug)
    app.router.add_post("/cancel", handle_cancel)

    return app


__all__ = ["WORKER_PORT", "create_app"]
