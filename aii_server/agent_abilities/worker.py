"""Worker process for ability endpoints.

Each ability gets a persistent worker process that:
1. Activates the ability's declared venv (site-packages)
2. Runs worker_init once (imports packages, creates session pools)
3. Handles concurrent requests via ThreadPoolExecutor
4. Communicates with Django via multiprocessing.Queue

This gives true venv isolation + parallelism without the GIL.
"""

import asyncio
import faulthandler
import os
import signal
import site
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process, Queue
from multiprocessing.managers import SyncManager  # noqa: TC003 — runtime instance-attr annotation
from pathlib import Path
from queue import Empty

from aii_lib.abilities.ability_server.logging_config import (
    _CRASH_LOG_DIR,
    DEFAULT_TIMEOUT,
)
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception as e:
        logger.debug(f"_get_memory_mb failed: {e}")
        return 0.0


def _write_crash_log(name: str, error_type: str, message: str, tb: str = "") -> None:
    """Write a crash log file for post-mortem debugging."""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = _CRASH_LOG_DIR / f"{name}_{error_type}_{ts}_{os.getpid()}.log"
        log_file.write_text(
            f"Error: {error_type}\nMessage: {message}\nPID: {os.getpid()}\n\n{tb}\n"
        )
    except Exception as e:
        logger.debug(f"_write_crash_log failed for {name}/{error_type}: {e}")


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _worker_process(
    name: str,
    handler: Callable[[dict], dict],
    worker_init: Callable[[], None] | None,
    request_queue: Queue,
    response_queues: dict,
    max_workers: int,
    venv_path: str | None = None,
):
    """Long-lived worker process for one ability endpoint."""
    log = logger.bind(source=name)

    # Activate venv site-packages so imports resolve from the ability's venv
    if venv_path:
        venv = Path(venv_path)
        for sp in sorted(venv.glob("lib/python*/site-packages")):
            site.addsitedir(str(sp))
            log.debug(f"Activated venv: {sp}")
            break

    faulthandler.enable()

    # Signal handling for clean shutdown
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.warning(f"Received {sig_name} (pid={os.getpid()})")
        _write_crash_log(name, "SIGNAL", f"Received {sig_name}", traceback.format_stack(frame))
        sys.exit(128 + signum)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        try:
            signal.signal(sig, signal_handler)
        except Exception as e:
            log.warning(f"Failed to register signal handler for {sig}: {e}")

    log.debug(f"Worker started (pid={os.getpid()}, mem={_get_memory_mb():.1f}MB)")

    # Run worker_init once (imports packages, creates session pools, etc.)
    if worker_init:
        try:
            start = time.perf_counter()
            worker_init()
            elapsed = time.perf_counter() - start
            log.debug(f"Init complete ({elapsed:.2f}s, mem={_get_memory_mb():.1f}MB)")
        except Exception as e:
            tb = traceback.format_exc()
            log.exception(f"Init FAILED: {e}\n{tb}")
            _write_crash_log(name, "INIT_ERROR", str(e), tb)
            raise RuntimeError(f"Init failed for {name}: {e}") from e

    # Thread pool for concurrent requests within this process
    executor = ThreadPoolExecutor(max_workers=max_workers)
    request_count = 0
    error_count = 0

    def handle_request(request_id: str, request_dict: dict):
        nonlocal request_count, error_count
        request_count += 1
        req_start = time.perf_counter()
        try:
            result = handler(request_dict)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            elapsed = time.perf_counter() - req_start
            log.debug(f"REQ#{request_count} ({elapsed:.2f}s)")
            return request_id, result
        except Exception as e:
            error_count += 1
            elapsed = time.perf_counter() - req_start
            tb = traceback.format_exc()
            log.exception(f"REQ#{request_count} FAILED ({elapsed:.2f}s): {e}\n{tb}")
            _write_crash_log(name, "HANDLER_ERROR", str(e), tb)
            return request_id, {"success": False, "error": str(e), "traceback": tb}

    def send_response(future):
        try:
            request_id, result = future.result()
            if request_id in response_queues:
                response_queues[request_id].put(result)
        except Exception as e:
            log.exception(f"Failed to send response: {e}")

    log.debug(f"Worker ready (max_workers={max_workers}, pid={os.getpid()})")

    # Main loop — read requests from queue, dispatch to thread pool
    while True:
        try:
            try:
                msg = request_queue.get(timeout=60)
            except Empty:
                # Heartbeat log
                continue

            # Shutdown signal
            if msg is None:
                log.info(f"Shutdown (reqs={request_count}, errs={error_count})")
                executor.shutdown(wait=True)
                break

            request_id, request_dict, response_queue = msg
            response_queues[request_id] = response_queue
            future = executor.submit(handle_request, request_id, request_dict)
            future.add_done_callback(send_response)

        except Exception as e:
            tb = traceback.format_exc()
            log.exception(f"Main loop error: {e}\n{tb}")
            _write_crash_log(name, "MAIN_LOOP_ERROR", str(e), tb)


# ---------------------------------------------------------------------------
# WorkerHandle — manages one worker process from the Django side
# ---------------------------------------------------------------------------


class WorkerHandle:
    """Handle to a persistent worker process for one ability."""

    def __init__(
        self,
        name: str,
        handler: Callable[[dict], dict],
        worker_init: Callable[[], None] | None,
        max_workers: int = 10,
        venv_path: str | None = None,
    ):
        self.name = name
        self.handler = handler
        self.worker_init = worker_init
        self.max_workers = max_workers
        self.venv_path = venv_path
        self.request_queue: Queue | None = None
        self.response_queues: dict | None = None
        self.process: Process | None = None
        self._manager: SyncManager | None = None
        self._restart_count = 0
        self._log = logger.bind(source=name)

    def start(self):
        """Start the worker process (idempotent)."""
        if self.process is not None and self.process.is_alive():
            return

        # Clean up old manager if restarting
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception as e:
                self._log.debug(f"Old manager shutdown during restart failed: {e}")

        self._manager = Manager()
        self.response_queues = self._manager.dict()
        self.request_queue = Queue()

        self.process = Process(
            target=_worker_process,
            args=(
                self.name,
                self.handler,
                self.worker_init,
                self.request_queue,
                self.response_queues,
                self.max_workers,
                self.venv_path,
            ),
            daemon=True,
        )
        self.process.start()
        self._log.debug(f"Worker started (pid={self.process.pid}, restarts={self._restart_count})")

    def stop(self):
        """Gracefully stop the worker process."""
        if self.process is None:
            return

        self._log.info(f"Stopping worker (pid={self.process.pid})...")
        try:
            self.request_queue.put(None)  # Shutdown signal
            self.process.join(timeout=5)
        except Exception as e:
            self._log.debug(f"Shutdown signal/join failed, will force-terminate: {e}")

        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)

        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)

        self.process = None

        if self._manager:
            try:
                self._manager.shutdown()
            except Exception as e:
                self._log.debug(f"Manager shutdown failed during stop: {e}")
            self._manager = None

        self._log.info("Worker stopped")

    def _ensure_alive(self):
        """Restart worker if it died."""
        if self.process is not None and self.process.is_alive():
            return

        if self.process is not None:
            exitcode = self.process.exitcode
            self._log.warning(f"Worker died (exitcode={exitcode}), restarting...")
            _write_crash_log(self.name, "WORKER_DEATH", f"exitcode={exitcode}")
            self._restart_count += 1

        self.start()

    async def call(self, request: dict, timeout: float | None = None) -> dict:
        """Send a request to the worker and await the response."""
        if timeout is None:
            timeout = DEFAULT_TIMEOUT

        self._ensure_alive()

        if self._manager is None:
            return {"success": False, "error": "Worker manager not initialized"}

        try:
            response_queue = self._manager.Queue()
        except Exception as e:
            # Manager died — restart everything
            self._log.warning(f"Manager error, restarting: {e}")
            self.stop()
            self.start()
            await asyncio.sleep(0.5)
            try:
                response_queue = self._manager.Queue()
            except Exception as e2:
                return {
                    "success": False,
                    "error": f"Failed to create response queue: {e2}",
                }

        request_id = str(uuid.uuid4())
        try:
            self.request_queue.put((request_id, request, response_queue))
        except Exception as e:
            return {"success": False, "error": f"Failed to send request: {e}"}

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, response_queue.get, True, timeout),
                timeout=timeout,
            )
            return result
        except TimeoutError:
            self._log.warning(f"Request timed out ({timeout}s)")
            return {"success": False, "error": f"Request timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Global worker registry
# ---------------------------------------------------------------------------

_workers: dict[str, WorkerHandle] = {}
_workers_lock = threading.Lock()


def get_or_create_worker(
    name: str,
    handler: Callable[[dict], dict],
    worker_init: Callable[[], None] | None,
    max_workers: int = 10,
    venv_path: str | None = None,
) -> WorkerHandle:
    """Get existing worker or create and start a new one."""
    # Fast path: lock-free read for the common case (worker already exists).
    handle = _workers.get(name)
    if handle is not None:
        return handle
    # Slow path: lock + double-check to avoid racing two concurrent first-requests.
    with _workers_lock:
        handle = _workers.get(name)
        if handle is None:
            handle = WorkerHandle(
                name=name,
                handler=handler,
                worker_init=worker_init,
                max_workers=max_workers,
                venv_path=venv_path,
            )
            _workers[name] = handle
    return handle


def start_all_workers():
    """Start all registered workers. Called at server boot."""
    for w in _workers.values():
        w.start()
