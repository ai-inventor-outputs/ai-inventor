"""Worker side of the ``agent_worker`` workflow — the in-pod HTTP server.

This is the program that runs *inside* a worker pod (or any container
hosting one agent dispatch). It's deployment-agnostic: aii_runpod
handles RunPod-specific pod creation, but the server itself doesn't
know whether it's on RunPod, k8s, docker-compose, etc.

Entry point: :func:`create_app` returns an aiohttp ``web.Application``
ready to be served. A small CLI wrapper (e.g. ``aii_runpod.comms.entrypoint``)
binds it to a port and adds a self-destruct watchdog.
"""

from .server import WORKER_PORT, create_app

__all__ = ["WORKER_PORT", "create_app"]
