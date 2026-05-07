"""``agent_worker`` — run an agent, stream its events back onto a Run.

One entry contract (:class:`AgentDispatcher`) with two channels in
``channel/``:

  - **Local** (:class:`LocalAgentDispatcher`) — agent runs in the
    caller's process; events flow into the host Run / DBOS journal
    directly via the agent's normal ``aii_lib.run.emit`` calls.

  - **RunPod** (:class:`RunPodAgentDispatcher`) — agent runs in an
    ephemeral worker pod; events stream back via ``GET /telemetry``
    SSE through the worker server in ``worker/server.py``.

Input direction is one-shot: the worker job is a single ``POST /job``.
SSE is only used for the output direction (events back from the
worker), where it pays for itself with live progress.

Submodules:

  - :mod:`.input` — :class:`AgentInputDispatcher` Protocol (the input contract).
  - :mod:`.channel` — concrete dispatchers (local / runpod).
  - :mod:`.worker` — worker-side aiohttp server (``create_app``).
"""

from .channel import LocalAgentDispatcher, RunPodAgentDispatcher
from .input import AgentInputDispatcher

__all__ = [
    "AgentInputDispatcher",
    "LocalAgentDispatcher",
    "RunPodAgentDispatcher",
]
