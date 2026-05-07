"""``AgentInputDispatcher`` — the input contract for the agent_worker workflow.

Two shapes share this protocol (in ``channel/``):

  - :class:`LocalAgentDispatcher` — runs the agent in the caller's
    process; events flow into the ambient ``current_run()`` via the
    agent's normal emit path.

  - :class:`RunPodAgentDispatcher` — spins up an ephemeral worker pod
    via ``aii_runpod.WorkerPod``, ships the ``AgentOptions`` over HTTP
    to the in-pod server (``worker/server.py``), and the worker's
    events stream back via the ``GET /telemetry`` SSE channel.

Both are interchangeable from the caller's perspective. Callers pick
one based on the deployment context (``config.execute_env.mode``) and
dispatch one job through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...run import Run


@runtime_checkable
class AgentInputDispatcher(Protocol):
    """Run one agent dispatch and yield the AgentResponse.

    Events emitted by the agent are surfaced via ``run`` (the live host
    Run on the caller's side). Local impls emit directly via the
    ambient ``current_run()``; remote impls bridge events from a worker
    process to ``run`` via SSE.
    """

    async def dispatch(
        self,
        options: Any,  # AgentOptions — typing avoided to keep aii_lib import-free of the agent backend
        prompts: list[str] | str,
        *,
        run: Run,
    ) -> Any:
        """Dispatch one agent run; return the resulting AgentResponse."""
        ...


__all__ = ["AgentInputDispatcher"]
