"""``LocalAgentDispatcher`` — in-process agent_worker dispatcher.

Trivial: instantiates the agent and awaits ``agent.run(prompts)``. The
agent's normal emit path goes through ``current_run()``, so events
land on the host Run automatically — no transport, no batch replay.

Used as the default branch when ``config.execute_env.mode == "local"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....run import Run


class LocalAgentDispatcher:
    """In-process :class:`AgentDispatcher`. Events flow via current_run()."""

    async def dispatch(
        self,
        options: Any,
        prompts: list[str] | str,
        *,
        run: Run,
    ) -> Any:
        """Run one agent in this process and return its response.

        ``run`` is accepted for protocol symmetry but isn't explicitly
        wired — the agent emits via the ambient ``current_run()``,
        which is set by the pipeline boot to this same Run.
        """
        from aii_lib.agent_backend import Agent

        return await Agent(options).run(prompts)


__all__ = ["LocalAgentDispatcher"]
