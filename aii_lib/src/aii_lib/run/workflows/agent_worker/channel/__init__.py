"""Concrete :class:`AgentDispatcher` implementations — one per channel.

- :class:`LocalAgentDispatcher` — in-process; events flow via the
  ambient ``current_run()``. No transport.
- :class:`RunPodAgentDispatcher` — runs the agent in an ephemeral
  RunPod worker pod; events come back as a single batch in the
  ``/result`` payload and get replayed onto the host Run.
"""

from .local import LocalAgentDispatcher
from .runpod import RunPodAgentDispatcher

__all__ = ["LocalAgentDispatcher", "RunPodAgentDispatcher"]
