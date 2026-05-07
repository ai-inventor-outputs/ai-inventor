"""Abstract base for exec envs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aii_lib.agent_backend import AgentOptions, AgentResponse
    from aii_lib.run import Run


@dataclass
class ComputeProfile:
    """Resource profile for agent execution.

    Base class — exec envs extend with their own fields.
    ``LocalEnv`` uses this directly (no extra config needed).
    ``RunPodEnv`` reads ``RunPodComputeProfile`` (Pydantic) directly off
    the loaded ``RunPodConfig`` instead of mirroring it as a dataclass.
    """

    name: str  # "local", "gpu", "cpu_heavy", "cpu_light"


class ExecuteEnv(ABC):
    """Abstract base for exec envs.

    Subclasses implement ``run_agent()`` to execute agent work on a specific
    infrastructure (local process, RunPod, etc.). The signature is the same
    on both — RunPod-only kwargs (``plan``, ``compute_profile``,
    ``pod_timeout``, ``pod_start_retries``) are silently ignored by
    ``LocalEnv`` so the call site can be uniform.
    """

    @abstractmethod
    async def run_agent(
        self,
        options: AgentOptions,
        prompts: list[str] | str,
        *,
        plan: Any | None = None,
        compute_profile: str = "cpu_light",
        pod_timeout: int | None = None,
        pod_start_retries: int | None = None,
        validation: dict | None = None,
        run: Run | None = None,
    ) -> AgentResponse:
        """Run an agent and return its response.

        Args:
            options: Agent configuration.
            prompts: Prompt(s) to send to the agent.
            plan: Pipeline ``BasePlan`` (for ``runpod_compute_profile``).
                Local ignores. RunPod uses ``plan.runpod_compute_profile``
                when present, falling back to ``compute_profile``.
            compute_profile: Profile name when no ``plan`` is supplied.
            pod_timeout: Override for RunPod per-pod timeout.
            pod_start_retries: Override for RunPod startup retries.
            validation: Optional post-run validation config (artifact_type,
                schema_retries, min_examples, max_file_size_mb, …).
            run: Host :class:`Run` — passed through to the underlying
                dispatcher for event routing.
        """
        ...
