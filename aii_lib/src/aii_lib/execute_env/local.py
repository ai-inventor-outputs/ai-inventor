"""Local exec env — runs agents in the current process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aii_lib.remote.contracts import DEFAULT_MAX_FILE_SIZE_MB
from aii_lib.run.workflows.agent_worker import LocalAgentDispatcher

from .base import ExecuteEnv

if TYPE_CHECKING:
    from aii_lib.agent_backend import AgentOptions, AgentResponse
    from aii_lib.run import Run


def _attach_post_validate(options: AgentOptions, validation: dict) -> None:
    """Wire ``validation`` config into ``options.post_validate`` for local dispatch.

    For RunPod dispatch the validation dict is shipped to the worker inside
    the job envelope and the worker constructs the validator there — see
    ``aii_lib.run.workflows.agent_worker.worker.server._run_agent``.
    """
    import importlib

    mod = importlib.import_module(validation["module"])
    options.post_validate = mod.make_post_validator(
        artifact_type=validation["artifact_type"],
        workspace_dir=str(options.cwd),
        min_examples=validation.get("min_examples", 3),
        max_file_size_mb=validation.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB),
    )
    options.post_validate_retries = validation.get("schema_retries", 2)


class LocalEnv(ExecuteEnv):
    """Run agents directly in the current process via :class:`LocalAgentDispatcher`."""

    async def run_agent(
        self,
        options: AgentOptions,
        prompts: list[str] | str,
        *,
        plan: Any | None = None,  # noqa: ARG002 — RunPod-only, accepted for protocol
        compute_profile: str = "cpu_light",  # noqa: ARG002 — RunPod-only
        pod_timeout: int | None = None,  # noqa: ARG002 — RunPod-only
        pod_start_retries: int | None = None,  # noqa: ARG002 — RunPod-only
        validation: dict | None = None,
        run: Run | None = None,
    ) -> AgentResponse:
        """Dispatch one agent in-process. Wires ``validation`` onto ``options``."""
        if validation:
            _attach_post_validate(options, validation)
        return await LocalAgentDispatcher().dispatch(options, prompts, run=run)
