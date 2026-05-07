"""Pipeline ↔ exec-env entry point — picks Local or RunPod per call.

Reads ``config.execute_env.mode`` and constructs the right
:class:`ExecuteEnv` (``LocalEnv`` or ``RunPodEnv``), then awaits
``env.run_agent(options, prompts, ...)``. The env owns the
pipeline-domain → transport-domain resolution; this module is just a
thin wrapper that returns ``(None, AgentResponse)`` to keep the
long-standing ``agent, result = await create_and_run_agent(...)``
destructure pattern working at every callsite (the agent handle was
already unused downstream).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aii_lib.execute_env import LocalEnv
from aii_lib.run import current_run

if TYPE_CHECKING:
    from aii_lib.agent_backend import AgentOptions, AgentResponse

    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
    )
    from aii_pipeline.utils import PipelineConfig


def _make_env(config: PipelineConfig):
    """Construct the right :class:`ExecuteEnv` for ``config.execute_env.mode``."""
    if config.execute_env.mode == "runpod":
        from aii_runpod.runpod_backend import RunPodEnv

        return RunPodEnv(
            config.execute_env.runpod,
            retry_context_truncate_chars=config.init.retry_context.truncate_chars,
            retry_context_messages=config.init.retry_context.messages,
        )
    return LocalEnv()


async def create_and_run_agent(
    options: AgentOptions,
    prompts: list[str] | str,
    config: PipelineConfig,
    plan: BasePlan,
    pod_timeout: int | None = None,
    pod_start_retries: int | None = None,
    validation: dict | None = None,
) -> tuple[Any, AgentResponse]:
    """Dispatch one agent job — local or RunPod, picked from config."""
    env = _make_env(config)
    response = await env.run_agent(
        options,
        prompts,
        plan=plan,
        pod_timeout=pod_timeout,
        pod_start_retries=pod_start_retries,
        validation=validation,
        run=current_run(),
    )
    return None, response


async def create_and_run_agent_simple(
    options: AgentOptions,
    prompts: list[str] | str,
    config: PipelineConfig,
    compute_profile: str = "cpu_light",
    pod_timeout: int | None = None,
    pod_start_retries: int | None = None,
) -> tuple[Any, AgentResponse]:
    """No-BasePlan variant — ``compute_profile`` named directly by the caller."""
    env = _make_env(config)
    response = await env.run_agent(
        options,
        prompts,
        compute_profile=compute_profile,
        pod_timeout=pod_timeout,
        pod_start_retries=pod_start_retries,
        run=current_run(),
    )
    return None, response
