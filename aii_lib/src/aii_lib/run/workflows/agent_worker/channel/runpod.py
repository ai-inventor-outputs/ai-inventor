"""``RunPodAgentDispatcher`` — runs one agent in an ephemeral RunPod worker pod, streams events back via SSE.

Lifecycle per dispatch:

  1. Boot a fresh worker pod via ``aii_runpod.WorkerPod`` against the
     pre-resolved template + hardware ID (RunPod API, healthcheck —
     all the existing transport machinery in aii_runpod).
  2. Submit the agent job to the pod over HTTP (``POST /job``).
  3. Open an SSE consumer against the pod's ``/telemetry`` URL; each
     frame is parsed via ``parse_message`` and delivered onto the host
     ``run`` through ``run._on``.
  4. Poll ``/result`` until the agent finishes (or fails).
  5. Close the SSE consumer + terminate the pod.

The dispatcher is **pipeline-agnostic**: it doesn't know about compute
profiles, artifact-type routing, or fallback chains by name. It takes
a fully-resolved RunPod template + hardware spec — the pipeline does
the profile-to-template lookup before constructing the dispatcher.

The aii_runpod dependency is **lazy** — imported inside :meth:`dispatch`
so callers in environments without aii_runpod installed don't fail at
import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....run import Run


class RunPodAgentDispatcher:
    """Remote :class:`AgentInputDispatcher` — runs the agent in a RunPod pod.

    Construct with a fully-resolved RunPod target: ``template_id``
    (the pre-registered RunPod template), ``runpod_id`` (the primary
    hardware identifier, e.g. ``"NVIDIA RTX A4500"``), and an optional
    ``fallback_runpod_ids`` list (for transient capacity issues —
    RunPod schedule failures fall through to the next ID).

    Pipeline maps its concept of "compute profile" to these fields
    upfront via ``config.execute_env.runpod.compute_profiles[name]``.
    The dispatcher never sees the profile name.
    """

    def __init__(
        self,
        *,
        worker_pod: Any,  # aii_runpod.WorkerPod (the RunPod-cluster handle)
        template_id: str,  # pre-resolved RunPod template id (e.g. worker_gpu's id)
        runpod_id: str,  # primary hardware identifier
        fallback_runpod_ids: list[str] | tuple[str, ...] = (),
        container_disk_gb: int = 40,
        pod_timeout: int = 3600,
        pod_start_retries: int = 2,
        pod_per_instance_retries: int = 3,
        startup_retry_delay: float = 10.0,
        healthcheck_timeout: int = 600,
        retry_context_truncate_chars: int = 3000,
        retry_context_messages: int = 20,
        validation: dict | None = None,
    ) -> None:
        self._worker_pod = worker_pod
        self._template_id = template_id
        self._runpod_id = runpod_id
        self._fallback_runpod_ids = list(fallback_runpod_ids)
        self._container_disk_gb = container_disk_gb
        self._pod_timeout = pod_timeout
        self._pod_retries = pod_start_retries
        self._pod_start_retries = pod_per_instance_retries
        self._startup_retry_delay = startup_retry_delay
        self._healthcheck_timeout = healthcheck_timeout
        self._retry_context_truncate_chars = retry_context_truncate_chars
        self._retry_context_messages = retry_context_messages
        self._validation = validation

    async def dispatch(
        self,
        options: Any,
        prompts: list[str] | str,
        *,
        run: Run,
    ) -> Any:
        """Run one agent on a fresh worker pod; return its AgentResponse.

        ``run`` is the host Run that will receive every typed event the
        worker emits, ferried via SSE against the pod's ``/telemetry``
        endpoint. ``aii_runpod.WorkerPod.run_agent_job`` accepts a
        ``run`` kwarg that internally constructs the SSE consumer and
        tears it down on completion — keeping all pod-lifecycle /
        SSE-bridge wiring on the aii_runpod side, this class is just
        the protocol-shaped adapter.
        """
        options.container_timeout = self._pod_timeout
        return await self._worker_pod.run_agent_job(
            agent_options=options,
            prompts=prompts if isinstance(prompts, list) else [prompts],
            template_id=self._template_id,
            runpod_id=self._runpod_id,
            fallback_runpod_ids=self._fallback_runpod_ids,
            container_disk_gb=self._container_disk_gb,
            workspace_dir=str(options.cwd),
            timeout=self._pod_timeout,
            max_retries=self._pod_retries,
            task_id=options.run_id,
            task_name=options.agent_context,
            validation=self._validation,
            pod_per_instance_retries=self._pod_start_retries,
            startup_retry_delay=self._startup_retry_delay,
            healthcheck_timeout=self._healthcheck_timeout,
            retry_context_truncate_chars=self._retry_context_truncate_chars,
            retry_context_messages=self._retry_context_messages,
            run=run,  # ← host Run for the SSE bridge
        )


__all__ = ["RunPodAgentDispatcher"]
