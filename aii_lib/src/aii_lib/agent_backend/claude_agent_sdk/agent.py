"""
Public API for aii_lib agent backend.

This is the main entry point for users. Import Agent and AgentOptions to get started.
"""

import asyncio
import json
import time as _time
import traceback
from functools import partial
from typing import Any

from loguru import logger
from pydantic import ValidationError
from tenacity import AsyncRetrying, stop_after_attempt, wait_random

from aii_lib.llm_backend.claude_max import get_monitor
from aii_lib.run import emit, get_current_run

from .core import initialize_execution
from .core.config import prepare_sdk_options
from .core.error_recovery import (
    AgentState,
    execute_with_message_retry,
    run_with_deadline,
)
from .core.expected_files import (
    collect_paths_recursive,
    validate_and_retry_expected_files,
)
from .core.hooks import ensure_project_skills_link, install_time_remaining_hook
from .core.retry import (
    build_agent_retry_context,
    make_retry_callback,
    prepend_retry_context,
)
from .models import AgentOptions, AgentResponse, PromptResult

# Error reason → failure_reason string
_EXCEPTION_REASONS = {
    asyncio.TimeoutError: "agent_timeout",
    asyncio.CancelledError: "agent_timeout",
    ValueError: "validation_error",
    ValidationError: "validation_error",
    json.JSONDecodeError: "validation_error",
    OSError: "connection_error",
    ConnectionError: "connection_error",
    TimeoutError: "connection_error",
}


class Agent:
    """
    Claude Code agent for executing prompts with full SDK capabilities.

    Example:
        >>> agent = Agent(AgentOptions(model="sonnet", max_turns=50))
        >>> result = await agent.run("Calculate 5 + 3")
        >>> print(result.final_response)
    """

    def __init__(self, options: AgentOptions | None = None):
        self.options = options or AgentOptions()
        ensure_project_skills_link(self.options)
        self._deadlines: dict[str, float | None] = {
            "prompt": None,
            "agent": None,
            "container": (
                _time.time() + self.options.container_timeout
                if self.options.container_timeout
                else None
            ),
        }
        if any(
            [
                self.options.seq_prompt_timeout,
                self.options.agent_timeout,
                self.options.container_timeout,
            ]
        ):
            install_time_remaining_hook(self.options, self._deadlines)
        self._sdk_options_cache = None
        self._prompt_count = 0
        self._expected_files_instructions_added = False
        self._state = AgentState(
            options=self.options,
            deadlines=self._deadlines,
        )

    @staticmethod
    def _collect_paths_recursive(obj: Any) -> list[str]:
        """Recursively collect all file paths from nested structure."""
        return collect_paths_recursive(obj)

    # ── run() — agent-level retry loop ─────────────────────────────────

    async def run(self, prompts: str | list[str]) -> AgentResponse:
        """Execute one or more prompts with agent-level timeout and retry.

        NEVER CRASHES: On failure after all retries, returns empty response.

        Resume turn override: when the dispatched task lives under the
        :attr:`Run._pending_resume_target` module AND has a captured
        ``session_id``, swap to ``session_type=FORK + resume_session_id=<sid>``
        and replace the prompts with ``[run.prompt]`` — the SDK then
        FORK-resumes the task's prior conversation and sends the
        user-supplied prompt as the next turn. This is the *only*
        resume-aware code in the agent backend (besides the Stage 4
        replay short-circuit below); everything else downstream
        (retries, structured output, agent_end) runs identically to a
        fresh dispatch.

        Stage 4 — replay-execute synthesis: when the run is in
        ``_playback_mode == "replay"`` and the dispatched Task is
        already terminal (DONE/FAILED), short-circuit BEFORE arming
        deadlines or emitting agent_start. The synthesized response
        is built from the recorded ``task.output`` + ``task.messages``
        and returned as if the SDK had run again. Replay-mode flips
        to live the moment ``execute()`` reaches the resume target's
        ``start_*_module``, so by the time the live SDK fires we're
        out of replay-mode entirely.
        """
        from aii_lib.run.node import NodeStatus

        from .models import SessionType
        from .replay import synthesize_agent_response_from_task

        if isinstance(prompts, str):
            prompts = [prompts]

        _task_id = self.options.run_id or ""
        _task_name = self.options.agent_context or ""

        # Stage 4 — replay short-circuit. Must precede deadline arming
        # and agent_start so we don't pay the cost of bringing up the
        # SDK transport + retry hooks for a call that's only going to
        # synthesize from the tree.
        run = get_current_run()
        if run is not None and run.playback_mode == "replay" and _task_id:
            task = run.find_task(_task_id)
            if task is not None and task.status in (
                NodeStatus.DONE,
                NodeStatus.FAILED,
            ):
                return synthesize_agent_response_from_task(task)
            # Falls through to live dispatch when:
            #  - task is not yet terminal (resume target's substep
            #    being re-dispatched); mode should already have flipped
            #    to live by ``start_*_module`` before reaching here.
            #  - task can't be found (no recorded entry; only happens
            #    for dispatches that diverge from the recorded tree —
            #    treated as live for safety).

        # Resume / fork override (must precede agent_start so the FORK
        # session and overridden prompt land before any monitor / retry
        # hooks see the dispatch).
        #
        # Two parallel sources for the parent's ``session_id``:
        #
        #   * ``run._fork_session_ids[_task_id]`` — populated by
        #     ``run_pipeline_workflow`` from the ``aii_fork_overrides``
        #     row when the workflow body is a DBOS-native fork. The
        #     fork's run tree starts empty (DBOS only inherits cached
        #     step outputs, not the in-process aggregate), so a tree
        #     walk would miss; the override carries the parent's
        #     captured session ids directly.
        #
        #   * ``run.find_task(_task_id).session_id`` — the legacy
        #     resume path's lookup (and the legacy fork path while it
        #     still exists). Resume rebuilds the tree from the clone
        #     log so ``find_task`` resolves to the parent's recorded
        #     task with its session id intact.
        run = get_current_run()
        if run is not None and run.prompt and run._pending_resume_target and _task_id:
            sid = run._fork_session_ids.get(_task_id) if run._fork_session_ids else None
            if not sid:
                task = run.find_task(_task_id)
                if (
                    task is not None
                    and getattr(task, "parent_id", None) == run._pending_resume_target
                ):
                    sid = getattr(task, "session_id", None)
            if sid:
                self.options.session_type = SessionType.FORK
                self.options.resume_session_id = sid
                prompts = [run.prompt]

        # Emit AGENT_START — substep emits its own task_start; this brackets
        # the LLM-call lifecycle inside it. Multiple agent_start events per
        # task_id are legitimate (e.g. agent retries).
        if _task_id:
            emit.agent_start(_task_id)

        agent_retries = self.options.agent_retries
        agent_timeout = self.options.agent_timeout
        last_error = None

        for attempt_num in range(1, agent_retries + 1):
            try:
                if attempt_num > 1:
                    retry_context = build_agent_retry_context(self.options, last_error)
                    self._reset_for_retry()
                    if _task_id:
                        emit.agent_retry(
                            _task_id,
                            attempt=attempt_num,
                            reason=str(last_error) if last_error else "",
                            text=f"Agent retry... (attempt {attempt_num}/{agent_retries}): {last_error}",
                            max_attempts=agent_retries,
                            backend="claude_agent",
                            model=self.options.model,
                        )
                    retry_prompts = (
                        prepend_retry_context(prompts, retry_context) if retry_context else prompts
                    )
                else:
                    retry_prompts = prompts

                if agent_timeout is not None:
                    self._deadlines["agent"] = _time.time() + agent_timeout
                    self._state.capacity_wait_total = 0.0
                    result = await run_with_deadline(
                        partial(self._run_internal, retry_prompts),
                        agent_timeout,
                        self._state,
                        get_monitor,
                    )
                else:
                    result = await self._run_internal(retry_prompts)

                # Check for non-exception failures
                should_retry = False
                if self.options.output_format and result.structured_output is None:
                    should_retry, last_error = True, "structured_output is None"
                    self._state.last_failure_reason = "structured_output_missing"
                elif result.failed:
                    should_retry, last_error = (
                        True,
                        result.error_message or "agent returned failed=True",
                    )
                    self._state.last_failure_reason = "agent_failed"

                if should_retry and attempt_num < agent_retries:
                    if _task_id:
                        emit.agent_retry(
                            _task_id,
                            attempt=attempt_num,
                            reason=str(last_error) if last_error else "",
                            text=f"Agent result indicates failure (attempt {attempt_num}/{agent_retries}): {last_error}",
                            max_attempts=agent_retries,
                            backend="claude_agent",
                            model=self.options.model,
                        )
                    continue

                # Emit AGENT_END with session_id for fork/resume. The substep
                # emits its own task_end with domain-level status; agent_end
                # carries the SDK session_id which is an agent-layer detail.
                if _task_id:
                    status = (
                        "OK"
                        if not result.failed
                        else f"FAILED: {result.error_message or 'unknown'}"
                    )
                    emit.agent_end(
                        _task_id,
                        session_id=self._state.session_id,
                        text=status,
                    )
                return result

            except Exception as e:
                self._state.last_failure_reason = next(
                    (r for t, r in _EXCEPTION_REASONS.items() if isinstance(e, t)),
                    "process_error",
                )
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                if _task_id:
                    emit.agent_retry(
                        _task_id,
                        attempt=attempt_num,
                        reason=str(last_error),
                        text=f"Agent error (attempt {attempt_num}/{agent_retries}): {last_error}",
                        max_attempts=agent_retries,
                        backend="claude_agent",
                        model=self.options.model,
                    )
                if attempt_num < agent_retries:
                    import random

                    await asyncio.sleep(random.uniform(1, 20))

        emit.status_public_error(f"Agent failed after {agent_retries} retries: {last_error}")

        result = AgentResponse(
            final_response="",
            failed=True,
            error_message=f"Agent failed after {agent_retries} retries: {last_error}",
        )

        # Emit AGENT_END with session_id even on failure (for fork/resume from
        # last known state). Substep's task_end fires separately at the
        # substep layer.
        if _task_id:
            emit.agent_end(
                _task_id,
                session_id=self._state.session_id,
                text=f"FAILED: {last_error}",
            )
        return result

    def _reset_for_retry(self):
        self._prompt_count = 0
        self._sdk_options_cache = None
        self._state.session_id = None
        self._expected_files_instructions_added = False

    # ── Prompt sequencing ──────────────────────────────────────────────

    async def _run_internal(self, prompts: str | list[str]) -> AgentResponse:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("At least one prompt must be provided")

        prompts = prompts.copy()
        if (
            self.options.expected_files_struct_out_field
            and not self._expected_files_instructions_added
        ):
            self._expected_files_instructions_added = True

        prompt_results = []
        total_prompts = len(prompts)

        for i, prompt in enumerate(prompts):
            global_idx = self._prompt_count + i
            try:
                result, self._sdk_options_cache = await self._execute_single_prompt(
                    prompt,
                    global_idx,
                    self._sdk_options_cache,
                    with_output_format=(i == total_prompts - 1),
                )
            except Exception as e:
                label = "timed out" if isinstance(e, asyncio.TimeoutError) else f"failed: {e}"
                emit.status_public_warning(f"Prompt {i + 1}/{total_prompts} {label}")
                raise

            prompt_results.append(result)

            if (
                self.options.max_turns is not None
                and result.num_turns >= self.options.max_turns
                and result.structured_output is None
            ):
                await self._do_force_output(prompt_results)
                break

        self._prompt_count += len(prompt_results)

        # Expected files validation
        expected_files_valid = True
        if self.options.expected_files_struct_out_field:
            expected_files_valid = await validate_and_retry_expected_files(
                self.options,
                prompt_results,
                execute_prompt_fn=self._execute_expected_files_prompt,
            )

        # Structured output
        structured_output = None
        if self.options.output_format and prompt_results:
            last = prompt_results[-1]
            if last.structured_output is not None:
                structured_output = last.structured_output

        # Post-run validation hook
        if self.options.post_validate and structured_output is not None:
            for _val_attempt in range(self.options.post_validate_retries + 1):
                valid, retry_prompt = self.options.post_validate(structured_output)
                if valid:
                    break
                if retry_prompt and _val_attempt < self.options.post_validate_retries:
                    emit.status_public_warning(
                        f"Post-validation failed (attempt {_val_attempt + 1}/{self.options.post_validate_retries}), retrying..."
                    )
                    result, self._sdk_options_cache = await self._execute_single_prompt(
                        retry_prompt,
                        self._prompt_count,
                        self._sdk_options_cache,
                        with_output_format=True,
                    )
                    self._prompt_count += 1
                    prompt_results.append(result)
                    if result.structured_output is not None:
                        structured_output = result.structured_output

        return AgentResponse(
            final_response=prompt_results[-1].response if prompt_results else "",
            structured_output=structured_output,
            expected_files_valid=expected_files_valid,
        )

    async def _execute_expected_files_prompt(
        self, prompt: str, with_output_format: bool = True
    ) -> tuple[PromptResult, Any]:
        """Callback for validate_and_retry_expected_files — executes a prompt and tracks count."""
        idx = self._prompt_count
        self._prompt_count += 1
        return await self._execute_single_prompt(
            prompt,
            idx,
            self._sdk_options_cache,
            with_output_format=with_output_format,
        )

    async def _do_force_output(self, prompt_results: list) -> None:
        emit.status_public_warning(
            f"Max turns ({self.options.max_turns}) reached. Sending force output prompt."
        )
        from .core.prompts import build_force_output_prompt

        force_prompt = build_force_output_prompt(self.options)
        try:
            result, self._sdk_options_cache = await self._execute_single_prompt(
                force_prompt,
                self._prompt_count + len(prompt_results),
                self._sdk_options_cache,
                with_output_format=True,
            )
            prompt_results.append(result)
            if result.session_id:
                self._state.session_id = result.session_id
        except Exception as e:
            logger.error(
                f"Force output prompt failed ({type(e).__name__}): {e}\n{traceback.format_exc()}"
            )
            emit.status_public_error(f"Force output prompt failed: {e}")

    # ── Single prompt execution ────────────────────────────────────────

    async def _execute_single_prompt(
        self,
        prompt: str,
        prompt_index: int,
        sdk_options_cache: Any,
        with_output_format: bool = False,
    ) -> tuple[PromptResult, Any]:
        """Execute a single prompt with retry and timeout support."""
        max_retries = self.options.seq_prompt_retries
        timeout = self.options.seq_prompt_timeout

        # Check usage limits
        monitor = get_monitor()
        if monitor._config["usage_tracking"]["enabled"]:
            monitor.start()
            wait_start = _time.monotonic()
            await monitor.async_wait_for_capacity()
            wait_duration = _time.monotonic() - wait_start
            if wait_duration > 0.1:
                self._state.capacity_wait_total += wait_duration

        original_prompt = prompt

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_random(min=1, max=20),
            before_sleep=make_retry_callback(
                max_retries, self.options.run_id, self.options.agent_context
            ),
            reraise=True,
        ):
            with attempt:
                # STEP 1: Prepare SDK options (emits status_private_info)
                sdk_options, effective_prompt, sdk_options_cache = prepare_sdk_options(
                    self.options,
                    self._state.session_id,
                    self._state.last_failure_reason,
                    prompt_index,
                    original_prompt,
                    sdk_options_cache,
                    with_output_format,
                )

                # STEP 2: Initialize execution (emits S_PROMPT + PROMPT)
                execution_state = initialize_execution(self.options, effective_prompt, prompt_index)

                from aii_lib.agent_backend.claude_agent_sdk.claude_agent_tel_adapter import (
                    adapt as _adapt_claude,
                )

                _task_id = self.options.run_id or ""
                _task_name = self.options.agent_context or ""

                # B023: ``_message_callback`` captures ``execution_state`` /
                # ``_task_id`` / ``_task_name`` from the outer retry-loop. The
                # callback is consumed by the SDK call below within this same
                # iteration; the SDK completes before the loop iterates, so no
                # cross-iteration aliasing is possible here.
                def _message_callback(msg_dict: dict):
                    execution_state["message_count"] += 1  # noqa: B023
                    run = get_current_run()
                    if run is None:
                        return
                    run._on(_adapt_claude(msg_dict, _task_id, _task_name))  # noqa: B023

                # STEP 3: Execute with message retry (launches Claude CLI).
                # Each ClaudeSDKClient instance emits its own agent_summary
                # via streaming.py's message_callback path; NodeStats sums
                # them via apply_leaf_summary — no fork-chain bookkeeping.
                result_tuple = await execute_with_message_retry(
                    state=self._state,
                    effective_prompt=effective_prompt,
                    original_prompt=original_prompt,
                    sdk_options=sdk_options,
                    sdk_options_cache=sdk_options_cache,
                    execution_state=execution_state,
                    message_callback=_message_callback,
                    prompt_index=prompt_index,
                    timeout=timeout,
                    with_output_format=with_output_format,
                    get_monitor=get_monitor,
                )
                (
                    response_text,
                    session_id,
                    _summary_data,
                    num_turns,
                    structured_output,
                ) = result_tuple

                # STEP 4: Post-execution bookkeeping
                if session_id:
                    self._state.session_id = session_id

                # The agent_summary Run event was already emitted by
                # streaming.py's message_callback when the ResultMessage
                # arrived — single source of truth, per-instance, NodeStats
                # sums them via ``apply_leaf_summary``. No re-emit here.

                return PromptResult(
                    response=response_text,
                    session_id=session_id,
                    num_turns=num_turns,
                    structured_output=structured_output,
                ), sdk_options_cache

        # Unreachable: AsyncRetrying with reraise=True either yields a
        # successful attempt (returned above) or re-raises the last
        # exception. ruff/mypy can't see that, so make the impossible
        # path explicit.
        raise RuntimeError("AsyncRetrying loop exited without raising or returning")


__all__ = ["Agent"]
