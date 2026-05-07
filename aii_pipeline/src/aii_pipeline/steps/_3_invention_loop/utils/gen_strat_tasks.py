"""Strategy generation tasks — OpenRouter and Claude agent backends."""

from __future__ import annotations

import json
from pathlib import Path

from aii_lib.agent_backend import Agent
from aii_lib.llm_backend.tool_loop import _emit_summary
from aii_lib.run import emit

from aii_lib import OpenRouterClient, chat
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
    Strategies,
    assign_artifact_direction_ids,
    verify_strategies,
)
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.u_prompt import (
    build_artifact_retry_prompt,
)


def _unwrap_structured_output(raw: dict | None) -> dict | None:
    """Extract structured output data, handling the 'output' wrapper bug."""
    if raw is None:
        return None
    data = raw if isinstance(raw, dict) else {}
    if "output" in data and isinstance(data["output"], dict) and len(data) == 1:
        data = data["output"]
    return data


async def gen_strat(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    iteration: int,
    existing_artifact_ids: set[str],
    artifact_pool_map: dict[str, str],
    num_strategies: int,
    reasoning_effort: str = "medium",
    suffix: str | None = None,
    llm_timeout: int = 600,
    verify_retries: int = 2,
    min_valid_artifacts: int = 1,
    allowed_artifacts: list[str] | None = None,
    art_limit: int | None = None,
) -> list[dict]:
    """Generate strategies using aii_lib chat() with structured output.

    Includes verification + retry loop similar to cited_args workflow.
    """
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        effective_model = f"{model}:{suffix}" if suffix else model

        async with OpenRouterClient(
            api_key=api_key, model=effective_model, timeout=llm_timeout
        ) as client:
            # Initial generation
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=Strategies,
                task_id=task_id,
                task_name=task_name,
                timeout=llm_timeout,
                emit_summary=False,  # Summary emitted after verification
            )

            messages = result.messages
            conv_stats = result.stats

            output_text = client.extract_json_from_response(result.response)
            output_text = output_text.strip() if output_text else ""

            if not output_text:
                emit.end_task(task_id, name=task_name, status="done", text="No output")
                return []

            data = json.loads(output_text)
            strategies = data.get("strategies", [])

            # Assign IDs to artifact directions (LLM doesn't generate IDs)
            # Use working copy so IDs accumulate across strategies but existing_artifact_ids stays clean for verification
            id_tracker = set(existing_artifact_ids)
            for s in strategies:
                assign_artifact_direction_ids(s, id_tracker, iteration)

            # Verification + retry loop
            for attempt in range(verify_retries + 1):
                verification = verify_strategies(
                    strategies=strategies,
                    num_expected=num_strategies,
                    existing_artifact_ids=existing_artifact_ids,
                    artifact_pool_map=artifact_pool_map,
                    min_valid_artifacts=min_valid_artifacts,
                    allowed_artifacts=allowed_artifacts,
                    art_limit=art_limit,
                )

                if verification["valid"]:
                    status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'}"
                    if attempt > 0:
                        status += f" (retry {attempt})"
                    _emit_summary(conv_stats, client, task_id=task_id, task_name=task_name)
                    emit.end_task(task_id, name=task_name, status="done", text=status)
                    return strategies

                # Log issues
                all_errors = (
                    verification["count_errors"]
                    + verification["id_errors"]
                    + verification["dep_errors"]
                    + verification["type_errors"]
                    + verification["limit_errors"]
                )
                for err in all_errors[:5]:  # Limit logging
                    emit.status_public_warning(err)

                # Log valid artifact count if below minimum
                valid_count = verification.get("valid_artifact_count", 0)
                total_count = verification.get("total_artifact_count", 0)
                if valid_count < min_valid_artifacts:
                    emit.status_public_warning(
                        f"Only {valid_count}/{total_count} valid artifacts (need {min_valid_artifacts})"
                    )

                # Retry if attempts left
                if attempt < verify_retries:
                    retry_prompt = build_artifact_retry_prompt(
                        verification=verification,
                        num_strategies_requested=num_strategies,
                        min_valid_artifacts=min_valid_artifacts,
                        art_limit=art_limit,
                    )
                    emit.status_private_info(
                        f"Verification failed ({len(all_errors)} issues), retrying..."
                    )

                    messages.append({"role": "user", "content": retry_prompt})

                    result = await chat(
                        client=client,
                        messages=messages,
                        reasoning_effort=reasoning_effort,
                        response_format=Strategies,
                        task_id=task_id,
                        task_name=task_name,
                        timeout=llm_timeout,
                        conversation_stats=conv_stats,
                        emit_summary=False,
                    )

                    messages = result.messages
                    output_text = client.extract_json_from_response(result.response)
                    output_text = output_text.strip() if output_text else ""

                    if output_text:
                        data = json.loads(output_text)
                        strategies = data.get("strategies", [])
                        # Assign IDs - fresh tracker since retry replaces all strategies
                        id_tracker = set(existing_artifact_ids)
                        for s in strategies:
                            assign_artifact_direction_ids(s, id_tracker, iteration)
                    else:
                        emit.status_private_info("Retry produced no output, keeping previous")

            # All retries exhausted - return what we have
            _emit_summary(conv_stats, client, task_id=task_id, task_name=task_name)
            status = f"{len(strategies)} strateg{'ies' if len(strategies) != 1 else 'y'} (invalid)"
            emit.end_task(task_id, name=task_name, status="done", text=status)
            return strategies

    except TimeoutError:
        emit.end_task(
            task_id,
            name=task_name,
            status="failed",
            text=f"Timeout ({llm_timeout}s)" if llm_timeout else "Timeout",
        )
        raise
    except json.JSONDecodeError as e:
        emit.end_task(task_id, name=task_name, status="failed", text=f"JSON parse error: {e}")
        raise
    except Exception as e:
        emit.status_public_error(f"Strategy generation failed for {model}: {e}")
        emit.end_task(task_id, name=task_name, status="failed", text=f"Error: {e}")
        raise


# =============================================================================
# CLAUDE AGENT BACKEND (with verification + retry)
# =============================================================================


async def gen_strat_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    output_dir: Path,  # noqa: ARG001 — interface parity with OpenRouter path
    iteration: int,
    existing_artifact_ids: set[str],
    artifact_pool_map: dict[str, str],
    num_strategies: int,
    task_name: str,
    parent_module_id: str,
    verify_retries: int = 2,
    min_valid_artifacts: int = 1,
    allowed_artifacts: list[str] | None = None,
    art_limit: int | None = None,
) -> list[dict]:
    """Generate strategies using Claude agent with structured JSON output."""
    from aii_lib import build_options

    abs_cwd = Path(cwd).resolve()

    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    # Closure: validate strategies from structured output
    # Captures iteration, existing_artifact_ids, etc. from outer scope
    _validated_strategies = []  # mutable — post_validate stores parsed strategies here

    def _validate_strategies(structured_output):
        if not structured_output:
            return False, (
                "Your previous response did not produce valid structured output. "
                "Call the StructuredOutput tool with your data directly as the tool input — "
                "do NOT wrap it in an 'output' key. Example: {\"strategies\": [...]}."
            )
        data = _unwrap_structured_output(structured_output)
        if data is None:
            return False, (
                "Your previous response did not produce valid structured output. "
                "Call the StructuredOutput tool with your data directly."
            )

        strategies = data.get("strategies", [])
        # Assign IDs (LLM doesn't generate them)
        id_tracker = set(existing_artifact_ids)
        for s in strategies:
            assign_artifact_direction_ids(s, id_tracker, iteration)

        verification = verify_strategies(
            strategies=strategies,
            num_expected=num_strategies,
            existing_artifact_ids=existing_artifact_ids,
            artifact_pool_map=artifact_pool_map,
            min_valid_artifacts=min_valid_artifacts,
            allowed_artifacts=allowed_artifacts,
            art_limit=art_limit,
        )

        _validated_strategies.clear()
        _validated_strategies.extend(strategies)

        if verification["valid"]:
            return True, None

        retry_prompt = build_artifact_retry_prompt(
            verification=verification,
            num_strategies_requested=num_strategies,
            min_valid_artifacts=min_valid_artifacts,
            art_limit=art_limit,
        )
        return False, retry_prompt

    options = build_options(
        agent_cfg,
        abs_cwd,
        task_id=task_id,
        task_name=task_name,
        system_prompt=system_prompt,
        output_format=Strategies.to_struct_output(),
        continue_seq_item=True,
        post_validate=_validate_strategies,
        post_validate_retries=verify_retries,
    )

    try:
        response = await Agent(options).run(prompt)

        if response.failed:
            emit.end_task(
                task_id,
                name=task_name,
                status="failed",
                text=f"FAILED: {response.error_message or 'unknown'}",
            )
            return []

        if not _validated_strategies:
            emit.end_task(task_id, name=task_name, status="done", text="No output")
            return []

        # Mirror raw structured_output to ``task.output`` so replay-execute
        # synthesis can recover it on a subsequent fork (see gen_hypo for
        # the same pattern + rationale). Coerce through
        # ``Strategies(**dict)`` so the ``kind="strategies"``
        # discriminator default populates before assignment to
        # ``task.output: AnyOutput`` — pydantic's tagged-union dispatch
        # runs before field defaults, so a bare dict would fail
        # ``union_tag_not_found``. Replay path re-runs
        # ``_validate_strategies`` from the same shape.
        if response.structured_output:
            parsed_output = Strategies.model_validate(response.structured_output)
            emit.task_output(
                task_id=task_id,
                output=parsed_output,
            )
        emit.end_task(
            task_id,
            name=task_name,
            status="done",
            text=f"{len(_validated_strategies)} strategies",
        )
        return _validated_strategies
    except Exception as e:
        emit.end_task(task_id, name=task_name, status="failed", text=f"Error: {e}")
        raise
