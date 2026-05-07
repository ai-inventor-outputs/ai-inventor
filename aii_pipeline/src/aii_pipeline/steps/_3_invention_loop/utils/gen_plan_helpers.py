"""Helper functions for gen_plan — single-artifact plan generation and testing strategies."""

import json
from pathlib import Path

from aii_lib.agent_backend import Agent
from aii_lib.run import emit

from aii_lib import OpenRouterClient, chat
from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
    ArtifactDep,
    ArtifactDirection,
    Strategy,
)
from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
    get_plan_schema,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    ArtifactType,
    BaseArtifact,
)


async def gen_plan_for_art(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    iteration: int,  # noqa: ARG001 — interface parity between OpenRouter / Claude paths
    artifact_direction: ArtifactDirection,
    reasoning_effort: str = "medium",
    suffix: str | None = None,
    llm_timeout: int = 600,
) -> dict | None:
    """Generate a plan for a single artifact direction using OpenRouter."""
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    plan_cls = get_plan_schema(artifact_direction.type)

    try:
        effective_model = f"{model}:{suffix}" if suffix else model

        async with OpenRouterClient(
            api_key=api_key, model=effective_model, timeout=llm_timeout
        ) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=plan_cls.plan_output_format()["schema"],
                task_id=task_id,
                task_name=task_name,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            output_text = output_text.strip() if output_text else ""
            if output_text:
                plan = json.loads(output_text)

                plan["type"] = artifact_direction.type
                plan["in_art_direction_id"] = artifact_direction.id
                plan["artifact_dependencies"] = [
                    d.model_dump() for d in artifact_direction.depends_on
                ]

                emit.end_task(task_id, name=task_name, status="done", text="1 plan")
                return plan

        emit.end_task(task_id, name=task_name, status="done", text="No output")
        return None

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
        emit.status_public_error(f"OpenRouter plan generation failed for {model}: {e}")
        emit.end_task(task_id, name=task_name, status="failed", text=f"Error: {e}")
        raise


async def gen_plan_for_art_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    iteration: int,  # noqa: ARG001 — interface parity between OpenRouter / Claude paths
    artifact_direction: ArtifactDirection,
    task_name: str,
    parent_module_id: str,
) -> dict | None:
    """Generate a plan for a single artifact direction using Claude agent."""
    from aii_lib import build_options

    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        plan_cls = get_plan_schema(artifact_direction.type)

        options = build_options(
            agent_cfg,
            cwd,
            task_id=task_id,
            task_name=task_name,
            system_prompt=system_prompt,
            output_format=plan_cls.plan_output_format(),
        )

        response = await Agent(options).run(prompt)

        if response.failed:
            emit.end_task(
                task_id,
                name=task_name,
                status="failed",
                text=f"FAILED: {response.error_message or 'unknown'}",
            )
            return None

        if response.structured_output:
            plan = (
                response.structured_output
                if isinstance(response.structured_output, dict)
                else (
                    response.structured_output.model_dump()
                    if hasattr(response.structured_output, "model_dump")
                    else {}
                )
            )
            plan["type"] = artifact_direction.type
            plan["in_art_direction_id"] = artifact_direction.id
            plan["artifact_dependencies"] = [d.model_dump() for d in artifact_direction.depends_on]
            # Mirror the post-processed plan to ``task.output`` so
            # replay-execute synthesis can recover it on a subsequent
            # fork. Coerce through ``plan_cls.model_validate`` so the
            # subclass's ``kind="<artifact-type>_plan"`` discriminator
            # default populates before assignment to ``task.output:
            # AnyOutput`` — pydantic's tagged-union dispatch runs before
            # field defaults. Wrapped in try/except so a Pydantic
            # rejection of extra fields doesn't fail the whole substep
            # (the post-processed ``plan`` dict still flows downstream
            # via the ``return plan`` below regardless).
            try:
                parsed_plan = plan_cls.model_validate(plan)
                emit.task_output(task_id=task_id, output=parsed_plan)
            except Exception as e:
                emit.status_private_info(
                    f"{plan_cls.__name__}.model_validate failed for task_output: {e}"
                )
            emit.end_task(task_id, name=task_name, status="done", text="1 plan")
            return plan

        emit.end_task(task_id, name=task_name, status="done", text="No output")
        return None
    except Exception as e:
        emit.end_task(task_id, name=task_name, status="failed", text=f"Error: {e}")
        raise


def _get_creatable_types(
    artifacts: list[BaseArtifact],
    allowed_artifacts: list[str] | None = None,
) -> list[str]:
    """Determine which artifact types can be created in the current iteration."""
    all_types = ["research", "dataset", "proof", "experiment", "evaluation"]
    allowed = allowed_artifacts if allowed_artifacts else all_types

    creatable = []
    for artifact_type in allowed:
        if artifact_type in ["research", "dataset", "proof"]:
            creatable.append(artifact_type)
        elif artifact_type == "experiment":
            has_dataset = any(a.type == ArtifactType.DATASET for a in artifacts)
            if has_dataset:
                creatable.append(artifact_type)
        elif artifact_type == "evaluation":
            has_experiment = any(a.type == ArtifactType.EXPERIMENT for a in artifacts)
            if has_experiment:
                creatable.append(artifact_type)

    return creatable


def _create_testing_strategy(
    artifacts: list[BaseArtifact],
    allowed_artifacts: list[str] | None,
    iteration: int,
) -> Strategy:
    """Create a synthetic strategy for testing mode with one artifact direction per creatable type."""
    creatable_types = _get_creatable_types(artifacts, allowed_artifacts)

    artifact_directions = []
    for i, artifact_type in enumerate(creatable_types, start=1):
        depends_on: list[ArtifactDep] = []
        if artifact_type == "experiment":
            datasets = [a for a in artifacts if a.type == ArtifactType.DATASET]
            if datasets:
                depends_on = [ArtifactDep(id=datasets[0].id, label="dataset")]
        elif artifact_type == "evaluation":
            experiments = [a for a in artifacts if a.type == ArtifactType.EXPERIMENT]
            if experiments:
                depends_on = [ArtifactDep(id=experiments[0].id, label="experiment")]

        artifact_directions.append(
            ArtifactDirection(
                id=f"test_{artifact_type}_iter{iteration}_idx{i}",
                type=artifact_type,
                objective=f"Test {artifact_type} generation for iteration {iteration}",
                approach=f"Generate a test {artifact_type} to verify the executor works",
                depends_on=depends_on,
            )
        )

    return Strategy(
        id=f"test_strat_it{iteration}__testing",
        title="Testing Strategy",
        objective="Test all artifact types",
        rationale="Synthetic strategy for testing mode",
        artifact_directions=artifact_directions,
        expected_outcome="One artifact of each creatable type",
    )
