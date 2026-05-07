"""Dataset executor — HuggingFace/OWID dataset acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aii_lib.run import emit

from aii_lib import (
    build_options,
    end_task_error,
    end_task_failure,
    end_task_success,
    end_task_timeout,
    setup_workspace,
    start_task,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset import u_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.out_schema import (
    DatasetArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.s_prompt import (
    get as get_system,
)

from .base import build_validation, enrich_result
from .exec_mode_router import create_and_run_agent

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )
    from aii_pipeline.utils import PipelineConfig

WORKSPACE_TEMPLATE = (
    Path(__file__).parent.parent.parent.parent
    / "prompts"
    / "steps"
    / "_3_invention_loop"
    / "_3_gen_art"
    / "dataset_workspace"
)


async def execute_dataset(
    plan: BasePlan,
    artifacts: list[BaseArtifact],
    config: PipelineConfig,
    run_dir: Path,
    iteration: int = 1,
    dataset_idx: int = 0,
    task_id: str | None = None,
    task_name: str | None = None,
    parent_module_id: str = "",
    user_uploads_path: str = "",
) -> tuple[dict, bool]:
    dataset_cfg = config.invention_loop.execute.dataset

    effective_task_name = task_name or task_id or f"dataset_workspace_idx{dataset_idx}"
    workspace_dir = run_dir / effective_task_name
    setup_workspace(workspace_dir, template_dir=WORKSPACE_TEMPLATE)
    effective_task_id = start_task(effective_task_name, parent_module_id)

    emit.status_private_info(f"Executing DATASET: {plan.title}")

    # Build prompts (dataset has extra config params)
    target_num = min(plan.target_num_datasets, dataset_cfg.dataset_chosen_final_cap)
    prompts = u_prompt.get_all_prompts(
        plan_text=plan.to_prompt_yaml(),
        artifacts=artifacts,
        dependency_ids=[d.id for d in plan.artifact_dependencies],
        target_num_datasets=target_num,
        max_dataset_size=dataset_cfg.dataset_max_size,
        search_tool_cap=dataset_cfg.dataset_search_tool_cap,
        chosen_for_preview_cap=dataset_cfg.dataset_chosen_for_preview_cap,
        chosen_for_download_cap=dataset_cfg.dataset_chosen_for_download_cap,
        workspace_path=str(workspace_dir),
        user_folder_path=user_uploads_path,
    )

    options = build_options(
        config.invention_loop.execute.dataset.claude_agent,
        workspace_dir,
        task_id=effective_task_id,
        task_name=effective_task_name,
        system_prompt=get_system(),
        output_format=DatasetArtifact.to_struct_output(),
    )
    validation = build_validation("dataset", config)

    try:
        _agent, result = await create_and_run_agent(
            options=options,
            prompts=prompts,
            config=config,
            plan=plan,
            pod_timeout=dataset_cfg.claude_agent.pod_timeout,
            pod_start_retries=dataset_cfg.claude_agent.pod_start_retries,
            validation=validation,
        )

        if result.failed:
            end_task_failure(
                effective_task_id,
                effective_task_name,
                f"Agent failed: {result.error_message or 'unknown'}",
            )
            return {}, False

        # Dataset result comes from structured output file paths
        from aii_lib.agent_backend import Agent

        result_dict = {
            "workspace_path": str(workspace_dir),
            "file_list": [],
            "data_file_paths": [],
            "valid": True,
        }
        if result.structured_output:
            out_expected = result.structured_output.get("out_expected_files", {})
            result_dict["file_list"] = Agent._collect_paths_recursive(out_expected)
            data_paths = []
            for ds in out_expected.get("datasets", []):
                if isinstance(ds, dict):
                    data_paths.extend(ds.get("full", []))
                    if ds.get("mini"):
                        data_paths.append(ds["mini"])
                    if ds.get("preview"):
                        data_paths.append(ds["preview"])
            result_dict["data_file_paths"] = data_paths or [
                p for p in result_dict["file_list"] if p.endswith(".json")
            ]

        if not result_dict["file_list"]:
            raise RuntimeError("No dataset files found in workspace")

        enrich_result(result_dict, result, plan, workspace_dir)
        # Mirror raw structured_output to ``task.output`` so replay-execute
        # synthesis can recover it on a subsequent fork (see
        # task_output_replay_pattern.md). The substep gates on
        # ``result.structured_output`` (line 123) — without this, a fork
        # at the dataset slot mid-run gets an empty ``out_expected_files``
        # and the file_list extraction silently produces an empty result.
        # Coerce through ``DatasetArtifact(**dict)`` so the
        # ``kind="dataset_artifact"`` discriminator default populates
        # before assignment to ``task.output: AnyOutput`` — pydantic's
        # tagged-union dispatch runs before field defaults.
        if result.structured_output:
            parsed_output = DatasetArtifact.model_validate(result.structured_output)
            emit.task_output(
                task_id=effective_task_id,
                output=parsed_output,
            )
        end_task_success(effective_task_id, effective_task_name)
        return result_dict, True

    except TimeoutError:
        end_task_timeout(
            effective_task_id,
            effective_task_name,
            dataset_cfg.claude_agent.seq_prompt_timeout or 0,
        )
        raise
    except Exception as e:
        end_task_error(effective_task_id, effective_task_name, str(e))
        raise
