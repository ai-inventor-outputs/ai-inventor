"""Proof executor — Lean 4 formal verification."""

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
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof import u_prompt
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.out_schema import (
    ProofArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.s_prompt import (
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
    / "proof_workspace"
)


def _find_proof_files(workspace_dir: Path) -> dict:
    result = {"lean_files": [], "code_files": [], "valid": False}
    result["lean_files"] = [f.name for f in workspace_dir.glob("*.lean")]
    result["code_files"] = [f.name for f in workspace_dir.glob("*.py")]
    result["valid"] = len(result["lean_files"]) > 0
    if not result["valid"]:
        result["error"] = "No Lean files found in workspace"
    return result


async def execute_proof(
    plan: BasePlan,
    artifacts: list[BaseArtifact],
    config: PipelineConfig,
    run_dir: Path,
    iteration: int = 1,
    proof_idx: int = 0,
    task_id: str | None = None,
    task_name: str | None = None,
    parent_module_id: str = "",
    user_uploads_path: str = "",
) -> tuple[dict, bool]:
    effective_task_name = task_name or task_id or f"proof_workspace_idx{proof_idx}"
    workspace_dir = run_dir / effective_task_name
    setup_workspace(workspace_dir, template_dir=WORKSPACE_TEMPLATE)
    effective_task_id = start_task(effective_task_name, parent_module_id)

    emit.status_private_info(f"Executing PROOF: {plan.title}")

    prompts = u_prompt.get_all_prompts(
        plan_text=plan.to_prompt_yaml(),
        artifacts=artifacts,
        dependency_ids=[d.id for d in plan.artifact_dependencies],
        workspace_path=str(workspace_dir),
        user_folder_path=user_uploads_path,
    )

    options = build_options(
        config.invention_loop.execute.proof.claude_agent,
        workspace_dir,
        task_id=effective_task_id,
        task_name=effective_task_name,
        system_prompt=get_system(),
        output_format=ProofArtifact.to_struct_output(),
    )
    validation = build_validation("proof", config)

    try:
        _agent, result = await create_and_run_agent(
            options=options,
            prompts=prompts,
            config=config,
            plan=plan,
            pod_timeout=config.invention_loop.execute.proof.claude_agent.pod_timeout,
            pod_start_retries=config.invention_loop.execute.proof.claude_agent.pod_start_retries,
            validation=validation,
        )

        if result.failed:
            end_task_failure(
                effective_task_id,
                effective_task_name,
                f"Agent failed: {result.error_message or 'unknown'}",
            )
            return {}, False

        result_dict = _find_proof_files(workspace_dir)
        if result_dict.get("error"):
            end_task_failure(effective_task_id, effective_task_name, result_dict["error"])
            return result_dict, False

        enrich_result(result_dict, result, plan, workspace_dir)
        end_task_success(effective_task_id, effective_task_name)
        return result_dict, True

    except TimeoutError:
        end_task_timeout(
            effective_task_id,
            effective_task_name,
            config.invention_loop.execute.proof.claude_agent.seq_prompt_timeout or 0,
        )
        raise
    except Exception as e:
        end_task_error(effective_task_id, effective_task_name, str(e))
        raise
