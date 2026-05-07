"""Research executor — Claude agent backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.run import emit

from aii_lib import (
    build_options,
    end_task_error,
    end_task_failure,
    end_task_success,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.out_schema import (
    ResearchArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.s_prompt import (
    get as get_exec_sysprompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.u_prompt import (
    get as get_exec_prompt,
)

from ..base import build_validation
from ..exec_mode_router import create_and_run_agent
from ..research import write_research_report

if TYPE_CHECKING:
    from pathlib import Path

    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )
    from aii_pipeline.utils import PipelineConfig


async def execute_research_claude(
    plan: BasePlan,
    artifacts: list[BaseArtifact],
    config: PipelineConfig,
    workspace_dir: Path,
    *,
    task_id: str,
    task_name: str,
    user_uploads_path: str = "",
) -> tuple[dict, bool]:
    """Execute research via Claude agent with structured output."""
    claude_cfg = config.invention_loop.execute.research.claude_agent

    prompt = get_exec_prompt(
        plan_text=plan.to_prompt_yaml(),
        artifacts=artifacts,
        dependency_ids=[d.id for d in plan.artifact_dependencies],
        agent_mode=True,
        workspace_path=str(workspace_dir),
        user_folder_path=user_uploads_path,
    )

    options = build_options(
        config.invention_loop.execute.research.claude_agent,
        workspace_dir,
        task_id=task_id,
        task_name=task_name,
        system_prompt=get_exec_sysprompt(),
        output_format=ResearchArtifact.to_struct_output(),
    )
    validation = build_validation("research", config, file_size_retries=0)

    try:
        _agent, response = await create_and_run_agent(
            options=options,
            prompts=prompt,
            config=config,
            plan=plan,
            pod_timeout=claude_cfg.pod_timeout,
            pod_start_retries=claude_cfg.pod_start_retries,
            validation=validation,
        )

        if response.failed:
            err = response.error_message or "unknown error"
            end_task_failure(task_id, task_name, f"Agent failed: {err}")
            return {"error": f"Agent failed: {err}"}, False

        if not response.structured_output:
            raise RuntimeError("No output from Claude agent for research executor")

        data = response.structured_output if isinstance(response.structured_output, dict) else {}
        answer = data.get("answer", "")
        if not answer:
            end_task_failure(task_id, task_name, "No answer in output")
            return {"error": "No answer in output"}, False

        title = data.get("title", "") or plan.title.strip()
        summary = data.get("summary", "") or (
            f"Research: {answer[:180]}..." if len(answer) > 180 else f"Research: {answer}"
        )

        research_result = {
            "question": plan.question or plan.research_plan,
            "answer": answer,
            "title": title,
            "sources": data.get("sources", []),
            "follow_up_questions": data.get("follow_up_questions", []),
            "summary": summary,
            "layman_summary": data.get("layman_summary", ""),
            "model": claude_cfg.model,
            "workspace_path": str(workspace_dir),
        }

        write_research_report(research_result, workspace_dir)

        # Mirror typed ResearchArtifact to ``task.output`` so replay-execute
        # synthesis can recover it on a subsequent fork. Emit through
        # ``ResearchArtifact.model_validate`` so the
        # ``kind="research_artifact"`` discriminator default populates
        # before assignment to ``task.output: AnyOutput`` — pydantic's
        # tagged-union dispatch runs before field defaults, so a bare
        # dict without ``kind`` would fail ``union_tag_not_found``.
        try:
            parsed_output = ResearchArtifact.model_validate(data)
            emit.task_output(task_id=task_id, output=parsed_output)
        except Exception as e:
            # Don't fail the whole executor if ResearchArtifact validation
            # rejects the raw structured_output (e.g. extra LLM fields).
            # The artifact is still produced + written to disk.
            emit.status_private_info(f"ResearchArtifact.model_validate failed for task_output: {e}")
        emit.status_public_success(
            f"Research complete: {len(answer)} chars, {len(research_result['sources'])} sources"
        )
        end_task_success(task_id, task_name)
        return research_result, True

    except Exception as e:
        end_task_error(task_id, task_name, str(e))
        raise
