"""Research executor — OpenRouter backend with research_workflow."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aii_lib.abilities.endpoint_names import AII_WEB_FETCH, AII_WEB_SEARCH
from aii_lib.run import emit

from aii_lib import (
    OpenRouterClient,
    ResearchWorkflowConfig,
    end_task_error,
    end_task_success,
    end_task_timeout,
    research_workflow,
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
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.u_prompt import (
    get_force_output_prompt,
)

from ..research import RESEARCH_TOOLS, write_research_report

if TYPE_CHECKING:
    from pathlib import Path

    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )
    from aii_pipeline.utils import PipelineConfig


async def execute_research_openrouter(
    plan: BasePlan,
    artifacts: list[BaseArtifact],
    config: PipelineConfig,
    workspace_dir: Path,
    *,
    task_id: str,
    task_name: str,
    user_uploads_path: str = "",
) -> tuple[dict, bool]:
    """Execute research via OpenRouter with research_workflow tool loop."""
    research_cfg = config.invention_loop.execute.research
    openrouter_key = config.api_keys.openrouter

    if not openrouter_key:
        raise ValueError("Missing OpenRouter API key for research executor")

    model = research_cfg.model
    if model == "claude-sonnet-4-5":
        raise ValueError(f"Model '{model}' not available on OpenRouter")

    prompt = get_exec_prompt(
        plan_text=plan.to_prompt_yaml(),
        artifacts=artifacts,
        dependency_ids=[d.id for d in plan.artifact_dependencies],
        user_folder_path=user_uploads_path,
    )

    try:
        effective_model = OpenRouterClient.resolve_model(model, research_cfg.suffix)

        async with OpenRouterClient(
            api_key=openrouter_key,
            model=effective_model,
            timeout=research_cfg.llm_timeout,
        ) as client:
            result = await research_workflow(
                client=client,
                prompt=prompt,
                system=get_exec_sysprompt(),
                config=ResearchWorkflowConfig(
                    max_tool_iterations=research_cfg.max_tool_iterations,
                    force_output_prompt=get_force_output_prompt(),
                    tools=RESEARCH_TOOLS,
                    timeout=research_cfg.llm_timeout,
                ),
                response_format=ResearchArtifact,
                task_id=task_id,
                task_name=task_name,
                reasoning_effort=research_cfg.reasoning_effort,
            )

            answer = result.output.get("answer") if result.output else None
            if not answer:
                stats = result.tool_result.stats
                error = (
                    f"No output after {result.tool_result.iterations_used} iterations. "
                    f"Tools: {stats.tool_calls.get(AII_WEB_SEARCH, 0)} searches, "
                    f"{stats.tool_calls.get(AII_WEB_FETCH, 0)} fetches. "
                    f"Cost: ${stats.total_cost:.4f}"
                )
                raise RuntimeError(f"Research produced no answer: {error}")

            cost = result.tool_result.stats.total_cost
            title = (
                (result.output.get("title", "") or plan.title.strip())
                if result.output
                else plan.title.strip()
            )
            summary = result.output.get("summary", "") if result.output else ""
            if not summary:
                summary = (
                    f"Research: {answer[:180]}..." if len(answer) > 180 else f"Research: {answer}"
                )

            research_result = {
                "question": plan.question or plan.research_plan,
                "answer": answer,
                "title": title,
                "sources": result.output.get("sources", []) if result.output else [],
                "follow_up_questions": result.output.get("follow_up_questions", [])
                if result.output
                else [],
                "summary": summary,
                "layman_summary": result.output.get("layman_summary", "") if result.output else "",
                "model": result.tool_result.stats.model or effective_model,
                "tool_calls": {
                    AII_WEB_SEARCH: result.tool_result.stats.tool_calls.get(AII_WEB_SEARCH, 0),
                    AII_WEB_FETCH: result.tool_result.stats.tool_calls.get(AII_WEB_FETCH, 0),
                },
                "iterations_used": result.tool_result.iterations_used,
                "forced_output": result.forced_output,
                "workspace_path": str(workspace_dir),
            }

            # Write output files
            (workspace_dir / "research_out.json").write_text(
                json.dumps(research_result, indent=2), encoding="utf-8"
            )
            write_research_report(research_result, workspace_dir)

            emit.status_public_success(
                f"Research complete: {len(answer)} chars, {len(research_result['sources'])} sources, ${cost:.4f}",
            )
            end_task_success(task_id, task_name, cost=cost)
            return research_result, True

    except TimeoutError:
        end_task_timeout(task_id, task_name, research_cfg.llm_timeout)
        raise
    except Exception as e:
        end_task_error(task_id, task_name, str(e))
        raise
