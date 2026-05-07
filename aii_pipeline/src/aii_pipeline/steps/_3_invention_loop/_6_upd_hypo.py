"""UPD_HYPO Step - Internal hypothesis revision.

"What do we now believe?" — same LLM revises the hypothesis based on
the evidence gathered (artifacts + paper text).

Output: RevisedHypothesis → converted to hypothesis dict for next iteration.

Supports two backends:
- OpenRouter: Uses chat() with structured output
- Claude agent: Uses Agent with SDK native output_format

Uses aii_lib for:
- OpenRouterClient: LLM calls (OpenRouter backend)
- Agent/AgentOptions: Claude agent calls
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aii_lib.agent_backend import Agent
from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import SingleTModule

from aii_lib import OpenRouterClient, chat
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._6_upd_hypo.out_schema import (
    ArtifactRelation,
    RevisedHypothesis,
)
from aii_pipeline.prompts.steps._3_invention_loop._6_upd_hypo.s_prompt import (
    get as get_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._6_upd_hypo.u_prompt import (
    get as get_user_prompt,
)
from aii_pipeline.steps.base import ModuleCtx

from .invention_loop import LoopCtx


def _merge_artifact_relations(
    artifacts: list[BaseArtifact],
    relations: list[ArtifactRelation],
) -> int:
    """Write artifact relations back onto in_dependency entries.

    Write each typed (from_id, to_id) relation back onto the dependent
    artifact's matching in_dependency entry. Returns the count of
    in_dependencies that were updated.
    """
    by_to: dict[str, dict[str, ArtifactRelation]] = {}
    for rel in relations:
        by_to.setdefault(rel.to_id, {})[rel.from_id] = rel

    updated = 0
    for artifact in artifacts:
        rel_map = by_to.get(artifact.id)
        if not rel_map:
            continue
        for dep in artifact.in_dependencies:
            rel = rel_map.get(dep.id)
            if rel is None:
                continue
            dep.relation_type = rel.relation_type
            dep.relation_rationale = rel.relation_rationale
            updated += 1
    return updated


async def upd_hypo_openrouter(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    reasoning_effort: str = "medium",
    llm_timeout: int = 600,
) -> RevisedHypothesis | None:
    """Revise hypothesis using OpenRouter chat."""
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=RevisedHypothesis,
                task_id=task_id,
                task_name=task_name,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            if output_text:
                data = json.loads(output_text)
                revised = RevisedHypothesis(
                    title=data.get("title", ""),
                    hypothesis=data.get("hypothesis", ""),
                    description=data.get("description", ""),
                    relation_rationale=data.get("relation_rationale", ""),
                    confidence_delta=data.get("confidence_delta", "unchanged"),
                    key_changes=data.get("key_changes", []),
                    relation_type=data.get("relation_type", "evolution"),
                    artifact_relations=data.get("artifact_relations", []),
                )
                emit.end_task(
                    task_id,
                    status="done",
                    name=task_name,
                    text=f"Revised (delta={revised.confidence_delta})",
                )
                return revised

        emit.end_task(task_id, status="done", name=task_name, text="No output")
        return None

    except Exception as e:
        emit.status_public_error(f"Hypothesis revision failed: {e}")
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


async def upd_hypo_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    task_name: str,
    parent_module_id: str,
) -> RevisedHypothesis | None:
    """Revise hypothesis using Claude agent."""
    from aii_lib import build_options

    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        options = build_options(
            agent_cfg,
            cwd,
            task_id=task_id,
            task_name=task_name,
            system_prompt=system_prompt,
            output_format=RevisedHypothesis.to_struct_output(),
        )

        response = await Agent(options).run(prompt)

        if response.failed:
            emit.end_task(
                task_id,
                status="failed",
                name=task_name,
                text=f"FAILED: {response.error_message or 'unknown'}",
            )
            return None

        if response.structured_output:
            data = (
                response.structured_output if isinstance(response.structured_output, dict) else {}
            )
            revised = RevisedHypothesis(
                title=data.get("title", ""),
                hypothesis=data.get("hypothesis", ""),
                description=data.get("description", ""),
                relation_rationale=data.get("relation_rationale", ""),
                confidence_delta=data.get("confidence_delta", "unchanged"),
                key_changes=data.get("key_changes", []),
                relation_type=data.get("relation_type", "evolution"),
                artifact_relations=data.get("artifact_relations", []),
            )
            # Mirror typed revised hypothesis to ``task.output`` so replay-
            # execute synthesis can recover it on a subsequent fork (same
            # pattern as gen_hypo / review_hypo).
            emit.task_output(task_id=task_id, output=revised)
            emit.end_task(
                task_id,
                status="done",
                name=task_name,
                text=f"Revised (delta={revised.confidence_delta})",
            )
            return revised

        emit.end_task(task_id, status="done", name=task_name, text="No output")
        return None
    except Exception as e:
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


@dataclass
class UpdHypoCtx(ModuleCtx):
    """Substep ctx for upd_hypo."""

    parent_ctx: LoopCtx | None = None
    iteration: int = 1
    paper_text: str = ""
    reviewer_feedback_text: str | None = None
    parent_id: str = ""


class UpdHypoModule(SingleTModule):
    """upd_hypo substep — internal hypothesis revision based on evidence."""

    kind: Literal["upd_hypo_module"] = "upd_hypo_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["upd_hypo"] = "upd_hypo"

    def get_context(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        paper_text: str,
        output_dir: Path | None = None,
        reviewer_feedback_text: str | None = None,
        parent_id: str,
    ) -> UpdHypoCtx:
        return UpdHypoCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            paper_text=paper_text,
            reviewer_feedback_text=reviewer_feedback_text,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        paper_text: str,
        output_dir: Path | None = None,
        reviewer_feedback_text: str | None = None,
        parent_id: str,
    ) -> dict | None:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                paper_text=paper_text,
                output_dir=output_dir,
                reviewer_feedback_text=reviewer_feedback_text,
                parent_id=parent_id,
            )
        ):
            """Run the UPD_HYPO step.

            Returns the updated hypothesis dict (merged with original), or
            ``None`` on failure.
            """
            config = ctx.config
            hypothesis = ctx.hypothesis
            artifacts = ctx.invention_loop_group.get_artifacts()
            current_iter_artifacts = ctx.invention_loop_group.get_artifacts(iteration=iteration)
            user_uploads_path = ctx.user_uploads_path

            mid = emit.start_single_module(
                name="upd_hypo",
                parent_id=parent_id,
            )

            upd_hypo_cfg = config.invention_loop.upd_hypo
            use_claude_agent = upd_hypo_cfg.use_claude_agent

            # Step subdir
            if output_dir:
                step_dir = (output_dir / "upd_hypo").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            if use_claude_agent:
                claude_cfg = upd_hypo_cfg.claude_agent
                llm_provider = "claude_agent"
                llm_timeout = claude_cfg.agent_timeout
            else:
                llm_cfg = upd_hypo_cfg.llm_client
                llm_provider = "openrouter"
                llm_timeout = llm_cfg.llm_timeout

            emit.status_private_info(f"Provider: {llm_provider}")
            emit.status_private_info(f"Timeout: {f'{llm_timeout}s' if llm_timeout else 'None'}")

            # Build prompts
            prompt = get_user_prompt(
                hypothesis=hypothesis,
                artifacts=artifacts,
                current_iter_artifacts=current_iter_artifacts,
                paper_text=paper_text,
                iteration=iteration,
                reviewer_feedback_text=reviewer_feedback_text,
                user_folder_path=user_uploads_path,
            )
            system_prompt = get_system_prompt()

            revised = None
            try:
                if use_claude_agent:
                    claude_cfg = upd_hypo_cfg.claude_agent
                    task_id = "upd_hypo"
                    task_cwd = (output_dir / task_id) if output_dir else Path.cwd().resolve()
                    task_cwd.mkdir(parents=True, exist_ok=True)

                    revised = await upd_hypo_claude_agent(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        agent_cfg=claude_cfg,
                        cwd=task_cwd,
                        task_name=task_id,
                        parent_module_id=mid,
                    )
                else:
                    llm_cfg = upd_hypo_cfg.llm_client
                    model_cfg = llm_cfg.models[0]
                    model_name = model_cfg.model
                    task_id = "upd_hypo"

                    revised = await upd_hypo_openrouter(
                        task_name=task_id,
                        parent_module_id=mid,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model_name,
                        api_key=config.api_keys.openrouter,
                        reasoning_effort=model_cfg.reasoning_effort,
                        llm_timeout=llm_cfg.llm_timeout,
                    )

            except Exception as e:
                emit.status_public_error(f"UPD_HYPO failed: {e}")

            result_dict = None
            if revised:
                result_dict = revised.to_hypothesis_dict(hypothesis)
                emit.status_public_success(
                    f"UPD_HYPO complete: confidence {revised.confidence_delta}"
                )
                emit.status_private_info(f"Changes: {', '.join(revised.key_changes[:3])}")

                # Merge typed A↔A relations from the agent into the artifact pool's in_dependencies.
                # Each in_dependency on each artifact gets relation_type + relation_rationale set.
                if revised.artifact_relations:
                    merged = _merge_artifact_relations(artifacts, revised.artifact_relations)
                    emit.status_private_info(f"Typed {merged} A↔A edges (MultiCite)")
            else:
                emit.status_public_warning(
                    "UPD_HYPO: No revision produced — keeping original hypothesis"
                )

            emit.module_output(
                module_id=mid,
                name="upd_hypo",
                output=revised,
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return result_dict
