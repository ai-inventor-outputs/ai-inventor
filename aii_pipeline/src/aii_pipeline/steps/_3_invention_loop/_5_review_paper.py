"""REVIEW_PAPER Step - External adversarial paper review.

"Fresh eyes on the paper" — uses a DIFFERENT LLM than the one that wrote
the paper, to avoid creator bias.

Output: ReviewerFeedback with actionable critiques. Critiques that were
addressed are pruned each iteration.

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
from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
    Critique,
    DimensionScore,
    ReviewerFeedback,
)
from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.s_prompt import (
    get as get_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.u_prompt import (
    get as get_user_prompt,
)
from aii_pipeline.steps.base import ModuleCtx

from .invention_loop import LoopCtx


def _parse_feedback(data: dict, task_id: str) -> ReviewerFeedback:
    """Parse structured output into ReviewerFeedback."""
    critiques = []
    for i, c in enumerate(data.get("critiques", []), 1):
        critiques.append(
            Critique(
                id=f"crit_{i}",
                category=c.get("category", "rigor"),
                severity=c.get("severity", "minor"),
                description=c.get("description", ""),
                suggested_action=c.get("suggested_action", ""),
            )
        )

    dim_scores = []
    for d in data.get("dimension_scores", []):
        dim_scores.append(
            DimensionScore(
                dimension=d.get("dimension", ""),
                score=d.get("score", 3),
                justification=d.get("justification", ""),
                improvements=d.get("improvements", []),
            )
        )

    return ReviewerFeedback(
        id=task_id,
        overall_assessment=data.get("overall_assessment", ""),
        strengths=data.get("strengths", []),
        dimension_scores=dim_scores,
        critiques=critiques,
        score=data.get("score", 5),
        confidence=data.get("confidence", 3),
    )


async def review_paper_openrouter(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    reasoning_effort: str = "medium",
    llm_timeout: int = 600,
) -> ReviewerFeedback | None:
    """Review paper using OpenRouter chat."""
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=ReviewerFeedback,
                task_id=task_id,
                task_name=task_name,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            if output_text:
                data = json.loads(output_text)
                feedback = _parse_feedback(data, task_id)
                emit.end_task(
                    task_id,
                    status="done",
                    name=task_name,
                    text=f"Score {feedback.score}/10, {len(feedback.critiques)} critiques",
                )
                return feedback

        emit.end_task(task_id, status="done", name=task_name, text="No output")
        return None

    except Exception as e:
        emit.status_public_error(f"Paper review failed: {e}")
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


async def review_paper_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    task_name: str,
    parent_module_id: str,
) -> ReviewerFeedback | None:
    """Review paper using Claude agent."""
    from aii_lib import build_options

    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        options = build_options(
            agent_cfg,
            cwd,
            task_id=task_id,
            task_name=task_name,
            system_prompt=system_prompt,
            output_format=ReviewerFeedback.to_struct_output(),
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
            feedback = _parse_feedback(data, task_id)
            if feedback is not None:
                # Mirror typed feedback to ``task.output`` so replay-execute
                # synthesis can recover it on a subsequent fork (same pattern
                # as gen_hypo / gen_paper_text).
                emit.task_output(task_id=task_id, output=feedback)
                emit.end_task(
                    task_id,
                    status="done",
                    name=task_name,
                    text=f"Score {feedback.score}/10, {len(feedback.critiques)} critiques",
                )
            else:
                emit.end_task(task_id, status="failed", name=task_name, text="Parse failed")
            return feedback

        emit.end_task(task_id, status="done", name=task_name, text="No output")
        return None
    except Exception as e:
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


@dataclass
class ReviewPaperCtx(ModuleCtx):
    """Substep ctx for review_paper."""

    parent_ctx: LoopCtx | None = None
    iteration: int = 1
    paper_text: str = ""
    previous_critiques_text: str | None = None
    parent_id: str = ""


class ReviewPaperModule(SingleTModule):
    """review_paper substep — adversarial review of the paper draft.

    Uses a DIFFERENT LLM than ``gen_paper_text`` to provide unbiased
    review. Critiques addressed in subsequent iterations are pruned.
    """

    kind: Literal["review_paper_module"] = "review_paper_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["review_paper"] = "review_paper"

    def get_context(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        paper_text: str,
        output_dir: Path | None = None,
        previous_critiques_text: str | None = None,
        parent_id: str,
    ) -> ReviewPaperCtx:
        return ReviewPaperCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            paper_text=paper_text,
            previous_critiques_text=previous_critiques_text,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        paper_text: str,
        output_dir: Path | None = None,
        previous_critiques_text: str | None = None,
        parent_id: str,
    ) -> ReviewerFeedback | None:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                paper_text=paper_text,
                output_dir=output_dir,
                previous_critiques_text=previous_critiques_text,
                parent_id=parent_id,
            )
        ):
            """Run the REVIEW_PAPER step."""
            config = ctx.config
            artifacts = ctx.invention_loop_group.get_artifacts()
            user_uploads_path = ctx.user_uploads_path

            mid = emit.start_single_module(
                name="review_paper",
                parent_id=parent_id,
            )

            review_cfg = config.invention_loop.review_paper
            use_claude_agent = review_cfg.use_claude_agent

            # Step subdir
            if output_dir:
                step_dir = (output_dir / "review_paper").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            if use_claude_agent:
                claude_cfg = review_cfg.claude_agent
                llm_provider = "claude_agent"
                llm_timeout = claude_cfg.agent_timeout
            else:
                llm_cfg = review_cfg.llm_client
                llm_provider = "openrouter"
                llm_timeout = llm_cfg.llm_timeout

            emit.status_private_info(f"Provider: {llm_provider}")
            emit.status_public_info(f"Paper length: {len(paper_text)} chars")
            emit.status_private_info(
                f"Previous critiques: {'yes' if previous_critiques_text else 'no'}"
            )
            emit.status_private_info(f"Timeout: {f'{llm_timeout}s' if llm_timeout else 'None'}")

            # Build prompts
            prompt = get_user_prompt(
                paper_text=paper_text,
                artifacts=artifacts,
                previous_critiques_text=previous_critiques_text,
                user_folder_path=user_uploads_path,
            )
            system_prompt = get_system_prompt()

            feedback = None
            try:
                if use_claude_agent:
                    claude_cfg = review_cfg.claude_agent
                    task_id = "review_paper"
                    task_cwd = (output_dir / task_id) if output_dir else Path.cwd().resolve()
                    task_cwd.mkdir(parents=True, exist_ok=True)

                    feedback = await review_paper_claude_agent(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        agent_cfg=claude_cfg,
                        cwd=task_cwd,
                        task_name=task_id,
                        parent_module_id=mid,
                    )
                else:
                    llm_cfg = review_cfg.llm_client
                    model_cfg = llm_cfg.models[0]
                    model_name = model_cfg.model
                    task_id = "review_paper"

                    feedback = await review_paper_openrouter(
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
                emit.status_public_error(f"REVIEW_PAPER failed: {e}")

            if feedback:
                major = sum(1 for c in feedback.critiques if c.severity == "major")
                minor = len(feedback.critiques) - major
                emit.status_public_success(
                    f"REVIEW_PAPER complete: score {feedback.score}/10, {major} major + {minor} minor critiques"
                )
            else:
                emit.status_public_warning("REVIEW_PAPER: No feedback produced")

            emit.module_output(
                module_id=mid,
                name="review_paper",
                output=feedback,
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return feedback
