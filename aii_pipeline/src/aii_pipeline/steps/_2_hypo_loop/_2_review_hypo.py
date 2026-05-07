"""REVIEW_HYPO Step - External hypothesis review before invention loop.

Different LLM reviews the hypothesis adversarially. Single pass — one
review per call. Feedback feeds back into the next ``gen_hypo``
iteration at the ``gen_hypo_loop`` level (no inner revise loop).

Flow: gen_hypo → review → (next iteration of gen_hypo_loop)

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
from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.out_schema import (
    HypoReviewerFeedback,
    ReviewHypoOut,
)
from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.s_prompt import (
    get_review as get_review_sysprompt,
)
from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.u_prompt import (
    get_review as get_review_prompt,
)
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig

# =============================================================================
# REVIEW — adversarial review of hypothesis
# =============================================================================


async def _review_openrouter(
    task_name: str,
    parent_module_id: str,
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    reasoning_effort: str = "medium",
    llm_timeout: int = 600,
) -> HypoReviewerFeedback | None:
    """Review hypothesis using OpenRouter chat."""
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            result = await chat(
                client=client,
                prompt=prompt,
                system=system_prompt,
                reasoning_effort=reasoning_effort,
                response_format=HypoReviewerFeedback,
                task_id=task_id,
                task_name=task_name,
                timeout=llm_timeout,
            )

            output_text = client.extract_json_from_response(result.response)
            if output_text:
                data = json.loads(output_text)
                from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
                    Critique,
                    DimensionScore,
                )

                critiques = [
                    Critique(**c) if isinstance(c, dict) else c for c in data.get("critiques", [])
                ]
                dim_scores = [
                    DimensionScore(**d) if isinstance(d, dict) else d
                    for d in data.get("dimension_scores", [])
                ]
                feedback = HypoReviewerFeedback(
                    id=task_id,
                    overall_assessment=data.get("overall_assessment", ""),
                    strengths=data.get("strengths", []),
                    dimension_scores=dim_scores,
                    critiques=critiques,
                    score=data.get("score", 5),
                    confidence=data.get("confidence", 3),
                    relation_type=data.get("relation_type"),
                    relation_rationale=data.get("relation_rationale", ""),
                )
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
        emit.status_public_error(f"Hypothesis review failed: {e}")
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


async def _review_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    task_name: str,
    parent_module_id: str,
) -> HypoReviewerFeedback | None:
    """Review hypothesis using Claude agent."""
    from aii_lib import build_options
    from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
        Critique,
        DimensionScore,
    )

    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        options = build_options(
            agent_cfg,
            cwd,
            task_id=task_id,
            task_name=task_name,
            system_prompt=system_prompt,
            output_format=HypoReviewerFeedback.to_struct_output(),
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
            critiques = [
                Critique(**c) if isinstance(c, dict) else c for c in data.get("critiques", [])
            ]
            dim_scores = [
                DimensionScore(**d) if isinstance(d, dict) else d
                for d in data.get("dimension_scores", [])
            ]
            feedback = HypoReviewerFeedback(
                id=task_id,
                overall_assessment=data.get("overall_assessment", ""),
                strengths=data.get("strengths", []),
                dimension_scores=dim_scores,
                critiques=critiques,
                score=data.get("score", 5),
                confidence=data.get("confidence", 3),
                relation_type=data.get("relation_type"),
                relation_rationale=data.get("relation_rationale", ""),
            )
            # Mirror to ``task.output`` so replay-execute synthesis can
            # recover it on a future fork (see gen_hypo for the same
            # pattern + rationale).
            emit.task_output(task_id=task_id, output=feedback)
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
        emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
        raise


# =============================================================================
# Typed substep class
# =============================================================================


@dataclass
class ReviewHypoCtx(ModuleCtx):
    """Substep ctx for review_hypo."""

    hypothesis: dict = None  # type: ignore[assignment]
    iteration: int = 1
    previous_feedback_text: str | None = None
    previous_hypothesis: dict | None = None
    user_uploads_path: str = ""
    parent_id: str = ""


class ReviewHypoModule(SingleTModule):
    """review_hypo substep — adversarial review of a hypothesis.

    Single-pass review (no revision). Feedback feeds back into the
    next ``gen_hypo`` iteration at the ``hypo_loop`` level.
    """

    kind: Literal["review_hypo_module"] = "review_hypo_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["review_hypo"] = "review_hypo"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        hypothesis: dict,
        iteration: int = 1,
        output_dir: Path | None = None,
        previous_feedback_text: str | None = None,
        previous_hypothesis: dict | None = None,
        user_uploads_path: str = "",
        parent_id: str,
    ) -> ReviewHypoCtx:
        return ReviewHypoCtx(
            config=config,
            output_dir=output_dir,
            hypothesis=hypothesis,
            iteration=iteration,
            previous_feedback_text=previous_feedback_text,
            previous_hypothesis=previous_hypothesis,
            user_uploads_path=user_uploads_path,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        hypothesis: dict,
        iteration: int = 1,
        output_dir: Path | None = None,
        previous_feedback_text: str | None = None,
        previous_hypothesis: dict | None = None,
        user_uploads_path: str = "",
        parent_id: str,
    ) -> ReviewHypoOut:
        with ctx_scope(
            self.get_context(
                config=config,
                hypothesis=hypothesis,
                iteration=iteration,
                output_dir=output_dir,
                previous_feedback_text=previous_feedback_text,
                previous_hypothesis=previous_hypothesis,
                user_uploads_path=user_uploads_path,
                parent_id=parent_id,
            )
        ):
            mid = emit.start_single_module(
                name="review_hypo",
                parent_id=parent_id,
            )

            review_cfg = config.review_hypo
            use_claude_agent = review_cfg.use_claude_agent

            if use_claude_agent:
                review_agent_cfg = review_cfg.claude_agent
                llm_provider = "claude_agent"
            else:
                review_llm_cfg = review_cfg.llm_client
                llm_provider = "openrouter"

            emit.status_private_info(f"Provider: {llm_provider}")
            emit.status_private_info(f"Hypothesis: {hypothesis.get('title', 'N/A')}")

            # === REVIEW (single pass) ===
            review_prompt = get_review_prompt(
                hypothesis=hypothesis,
                previous_feedback_text=previous_feedback_text,
                previous_hypothesis=previous_hypothesis,
                user_folder_path=user_uploads_path,
            )
            review_sysprompt = get_review_sysprompt()

            feedback = None
            try:
                if use_claude_agent:
                    task_id = "review_hypo"
                    task_cwd = output_dir if output_dir else Path.cwd().resolve()
                    feedback = await _review_claude_agent(
                        prompt=review_prompt,
                        system_prompt=review_sysprompt,
                        agent_cfg=review_agent_cfg,
                        cwd=task_cwd,
                        task_name=task_id,
                        parent_module_id=mid,
                    )
                else:
                    model_cfg = review_llm_cfg.models[0]
                    task_id = "review_hypo"
                    feedback = await _review_openrouter(
                        task_name=task_id,
                        parent_module_id=mid,
                        prompt=review_prompt,
                        system_prompt=review_sysprompt,
                        model=model_cfg.model,
                        api_key=config.api_keys.openrouter,
                        reasoning_effort=model_cfg.reasoning_effort,
                        llm_timeout=review_llm_cfg.llm_timeout,
                    )
            except Exception as e:
                emit.status_public_error(f"   Review failed: {e}")

            if feedback:
                major = sum(1 for c in feedback.critiques if c.severity == "major")
                minor = len(feedback.critiques) - major
                emit.status_public_info(
                    f"Review: score {feedback.score}/10, {major} major + {minor} minor"
                )
            else:
                emit.status_public_warning("   Review: no feedback produced")

            # Build result — hypothesis unchanged, feedback attached
            result = ReviewHypoOut(
                hypothesis=hypothesis,
                final_review=feedback.model_dump() if feedback else None,
            )

            emit.status_public_success("REVIEW_HYPO complete")
            if feedback:
                emit.status_private_info(f"Score: {feedback.score}/10")

            emit.module_output(
                module_id=mid,
                name="review_hypo",
                output=result,
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return result
