"""GEN_PAPER_TEXT Step - Generate paper drafts in the invention loop.

Replaces GEN_NARR — instead of narratives, produces actual paper drafts with
[FIGURE:fig_id] markers and structured figure specs each iteration.

Claude-agent-only (needs tool access for reading artifacts, web search for
literature review, structured output for paper text + figures).

Uses aii_lib for:
- Agent/AgentOptions: Claude agent calls
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aii_lib.agent_backend import Agent
from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import SingleTModule

from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
    FigureSpec,
    PaperText,
    get_figures_from_data,
    verify_figures,
)
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.s_prompt import (
    get as get_system_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.u_prompt import (
    build_figure_retry_prompt,
)
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.u_prompt import (
    get as get_user_prompt,
)
from aii_pipeline.steps.base import ModuleCtx

from .invention_loop import LoopCtx


def _build_paper_text(data: dict, iteration: int) -> PaperText:
    """Build PaperText from parsed structured output data."""
    figures = [FigureSpec(**f) if isinstance(f, dict) else f for f in data.get("figures", [])]
    return PaperText(
        id=f"paper_it{iteration}",
        title=data.get("title", ""),
        abstract=data.get("abstract", ""),
        paper_text=data.get("paper_text", ""),
        figures=figures,
        summary=data.get("summary", ""),
    )


async def gen_paper_text_claude_agent(
    prompt: str,
    system_prompt: str,
    agent_cfg,
    cwd: Path,
    iteration: int,
    task_id: str,
    task_name: str,
    verify_retries: int = 2,
) -> PaperText | None:
    """Generate paper text using Claude agent with structured JSON output."""
    from aii_lib import build_options

    # Post-validate: check figure markers <-> figures array consistency
    def validate_figures_output(structured_output):
        if not structured_output:
            return False, (
                "Your previous response did not include the required structured output. "
                "Please call the StructuredOutput tool now with the paper fields "
                "(title, abstract, paper_text, summary). This must be your final action."
            )
        data = structured_output if isinstance(structured_output, dict) else {}
        figs = get_figures_from_data(data)
        verification = verify_figures(
            paper_text=data.get("paper_text", ""),
            figures=figs,
        )
        if verification["valid"]:
            return True, None
        return False, build_figure_retry_prompt(verification=verification)

    options = build_options(
        agent_cfg,
        cwd,
        task_id=task_id,
        task_name=task_name,
        system_prompt=system_prompt,
        output_format=PaperText.to_struct_output(),
        post_validate=validate_figures_output,
        post_validate_retries=verify_retries,
    )

    response = await Agent(options).run(prompt)

    if response.failed:
        return None

    if not response.structured_output:
        return None

    data = response.structured_output if isinstance(response.structured_output, dict) else {}
    draft = _build_paper_text(data, iteration)
    # Mirror typed PaperText to ``task.output`` so replay-execute
    # synthesis can recover it on a subsequent fork. Emit the typed
    # ``draft`` (not the raw dict) so the discriminator default
    # ``kind="paper_text"`` populates before assignment to
    # ``task.output: AnyOutput`` — pydantic's tagged-union dispatch
    # runs before field defaults, so a bare dict without ``kind``
    # would fail ``union_tag_not_found`` on a subsequent reload.
    if task_id and draft is not None:
        emit.task_output(task_id=task_id, output=draft)
    return draft


@dataclass
class GenPaperTextCtx(ModuleCtx):
    """Substep ctx for gen_paper_text."""

    parent_ctx: LoopCtx | None = None
    iteration: int = 1
    previous_paper_text: str | None = None
    reviewer_feedback_text: str | None = None
    parent_id: str = ""


class GenPaperTextModule(SingleTModule):
    """gen_paper_text substep — produce paper draft from artifact pool."""

    kind: Literal["gen_paper_text_module"] = "gen_paper_text_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_paper_text"] = "gen_paper_text"

    def get_context(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        previous_paper_text: str | None = None,
        reviewer_feedback_text: str | None = None,
        parent_id: str,
    ) -> GenPaperTextCtx:
        return GenPaperTextCtx(
            config=ctx.config,
            output_dir=output_dir,
            parent_ctx=ctx,
            iteration=iteration,
            previous_paper_text=previous_paper_text,
            reviewer_feedback_text=reviewer_feedback_text,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        ctx: LoopCtx,
        iteration: int,
        output_dir: Path | None = None,
        previous_paper_text: str | None = None,
        reviewer_feedback_text: str | None = None,
        parent_id: str,
    ) -> PaperText | None:
        with ctx_scope(
            self.get_context(
                ctx=ctx,
                iteration=iteration,
                output_dir=output_dir,
                previous_paper_text=previous_paper_text,
                reviewer_feedback_text=reviewer_feedback_text,
                parent_id=parent_id,
            )
        ):
            """Run the GEN_PAPER_TEXT step.

            Generates a paper draft using Claude agent. On first run, writes from scratch.
            On subsequent runs, revises based on new artifacts + reviewer feedback.
            """
            config = ctx.config
            hypothesis = ctx.hypothesis
            artifacts = ctx.invention_loop_group.get_artifacts()
            current_iter_artifacts = ctx.invention_loop_group.get_artifacts(iteration=iteration)
            user_uploads_path = ctx.user_uploads_path

            mid = emit.start_single_module(
                name="gen_paper_text",
                parent_id=parent_id,
            )

            gen_paper_text_cfg = config.invention_loop.gen_paper_text
            claude_cfg = gen_paper_text_cfg.claude_agent
            verify_retries = gen_paper_text_cfg.verify_retries

            # Step subdir
            if output_dir:
                step_dir = (output_dir / "gen_paper_text").resolve()
                step_dir.mkdir(parents=True, exist_ok=True)
                output_dir = step_dir

            emit.status_private_info("Provider: claude_agent")
            emit.status_private_info(f"Model: {claude_cfg.model}")
            emit.status_private_info(f"Artifacts available: {len(artifacts)}")
            emit.status_private_info(f"Previous paper: {'yes' if previous_paper_text else 'no'}")
            emit.status_private_info(
                f"Reviewer feedback: {'yes' if reviewer_feedback_text else 'no'}"
            )
            emit.status_private_info(f"Verify retries: {verify_retries}")
            emit.status_private_info(
                f"Timeout: {f'{claude_cfg.agent_timeout}s' if claude_cfg.agent_timeout else 'None'}"
            )

            # Build prompts
            prompt = get_user_prompt(
                hypothesis=hypothesis,
                artifacts=artifacts,
                current_iter_artifacts=current_iter_artifacts,
                iteration=iteration,
                previous_paper_text=previous_paper_text,
                reviewer_feedback_text=reviewer_feedback_text,
                user_folder_path=user_uploads_path,
            )
            system_prompt = get_system_prompt()

            task_name = "gen_paper_text"
            task_cwd = (output_dir / task_name) if output_dir else Path.cwd().resolve() / task_name
            task_cwd.mkdir(parents=True, exist_ok=True)

            # Bracket the substep with task_start/task_end so the dashboard sees
            # a single lifecycle for the substep. The agent inside emits its own
            # agent_start/agent_end with the SDK session_id.
            task_id = emit.start_task(name=task_name, parent_module_id=mid)

            emit.status_private_info("Running paper text generator...")

            try:
                paper_text = await gen_paper_text_claude_agent(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    agent_cfg=claude_cfg,
                    cwd=task_cwd,
                    iteration=iteration,
                    task_id=task_id,
                    task_name=task_name,
                    verify_retries=verify_retries,
                )

                if paper_text:
                    # No more paper_text_pool.add — emit_module_output below writes
                    # the data and ctx.invention_loop_group.get_paper_texts()
                    # reconstitutes it from the run tree.
                    emit.status_public_success(
                        f"GEN_PAPER_TEXT complete: {len(paper_text.figures)} figures, {len(paper_text.paper_text)} chars"
                    )
                    emit.end_task(
                        task_id,
                        status="done",
                        name=task_name,
                        text=f"{len(paper_text.figures)} figs, {len(paper_text.paper_text)} chars",
                    )
                else:
                    emit.status_public_warning("GEN_PAPER_TEXT: No paper text generated")
                    emit.end_task(task_id, status="done", name=task_name, text="No output")

            except Exception as e:
                emit.status_public_error(f"GEN_PAPER_TEXT failed: {e}")
                emit.end_task(task_id, status="failed", name=task_name, text=f"Error: {e}")
                paper_text = None

            emit.module_output(
                module_id=mid,
                name="gen_paper_text",
                output=paper_text,
            )

            emit.end_module(
                parent_id=parent_id,
                module_id=mid,
            )

            return paper_text
