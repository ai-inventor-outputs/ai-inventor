"""VIZ_GEN Step - Generate visualizations from paper figures.

Takes the paper's figure specs and generates actual figures.

Two backends:
  use_claude_agent=True  → Claude agent with aii-image-gen skill (Gemini 3 Pro Image)
                           Agent gets workspace + figure spec, uses skill to generate,
                           verifies output, and saves JPEG to figures/ directory.
  use_claude_agent=False → Direct Gemini image gen via OpenRouter (free_viz)
                           Pure image output from models like gemini-3-pro-image-preview.

All figures: one per placeholder, no variations, no ranking.
Output is JPEG format (Gemini's native output).

Uses aii_lib for:
- Agent + AgentOptions: Claude Code SDK agent orchestration
- OpenRouterClient: LLM client with generate_image() for pure image output
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from aii_lib.agent_backend import Agent
from aii_lib.run import current_run, emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import ParallelTModule

from aii_lib import OpenRouterClient
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import (
    Figure,
    VizFigureOutput,
    get_output_filename,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.s_prompt import (
    get as get_viz_system_prompt,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.u_prompt import (
    get as get_viz_user_prompt,
)
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig, rel_path


def _emit_viz_task_output(parent_module_id: str, task_name: str, figure: Figure | None) -> None:
    """Surface a per-task ``Figure`` on the matching run-tree task.

    Looks up the task by ``name`` (unique within a single gen_viz
    invocation) under the gen_viz module and routes the typed output
    through ``Run.task_output`` so dispatch lands it on
    ``task.output``. Skipped silently when ``figure`` is None or the
    task isn't in the parent module's children yet.
    """
    if figure is None:
        return
    run = current_run()
    parent = run.find_module(parent_module_id)
    if parent is None:
        return
    for t in parent.children:
        if t.name == task_name:
            emit.task_output(task_id=t.node_id, output=figure)
            return


# =============================================================================
# CLAUDE AGENT BACKEND
# =============================================================================


async def generate_image_viz_agent(
    figure: Figure,
    agent_cfg,
    cwd: Path,
    figures_dir: Path,
    task_id: str,
    task_name: str,
    config: PipelineConfig | None = None,
) -> Figure | None:
    """Generate a figure using Claude agent with aii-image-gen skill.

    Mirrors the invention_loop executor pattern (see
    ``executors/experiment.py``):

    * The agent's workspace is its own per-task ``cwd`` — never the
      shared ``figures_dir``. The prompt tells it to save its final
      JPEG as ``{figure_id}_v0.jpg`` in the workspace root.
    * ``expected_files_field="out_expected_files"`` enables the
      shared expected-files validation + retry loop: the agent's
      structured-output ``out_expected_files.image_path`` is checked
      against the filesystem; missing files trigger an automatic
      retry with feedback (same machinery experiment / evaluation
      executors use).
    * Once validation passes, we programmatically copy
      ``<cwd>/{figure_id}_v0.jpg`` → ``<figures_dir>/{figure_id}_v0.jpg``.
      gen_full_paper / deploy_gh both read figures from
      ``figures_dir``; the per-task CWD is scratch only.

    Returns the same Figure with figure_path filled in (pointing at
    the copy in figures_dir).
    """
    from aii_lib import build_options

    abs_cwd = Path(cwd).resolve()

    prompt = get_viz_user_prompt(figure_spec=figure, workspace_path=str(abs_cwd))
    system_prompt = get_viz_system_prompt()
    output_filename = get_output_filename(figure.id, 0)

    options = build_options(
        agent_cfg,
        abs_cwd,
        task_id=task_id,
        task_name=task_name,
        system_prompt=system_prompt,
        output_format=VizFigureOutput.to_struct_output(),
        expected_files_field="out_expected_files",
        verify_retries=getattr(agent_cfg, "verify_retries", 2),
    )

    # Route via exec_mode_router for RunPod support
    if config is not None and config.execute_env.mode == "runpod":
        from aii_pipeline.steps._3_invention_loop.executors.exec_mode_router import (
            create_and_run_agent_simple,
        )

        _, response = await create_and_run_agent_simple(
            options=options,
            prompts=prompt,
            config=config,
            compute_profile=getattr(agent_cfg, "runpod_compute_profile", "cpu_light"),
            pod_timeout=getattr(agent_cfg, "pod_timeout", None),
            pod_start_retries=getattr(agent_cfg, "pod_start_retries", None),
        )
    else:
        agent = Agent(options)
        response = await agent.run(prompt)

    if response.failed:
        emit.status_public_error(f"Agent failed: {response.error_message or 'unknown'}")
        return None

    # Expected-files validation has already confirmed the file is at
    # <cwd>/{figure_id}_v0.jpg — copy into the shared figures dir.
    src = abs_cwd / output_filename
    if not src.exists():
        emit.status_public_error(
            f"Expected image missing after agent run + validation: {src}",
        )
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    dest = figures_dir / output_filename
    shutil.copy2(src, dest)

    figure.figure_path = str(dest)
    emit.status_public_success(
        f"Image saved: {dest.name} ({dest.stat().st_size} bytes)",
    )
    return figure


# =============================================================================
# OPENROUTER (FREE_VIZ) BACKEND
# =============================================================================


async def generate_image_viz_openrouter(
    task_name: str,
    parent_module_id: str,
    figure: Figure,
    model: str,
    api_key: str,
    output_dir: Path,
    llm_timeout: int = 120,
    image_size: str | None = None,
) -> Figure | None:
    """Generate a figure using direct Gemini image generation via OpenRouter.

    Uses OpenRouterClient.generate_image() with modalities=["image"] to get
    pure image output from models like gemini-3-pro-image-preview.

    Returns the same Figure with figure_path filled in.

    Args:
        image_size: Gemini image resolution - "1K" (default), "2K" (higher), "4K" (highest)
    """
    task_id = emit.start_task(name=task_name, parent_module_id=parent_module_id)

    try:
        prompt = get_viz_user_prompt(figure)
        system_prompt = get_viz_system_prompt()

        async with OpenRouterClient(api_key=api_key, model=model, timeout=llm_timeout) as client:
            image_bytes = await client.generate_image(
                prompt=prompt,
                model=model,
                system=system_prompt,
                image_size=image_size,
                task_id=task_id,
                task_name=task_name,
            )

            if image_bytes:
                # Save JPEG bytes directly (Gemini's native output format)
                output_filename = get_output_filename(figure.id, 0)
                output_path = output_dir / output_filename

                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                emit.status_public_success(
                    f"Image saved: {output_filename} ({len(image_bytes)} bytes)"
                )
                emit.end_task(task_id, name=task_name, status="done", text="Success")

                # Return figure with figure_path set
                figure.figure_path = str(output_path)
                return figure

            emit.status_public_error(f"No image returned from {model}")
            emit.end_task(task_id, name=task_name, status="failed", text="Failed: no image")
            raise RuntimeError(f"No image returned from {model} for figure {figure.id}")

    except Exception as e:
        emit.status_public_error(f"Exception: {e}")
        emit.end_task(task_id, name=task_name, status="failed", text=f"Failed: {e}")
        raise


# =============================================================================
# MODULE RUNNER
# =============================================================================


@dataclass
class GenVizCtx(ModuleCtx):
    """Substep ctx for gen_viz — paper figures + parent module id."""

    figures: list = field(default_factory=list)
    parent_module_id: str = ""


class GenVizModule(ParallelTModule):
    """gen_viz substep — generate one figure per placeholder.

    Backend selected by ``viz_gen.use_claude_agent`` config flag.
    """

    kind: Literal["gen_viz_module"] = "gen_viz_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_viz"] = "gen_viz"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        figures: list[Figure],
        output_dir: Path | None = None,
        parent_module_id: str,
    ) -> GenVizCtx:
        return GenVizCtx(
            config=config,
            output_dir=output_dir,
            figures=list(figures),
            parent_module_id=parent_module_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        figures: list[Figure],
        output_dir: Path | None = None,
        parent_module_id: str,
    ) -> list[Figure]:
        with ctx_scope(
            self.get_context(
                config=config,
                figures=figures,
                output_dir=output_dir,
                parent_module_id=parent_module_id,
            )
        ):
            """Run the gen_viz step (step 2 in gen_paper_repo).

            Generates one figure per placeholder. Returns the list of
            ``Figure`` objects with ``figure_path`` filled in.
            """
            if not figures:
                emit.status_public_warning("No figures to generate")
                return []

            gen_paper_cfg = config.gen_paper_repo
            viz_cfg = gen_paper_cfg.viz_gen
            use_claude_agent = viz_cfg.use_claude_agent

            # Create step-scoped figures output: _2_gen_viz/figures/
            if output_dir:
                step_dir = output_dir / "_2_gen_viz"
                step_dir.mkdir(parents=True, exist_ok=True)
                figures_dir = step_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
            else:
                figures_dir = Path("./_2_gen_viz/figures")
                figures_dir.mkdir(parents=True, exist_ok=True)

            # =====================================================================
            # SETUP BACKEND
            # =====================================================================
            if use_claude_agent:
                claude_cfg = viz_cfg.claude_agent
                max_concurrent = claude_cfg.max_concurrent_agents
                llm_provider = "claude_agent"

                # Per-task agent CWD: each parallel agent gets its own dir
                # (mirrors invention_loop convention — see executors/experiment.py).
                # Without this, concurrent agents race on intermediate file writes
                # in a shared dir.
                agent_step_dir = (
                    (output_dir / "_2_gen_viz").resolve() if output_dir else Path.cwd().resolve()
                )
                agent_step_dir.mkdir(parents=True, exist_ok=True)

                emit.status_private_info(
                    "Provider: claude_agent (aii-image-gen skill, gemini-3-pro-image-preview)"
                )
                emit.status_private_info(f"Model: {claude_cfg.model}")
            else:
                api_key = config.api_keys.openrouter
                free_viz_cfg = viz_cfg.free_viz
                max_concurrent = free_viz_cfg.max_concurrent if free_viz_cfg else 10
                llm_provider = "openrouter"

                # Get image gen models
                free_viz_models = (
                    [
                        {
                            "model": m.model,
                            "llm_timeout": m.llm_timeout,
                        }
                        for m in free_viz_cfg.models
                    ]
                    if free_viz_cfg and free_viz_cfg.models
                    else []
                )
                free_viz_image_size = (
                    getattr(free_viz_cfg, "image_size", None) if free_viz_cfg else None
                )

                if not free_viz_models:
                    emit.status_public_error(
                        "No image_gen models configured in viz_gen.free_viz.models"
                    )
                    return []

                emit.status_private_info("Provider: openrouter (free_viz)")
                emit.status_private_info(
                    f"Image gen models: {[m['model'] for m in free_viz_models]}"
                )
                emit.status_private_info(f"Image size: {free_viz_image_size or '1K (default)'}")

            # =================================================================
            # Per-figure resume: check for existing generated images.
            # Each resumed figure gets a synthetic task registered + closed
            # immediately so the run-tree reader (``get_figures``) sees it
            # via ``task.output`` uniformly with freshly-generated ones.
            # =================================================================
            existing_figures: dict[str, Figure] = {}
            for resume_idx, figure in enumerate(figures):
                expected_name = get_output_filename(figure.id, 0)
                existing_path = figures_dir / expected_name
                if existing_path.exists() and existing_path.stat().st_size > 0:
                    figure.figure_path = str(existing_path)
                    existing_figures[figure.id] = figure
                    emit.status_public_info(f"{figure.id} [RESUMED - {expected_name} exists]")
                    resume_task_name = f"gen_viz_resume_{resume_idx + 1}"
                    resume_task_id = emit.start_task(
                        name=resume_task_name,
                        parent_module_id=parent_module_id,
                    )
                    emit.task_output(task_id=resume_task_id, output=figure)
                    emit.end_task(
                        resume_task_id,
                        name=resume_task_name,
                        status="done",
                        text=f"resumed: {expected_name}",
                    )

            figures_to_generate = [f for f in figures if f.id not in existing_figures]

            emit.status_private_info(
                f"Figures: {len(figures)} total, {len(existing_figures)} resumed, {len(figures_to_generate)} to generate"
            )
            emit.status_private_info(f"Max concurrent: {max_concurrent}")

            # Semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            # =====================================================================
            # BUILD TASKS (only for figures that need generation)
            # =====================================================================
            if not figures_to_generate:
                results = []
            elif use_claude_agent:

                async def run_agent_task(counter: int, figure: Figure):
                    async with semaphore:
                        # 1-based sequential per gen_viz invocation. The figure.id
                        # + model_short used to ride in the slug; both still go on
                        # the workspace path via ``figure`` / ``model_short``, but
                        # the FE tree row reads cleaner as "Visualization N".
                        task_name = f"gen_viz_{counter + 1}"
                        # Per-task CWD: _2_gen_viz/gen_viz_N/ — isolated scratch
                        # space so concurrent agents don't race on shared files.
                        task_cwd = agent_step_dir / task_name
                        task_cwd.mkdir(parents=True, exist_ok=True)
                        # Open a Task lifecycle so the FE tree shows a child row
                        # for each viz attempt — mirrors the OpenRouter path
                        # above. Without this, the gen_viz module would have no
                        # children in the tree and ``_emit_viz_task_output``'s
                        # name-based lookup would silently fail.
                        task_id = emit.start_task(
                            name=task_name,
                            parent_module_id=parent_module_id,
                        )
                        try:
                            result = await generate_image_viz_agent(
                                figure=figure,
                                agent_cfg=claude_cfg,
                                cwd=task_cwd,
                                task_id=task_id,
                                task_name=task_name,
                                figures_dir=figures_dir,
                                config=config,
                            )
                        except Exception as e:
                            emit.end_task(
                                task_id,
                                name=task_name,
                                status="failed",
                                text=f"Failed: {e}",
                            )
                            raise
                        if result is None:
                            emit.end_task(
                                task_id,
                                name=task_name,
                                status="failed",
                                text="Failed: agent returned no figure",
                            )
                        else:
                            emit.end_task(
                                task_id,
                                name=task_name,
                                status="done",
                                text="Success",
                            )
                        _emit_viz_task_output(parent_module_id, task_name, result)
                        return result

                results = await asyncio.gather(
                    *[run_agent_task(i, fig) for i, fig in enumerate(figures_to_generate)],
                    return_exceptions=True,
                )
            else:
                # OpenRouter free_viz path
                task_params = []
                for counter, figure in enumerate(figures_to_generate):
                    model_cfg = free_viz_models[counter % len(free_viz_models)]
                    task_id = f"gen_viz_{counter + 1}"
                    task_params.append(
                        {
                            "task_id": task_id,
                            "task_name": task_id,
                            "figure": figure,
                            "model_cfg": model_cfg,
                        }
                    )

                async def run_openrouter_task(params: dict):
                    async with semaphore:
                        result = await generate_image_viz_openrouter(
                            task_name=params["task_name"],
                            parent_module_id=parent_module_id,
                            figure=params["figure"],
                            model=params["model_cfg"]["model"],
                            api_key=api_key,
                            output_dir=figures_dir,
                            llm_timeout=params["model_cfg"]["llm_timeout"],
                            image_size=free_viz_image_size,
                        )
                        _emit_viz_task_output(
                            parent_module_id,
                            params["task_name"],
                            result,
                        )
                        return result

                results = await asyncio.gather(
                    *[run_openrouter_task(p) for p in task_params],
                    return_exceptions=True,
                )

            # =====================================================================
            # COLLECT RESULTS (merge existing + newly generated)
            # =====================================================================
            generated_figures: list[Figure] = list(existing_figures.values())
            for r in results:
                if isinstance(r, Exception):
                    emit.status_public_warning(f"Figure task failed: {r}")
                elif r is not None:
                    generated_figures.append(r)

            emit.status_public_success(
                f"gen_viz complete: {len(generated_figures)}/{len(figures)} figures generated"
            )

            # Save output (step-scoped: _2_gen_viz/gen_viz_results.json)
            if output_dir:
                output_file = output_dir / "_2_gen_viz" / "gen_viz_results.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "figures": [fig.model_dump() for fig in generated_figures],
                            "mode": "claude_agent" if use_claude_agent else "free_viz",
                            "metadata": {
                                "generated_at": datetime.now(UTC).isoformat(),
                                "module": "gen_viz",
                                "llm_provider": llm_provider,
                                "total_figures": len(figures),
                                "successful_figures": len(generated_figures),
                                "output_dir": str(output_dir) if output_dir else None,
                            },
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                emit.status_private_info(f"Saved to: {rel_path(output_file)}")

            return generated_figures
