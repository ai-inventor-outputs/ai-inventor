"""gen_full_paper — step 4 in gen_paper_repo. Generate LaTeX paper and PDF.

Uses Claude Agent to:
1. Create paper.tex from paper text content
2. Insert figures at appropriate locations
3. Compile to PDF using pdflatex

Pushing the paper to GitHub is handled later by step 5 (deploy_gh).
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import SingleTModule

from aii_lib import (
    Agent,
    build_options,
    end_task,
    end_task_error,
    end_task_failure,
    end_task_success,
    end_task_timeout,
    setup_workspace,
    start_task,
)
from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
    PaperText,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import Figure
from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.out_schema import (
    FullPaper,
    GenPaperRepoOut,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.s_prompt import (
    get as get_latex_system_prompt,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.u_prompt import (
    get as get_latex_user_prompt,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.u_prompt import (
    get_expected_out_files,
    get_figures_folder,
    get_latex_filename,
    get_pdf_filename,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import GistDeployment
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig, rel_path


def _verify_figures_in_tex(
    tex_path: Path,
    figures: list[Figure],
) -> list[Figure]:
    r"""Verify all figures are referenced in paper.tex via \\includegraphics.

    Checks that each figure's filename appears in an \\includegraphics command.
    Returns list of missing figures (empty = all good).
    """
    if not tex_path.exists():
        return figures  # All missing if no tex file

    tex_content = tex_path.read_text(encoding="utf-8")

    # Extract all filenames from \includegraphics{...} commands
    # Handles optional arguments like \includegraphics[width=...]{path}
    includegraphics_paths = re.findall(r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}", tex_content)
    # Normalize: extract just filenames for comparison
    included_filenames = {Path(p).name for p in includegraphics_paths}

    missing = []
    for fig in figures:
        if not fig.figure_path:
            continue
        fig_filename = Path(fig.figure_path).name
        if fig_filename not in included_filenames:
            missing.append(fig)

    # Log results
    total = len([f for f in figures if f.figure_path])
    found = total - len(missing)
    emit.status_private_info(
        f"Figure verification: {found}/{total} figures found in paper.tex "
        f"({len(includegraphics_paths)} \\includegraphics total)",
    )

    for fig in missing:
        emit.status_public_warning(
            f"Missing figure in paper.tex: {fig.id} ({Path(fig.figure_path).name}) — '{fig.title}'",
        )

    return missing


def _build_figure_fix_prompt(missing: list[Figure]) -> str:
    """Build a prompt telling the agent to insert missing figures."""
    fig_list = "\n".join(
        f'- {fig.id}: figures/{Path(fig.figure_path).name} — "{fig.title}" (caption: "{fig.caption}")'
        for fig in missing
    )
    return (
        f"FIGURE VERIFICATION FAILED: {len(missing)} figure(s) are missing from paper.tex.\n\n"
        f"The following figures exist in the figures/ directory but have NO \\includegraphics in paper.tex:\n"
        f"{fig_list}\n\n"
        f"You MUST:\n"
        f"1. Insert each missing figure using \\begin{{figure}}[!htbp] ... \\includegraphics{{figures/filename.jpg}} ... \\end{{figure}}\n"
        f"2. Place them at appropriate locations in the paper (near where they are discussed, or at the end of the relevant section)\n"
        f"3. Use the caption provided above for each figure\n"
        f"4. Recompile the PDF with pdflatex/latexmk\n"
        f"5. Verify the fix: grep -c 'includegraphics' paper.tex\n"
    )


async def generate_paper_with_agent(
    config: PipelineConfig,
    paper: PaperText,
    figures: list[Figure],
    workspace_dir: Path,
    parent_module_id: str,
) -> tuple[Path | None, Path | None]:
    """Generate LaTeX paper and compile to PDF using Claude Agent."""
    task_name = "gen_full_paper"

    setup_workspace(workspace_dir)
    task_id = start_task(task_name, parent_module_id)

    try:
        # Copy figures to workspace
        figures_dir = workspace_dir / get_figures_folder()
        figures_dir.mkdir(parents=True, exist_ok=True)

        for fig in figures:
            if fig.figure_path:
                src = Path(fig.figure_path)
                if src.exists():
                    dst = figures_dir / src.name
                    shutil.copy(src, dst)

        # Create copies with workspace-relative paths for the prompt.
        # Do NOT mutate originals — they're shared with figure_pool, result.json, and push.
        prompt_figures = [
            fig.model_copy(
                update={"figure_path": f"{get_figures_folder()}/{Path(fig.figure_path).name}"}
            )
            if fig.figure_path
            else fig.model_copy()
            for fig in figures
        ]

        # Get agent config
        agent_cfg = config.gen_paper_repo.gen_full_paper.claude_agent

        # Post-validate: check figures are included in paper.tex
        figures_with_path = [f for f in prompt_figures if f.figure_path]

        def _validate_tex_figures(structured_output):  # noqa: ARG001 — post-validate cb sig
            tex_file = workspace_dir / get_latex_filename()
            if not tex_file.exists() or not figures_with_path:
                return True, None
            missing = _verify_figures_in_tex(tex_file, prompt_figures)
            if not missing:
                return True, None
            return False, _build_figure_fix_prompt(missing)

        options = build_options(
            agent_cfg,
            workspace_dir,
            task_id=task_id,
            task_name=task_name,
            system_prompt=get_latex_system_prompt(),
            output_format=FullPaper.to_struct_output(),
            expected_files_field="out_expected_files",
            post_validate=_validate_tex_figures,
            post_validate_retries=2,
        )

        # Build prompt (GitHub push is handled by Python code, not agent)
        prompt = get_latex_user_prompt(
            paper=paper,
            figures=prompt_figures,
            workspace_path=str(workspace_dir),
        )

        emit.status_private_info("Starting LaTeX generation and compilation")

        # Run agent
        agent = Agent(options)
        result = await agent.run([prompt])

        if result.failed:
            err = result.error_message or "unknown error"
            emit.status_public_error(f"GEN_PAPER agent failed: {err}")
            end_task_failure(task_id, task_name, f"Agent failed: {err}")
            raise RuntimeError(f"GEN_PAPER agent failed: {err}")

        # Check output files
        tex_path = workspace_dir / get_latex_filename()
        pdf_path = workspace_dir / get_pdf_filename()

        if pdf_path.exists():
            emit.status_public_success(f"PDF generated: {tex_path.name}")
            end_task_success(task_id, task_name)
            return tex_path, pdf_path

        if tex_path.exists():
            emit.status_public_warning("LaTeX created but PDF compilation failed")
            end_task(task_id, task_name, "done", text="Partial")
            return tex_path, None

        emit.status_public_error("LaTeX generation failed - no output files")
        end_task_failure(task_id, task_name, "No output files")
        raise RuntimeError("LaTeX generation produced no output files")

    except TimeoutError:
        emit.status_public_error("GEN_PAPER agent timed out")
        end_task_timeout(task_id, task_name, agent_cfg.seq_prompt_timeout)
        raise

    except Exception as e:
        emit.status_public_error(f"GEN_PAPER failed: {e}")
        end_task_error(task_id, task_name, str(e))
        raise


@dataclass
class GenFullPaperCtx(ModuleCtx):
    """Substep ctx for gen_full_paper."""

    paper: Any = None  # PaperText | None
    figures: list = field(default_factory=list)
    gist_deployments: list | None = None
    repo_url: str | None = None
    parent_id: str = ""


class GenFullPaperModule(SingleTModule):
    """gen_full_paper substep — generate LaTeX + compile PDF.

    Uses Claude Agent to generate LaTeX from the paper draft and
    compile it to PDF. The push to GitHub happens in step 5
    (``deploy_gh``).
    """

    kind: Literal["gen_full_paper_module"] = "gen_full_paper_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_full_paper"] = "gen_full_paper"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        paper: PaperText | None,
        figures: list[Figure],
        gist_deployments: list[GistDeployment] | None = None,
        output_dir: Path | None = None,
        repo_url: str | None = None,
        parent_id: str,
    ) -> GenFullPaperCtx:
        return GenFullPaperCtx(
            config=config,
            output_dir=output_dir,
            paper=paper,
            figures=list(figures),
            gist_deployments=list(gist_deployments) if gist_deployments else None,
            repo_url=repo_url,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        paper: PaperText | None,
        figures: list[Figure],
        gist_deployments: list[GistDeployment] | None = None,
        output_dir: Path | None = None,
        repo_url: str | None = None,
        parent_id: str,
    ) -> GenPaperRepoOut:
        with ctx_scope(
            self.get_context(
                config=config,
                paper=paper,
                figures=figures,
                gist_deployments=gist_deployments,
                output_dir=output_dir,
                repo_url=repo_url,
                parent_id=parent_id,
            )
        ):
            """Run the gen_full_paper step."""
            mid = emit.start_single_module(
                name="gen_full_paper",
                parent_id=parent_id,
            )

            try:
                if not output_dir:
                    output_dir = Path("./gen_paper_output")

                output_dir.mkdir(parents=True, exist_ok=True)
                # Step-scoped output: 4_gen_paper_repo/_4_assemble_paper/paper/
                step_dir = output_dir / "_4_assemble_paper"
                step_dir.mkdir(parents=True, exist_ok=True)
                paper_dir = step_dir / "paper"
                paper_dir.mkdir(parents=True, exist_ok=True)

                result = GenPaperRepoOut(
                    output_dir=str(output_dir),
                    repo_url=repo_url,
                    figures=figures,
                    gist_deployments=gist_deployments or [],
                    metadata={
                        "generated_at": datetime.now(UTC).isoformat(),
                        "module": "gen_paper",
                    },
                )

                if not paper:
                    emit.status_public_warning("No paper to process")
                    return result

                emit.status_private_info(f"Paper: {paper.id}")
                emit.status_private_info(f"Figures: {len(figures)}")
                if gist_deployments:
                    emit.status_private_info(f"Gists: {len(gist_deployments)}")
                if repo_url:
                    emit.status_private_info(f"Repo: {repo_url}")

                if not paper.title:
                    raise ValueError("Paper has no title — cannot generate paper")

                # Generate paper with Claude Agent (GitHub push handled separately)
                workspace_dir = paper_dir / "workspace"
                tex_path, pdf_path = await generate_paper_with_agent(
                    config=config,
                    paper=paper,
                    figures=figures,
                    workspace_dir=workspace_dir,
                    parent_module_id=mid,
                )

                # Copy final outputs to paper_dir
                final_tex = None
                final_pdf = None

                if tex_path and tex_path.exists():
                    final_tex = paper_dir / get_latex_filename()
                    shutil.copy(tex_path, final_tex)
                    emit.status_private_info(f"LaTeX: {rel_path(final_tex)}")

                if pdf_path and pdf_path.exists():
                    final_pdf = paper_dir / get_pdf_filename()
                    shutil.copy(pdf_path, final_pdf)
                    emit.status_private_info(f"PDF: {rel_path(final_pdf)}")

                # Copy references.bib if it exists
                final_bib = None
                bib_path = workspace_dir / "references.bib"
                if bib_path.exists():
                    final_bib = paper_dir / "references.bib"
                    shutil.copy(bib_path, final_bib)
                    emit.status_private_info(f"Bibliography: {rel_path(final_bib)}")

                # Copy figures to paper_dir from gen_viz output directory.
                # Uses the known absolute location instead of fig.figure_path which may
                # have been changed to relative paths during agent prompt construction.
                figures_out = paper_dir / get_figures_folder()
                figures_out.mkdir(parents=True, exist_ok=True)
                gen_viz_figures_dir = output_dir / "_2_gen_viz" / "figures"
                if gen_viz_figures_dir.exists():
                    for fig_file in gen_viz_figures_dir.iterdir():
                        if fig_file.is_file():
                            shutil.copy(fig_file, figures_out / fig_file.name)

                # Paper PDF is pushed to GitHub in the deploy_gh_paper step (not here)

                # Update result
                result.paper = paper
                # Sync the nested figures with the populated top-level list
                # so ``paper.figures`` doesn't ship as null-everywhere
                # (the LLM-emitted ``PaperText.figures`` carries the spec
                # only — figure_path / aspect_ratio land on
                # ``result.figures`` after gen_viz). Was the source of
                # `paper.figures[].figure_path: None` in gen_paper_result.json
                # (errors-doc #46) which would crash any downstream reader
                # that touched ``paper.figures``.
                if result.figures:
                    result.paper.figures = result.figures
                result.metadata["paper_tex"] = str(final_tex) if final_tex else None
                result.metadata["paper_pdf"] = (
                    str(final_pdf) if final_pdf and final_pdf.exists() else None
                )
                result.metadata["repo_url"] = repo_url
                result.metadata["expected_files"] = [f.path for f in get_expected_out_files()]
                result.metadata["llm_provider"] = "claude_agent"
                result.metadata["output_dir"] = str(output_dir) if output_dir else None

                # Save final result (step-scoped: _4_assemble_paper/gen_paper_result.json)
                result_file = step_dir / "gen_paper_result.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
                emit.status_private_info(f"Saved result: {rel_path(result_file)}")

                if final_pdf and final_pdf.exists():
                    emit.status_public_success(f"gen_full_paper complete: {rel_path(final_pdf)}")
                elif final_tex and final_tex.exists():
                    emit.status_public_success(
                        f"gen_full_paper complete (LaTeX only): {rel_path(final_tex)}"
                    )
                else:
                    emit.status_public_warning("gen_full_paper complete (no outputs)")

                emit.module_output(
                    module_id=mid,
                    name="gen_full_paper",
                    output=result,
                )
                return result

            finally:
                emit.end_module(parent_id=parent_id, module_id=mid)
