"""Per-substep execution functions for gen_paper_repo, run sequentially.

Order: gen_repo → gen_viz → gen_demos → gen_full_paper → deploy_gh.
Every substep starts/ends its own telemetry module. No more concurrent
gather, no more deferred-buffer deploys — each step runs to completion
before the next begins.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aii_lib.run import current_run, emit

from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
    get_figures_from_data,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
    GenArtDemoOut,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.out_schema import (
    GenPaperRepoOut,
)
from aii_pipeline.utils import rel_path

if TYPE_CHECKING:
    from ..gen_paper_repo import GenPaperCtx


# ---------------------------------------------------------------------------
# Step 1 — gen_repo
# ---------------------------------------------------------------------------


async def step_gen_repo(ctx: GenPaperCtx) -> GenPaperRepoOut | None:
    """Resolve repo URL (no network creation, no clone yet)."""
    if ctx.should_skip("gen_repo"):
        emit.status_public_progress("gen_repo [SKIPPED — reading from run tree]")
        # Resume reads typed ``Module.output`` left by the prior run's
        # ``module_output(output=GenRepoOut(...))`` event (replayed
        # from the clone log on resume).
        from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import GenRepoOut

        gpr = current_run().find_group(ctx.gen_paper_repo_gid)
        gen_repo_out = (
            next(
                (
                    m.output
                    for m in gpr.children
                    if getattr(m, "name", None) == "gen_repo"
                    and isinstance(getattr(m, "output", None), GenRepoOut)
                ),
                None,
            )
            if gpr
            else None
        )
        if gen_repo_out is None:
            raise RuntimeError(
                "gen_repo skipped but no GenRepoOut on the run tree — "
                "did the parent run record a module_output event?",
            )
        ctx.repo_url = gen_repo_out.repo_url or None
        ctx.repo_name = gen_repo_out.repo_name or None
        ctx.repo_description = gen_repo_out.description or None
        emit.status_private_info(f"Loaded repo_url: {ctx.repo_url}")
    else:
        # Prefer the most recent paper_text title (output of invention_loop's
        # last gen_paper_text iter) over the hypothesis title — paper_text
        # titles capture the actual contribution after experiments ran,
        # whereas hypothesis titles often stay frozen at iter_1's framing
        # (the upd_hypo prompt explicitly allows the LLM to keep the title
        # unchanged "if still accurate").
        latest_paper_title = ""
        if ctx._all_paper_texts:
            latest = ctx._all_paper_texts[-1]
            latest_paper_title = getattr(latest, "title", "") or ""
        repo_hypothesis = dict(ctx.hypothesis)
        if latest_paper_title:
            repo_hypothesis["title"] = latest_paper_title

        gen_repo_module = _find_substep(ctx, "gen_repo")
        repo_info = await gen_repo_module.execute(
            config=ctx.config,
            hypothesis=repo_hypothesis,
            output_dir=ctx.output_dir,
            parent_id=ctx.gen_paper_repo_gid,
        )
        ctx.repo_url = repo_info.get("repo_url") if repo_info else None
        ctx.repo_name = repo_info.get("repo_name") if repo_info else None
        ctx.repo_description = repo_info.get("description") if repo_info else None

    _check_token(ctx)

    if ctx.should_stop("gen_repo"):
        emit.status_public_progress("\n[STOPPING after gen_repo as configured]")
        return _partial(ctx, "gen_repo")
    return None


# ---------------------------------------------------------------------------
# Step 2 — gen_viz
# ---------------------------------------------------------------------------


async def step_gen_viz(ctx: GenPaperCtx) -> GenPaperRepoOut | None:
    """Generate paper figures (image generation)."""
    # Pull the latest paper text — we need its figure list to know what to draw.
    latest = ctx._all_paper_texts[-1] if ctx._all_paper_texts else None
    if latest:
        ctx.paper = latest
    paper_figures = get_figures_from_data(ctx.paper.model_dump()) if ctx.paper else []

    mid = emit.start_parallel_module(
        name="gen_viz",
        parent_id=ctx.gen_paper_repo_gid,
    )
    try:
        if ctx.should_skip("gen_viz"):
            emit.status_public_progress("gen_viz [SKIPPED — reading from run tree]")
            # Resume reads typed ``Figure`` instances from the prior
            # run's per-task ``Task.output`` (replayed from the clone
            # log). ``GenPaperRepoGroup.get_figures`` walks each
            # gen_viz module's tasks and returns ``Figure``s in order.
            gpr = current_run().find_group(ctx.gen_paper_repo_gid)
            ctx.figures = list(gpr.get_figures()) if gpr else []
            if not ctx.figures:
                emit.status_public_warning(
                    "gen_viz skipped but no Figure tasks on the run tree.",
                )
        else:
            gen_viz_module = _find_substep(ctx, "gen_viz")
            ctx.figures = await gen_viz_module.execute(
                config=ctx.config,
                figures=paper_figures,
                output_dir=ctx.output_dir,
                parent_module_id=mid,
            )
        # Per-task ``task_output`` events inside ``gen_viz_module.execute``
        # already populated ``task.output`` for each Figure;
        # ``GenPaperRepoGroup.get_figures()`` walks tasks directly.
    finally:
        emit.end_module(parent_id=ctx.gen_paper_repo_gid, module_id=mid)

    _check_token(ctx)

    if ctx.should_stop("gen_viz"):
        emit.status_public_progress("\n[STOPPING after gen_viz as configured]")
        return _partial(ctx, "gen_viz")
    return None


# ---------------------------------------------------------------------------
# Step 3 — gen_demos
# ---------------------------------------------------------------------------


async def step_gen_demos(ctx: GenPaperCtx) -> GenPaperRepoOut | None:
    """Build per-artifact demo notebooks/markdown."""
    mid = emit.start_parallel_module(
        name="gen_art_demo",
        parent_id=ctx.gen_paper_repo_gid,
    )
    try:
        if ctx.paper:
            emit.status_private_info(f"Using paper text from invention loop: {ctx.paper.id}")
        else:
            emit.status_public_warning("   No paper text from invention loop")

        if ctx.should_skip("gen_demos"):
            emit.status_public_progress("gen_demos [SKIPPED — reading from run tree]")
            # Resume reads ``GenArtDemoOut.demos`` from the prior
            # run's ``module_output`` event (replayed from the clone
            # log). ``GenPaperRepoGroup.get_demos`` returns the typed
            # demo list directly.
            gpr = current_run().find_group(ctx.gen_paper_repo_gid)
            ctx.prepared_artifacts = list(gpr.get_demos()) if gpr else []
            if not ctx.prepared_artifacts:
                emit.status_public_warning(
                    "gen_demos skipped but no demos on the run tree.",
                )
        else:
            gen_art_demo_module = _find_substep(ctx, "gen_art_demo")
            ctx.prepared_artifacts = await gen_art_demo_module.execute(
                config=ctx.config,
                artifacts=ctx.all_artifacts,
                output_dir=ctx.output_dir,
                repo_url=ctx.repo_url,
                parent_module_id=mid,
            )
        ctx.demos = list(ctx.prepared_artifacts or [])
        emit.module_output(
            module_id=mid,
            name="gen_art_demo",
            output=GenArtDemoOut(demos=ctx.demos or []),
        )
    finally:
        emit.end_module(parent_id=ctx.gen_paper_repo_gid, module_id=mid)

    _check_token(ctx)

    if ctx.should_stop("gen_demos"):
        emit.status_public_progress("\n[STOPPING after gen_demos as configured]")
        return _partial(ctx, "gen_demos")
    return None


# ---------------------------------------------------------------------------
# Step 4 — gen_full_paper
# ---------------------------------------------------------------------------


async def step_gen_full_paper(ctx: GenPaperCtx) -> GenPaperRepoOut | None:
    """Compile the LaTeX/PDF paper draft."""
    if ctx.should_skip("gen_full_paper"):
        emit.status_public_progress(
            "gen_full_paper [SKIPPED — reading from run tree]",
        )
        # Resume reads typed ``GenPaperRepoOut`` from the prior run's
        # ``module_output(output=GenPaperRepoOut(...))`` event
        # (replayed from the clone log).
        gpr = current_run().find_group(ctx.gen_paper_repo_gid)
        prev = (
            next(
                (
                    m.output
                    for m in gpr.children
                    if getattr(m, "name", None) == "gen_full_paper"
                    and isinstance(getattr(m, "output", None), GenPaperRepoOut)
                ),
                None,
            )
            if gpr
            else None
        )
        if prev is None:
            raise RuntimeError(
                "gen_full_paper skipped but no GenPaperRepoOut on the run tree.",
            )
        ctx.result = prev
    elif not ctx.run_gen_full_paper:
        emit.status_public_progress("\ngen_full_paper [SKIPPED — disabled in config]")
        return _partial(ctx, "gen_full_paper disabled")
    elif not ctx.paper:
        emit.status_public_warning("\ngen_full_paper [SKIPPED — no paper draft]")
        return _partial(ctx, "no paper draft")
    else:
        # Resolve [ARTIFACT:id] markers → \footnote{Code: \url{...}} links
        if ctx.paper and ctx.repo_url and ctx.all_artifacts:
            from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
                resolve_artifact_markers,
            )

            resolved = resolve_artifact_markers(
                ctx.paper.paper_text, ctx.repo_url, ctx.all_artifacts
            )
            ctx.paper.paper_text = resolved
            emit.status_private_info("Resolved artifact markers → GitHub hyperlinks")

        gen_full_paper_module = _find_substep(ctx, "gen_full_paper")
        ctx.result = await gen_full_paper_module.execute(
            config=ctx.config,
            paper=ctx.paper,
            figures=ctx.figures,
            gist_deployments=ctx.gist_deployments,
            output_dir=ctx.output_dir,
            repo_url=ctx.repo_url,
            parent_id=ctx.gen_paper_repo_gid,
        )

    _check_token(ctx)

    if ctx.should_stop("gen_full_paper"):
        emit.status_public_progress("\n[STOPPING after gen_full_paper as configured]")
        return _partial(ctx, "gen_full_paper")
    return None


# ---------------------------------------------------------------------------
# Step 5 — deploy_gh (clone + push src + push demos + push paper)
# ---------------------------------------------------------------------------


async def step_deploy_gh(ctx: GenPaperCtx) -> GenPaperRepoOut | None:
    """Clone the GitHub repo, push everything in three sequential phases."""
    if ctx.should_skip("deploy_gh"):
        emit.status_public_progress("deploy_gh [SKIPPED — reading from run tree]")
        # Resume reads ``DeployGhOut.deployments`` from the prior
        # run's ``module_output`` event (replayed from the clone log).
        from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import DeployGhOut

        gpr = current_run().find_group(ctx.gen_paper_repo_gid)
        agg = (
            next(
                (
                    m.output
                    for m in gpr.children
                    if getattr(m, "name", None) == "deploy_gh"
                    and isinstance(getattr(m, "output", None), DeployGhOut)
                ),
                None,
            )
            if gpr
            else None
        )
        ctx.gist_deployments = list(agg.deployments) if agg else []
        if ctx.result:
            ctx.result.gist_deployments = ctx.gist_deployments
        return None

    deploy_cfg = ctx.config.gen_paper_repo.deploy_gh
    github_cfg = ctx.config.gen_paper_repo.github
    if not (ctx.repo_url and deploy_cfg.enabled):
        emit.status_public_warning("deploy_gh [SKIPPED — no repo_url or deploy disabled]")
        return None

    paper_pdf_path = (
        Path(ctx.result.metadata["paper_pdf"])
        if ctx.result and ctx.result.metadata.get("paper_pdf")
        else None
    )
    paper_latex_dir = paper_pdf_path.parent if paper_pdf_path else None

    deploy_gh_module = _find_substep(ctx, "deploy_gh")
    ctx.gist_deployments = await deploy_gh_module.execute(
        repo_url=ctx.repo_url,
        repo_name=ctx.repo_name or "",
        repo_description=ctx.repo_description or "",
        output_dir=ctx.output_dir,
        artifacts=ctx.all_artifacts,
        prepared_artifacts=ctx.prepared_artifacts,
        paper_pdf_path=paper_pdf_path,
        paper_latex_dir=paper_latex_dir,
        deploy_cfg=deploy_cfg,
        github_cfg=github_cfg,
        max_file_size_mb=ctx.config.max_file_size_mb,
        parent_id=ctx.gen_paper_repo_gid,
    )

    if ctx.result:
        ctx.result.gist_deployments = ctx.gist_deployments

    _check_token(ctx)

    if ctx.should_stop("deploy_gh"):
        emit.status_public_progress("\n[STOPPING after deploy_gh as configured]")
        return _partial(ctx, "deploy_gh")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_substep(ctx: GenPaperCtx, name: str):
    """Locate the typed substep Module under the gen_paper_repo seq group.

    The scaffold instantiates each substep as its typed
    ``XxxModule`` subclass (``GenRepoModule`` / ``GenVizModule`` /
    ...) under ``gen_paper_repo``'s children. Each carries an
    ``execute(...)`` method that delegates to the in-file
    ``run_*_module`` body.
    """
    gpr = current_run().find_group(ctx.gen_paper_repo_gid)
    if gpr is None:
        raise RuntimeError(
            f"step_runner: gen_paper_repo group {ctx.gen_paper_repo_gid!r} "
            f"missing from the live Run",
        )
    for m in gpr.children:
        if m.name == name:
            return m
    raise RuntimeError(
        f"step_runner: substep {name!r} not found under gen_paper_repo "
        f"(children: {[m.name for m in gpr.children]!r})",
    )


def _check_token(ctx: GenPaperCtx) -> None:
    if ctx.min_token_validity:
        from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token_fresh

        ensure_oauth_token_fresh(ctx.min_token_validity)


def _partial(ctx: GenPaperCtx, stop_reason: str) -> GenPaperRepoOut:
    """Assemble GenPaperRepoOut from partial state on early stop.

    Assemble a ``GenPaperRepoOut`` from the partial state in ``ctx``
    when an early-stop substep boundary is hit (``last_step`` matched
    or ``should_stop`` returned True). Whatever has been produced so
    far is returned — missing fields default to empty list / None.
    """
    emit.status_private_info(f"Output: {rel_path(ctx.output_dir)}")
    if ctx.repo_url:
        emit.status_private_info(f"Repo URL: {ctx.repo_url}")
    if ctx.paper:
        emit.status_private_info(f"Paper: {ctx.paper.id}")
    if ctx.figures:
        emit.status_private_info(f"Figures: {len(ctx.figures)}")

    return GenPaperRepoOut(
        paper=ctx.paper,
        figures=ctx.figures or [],
        gist_deployments=ctx.gist_deployments or [],
        repo_url=ctx.repo_url,
        output_dir=str(ctx.output_dir),
        metadata={
            "gen_full_paper_skipped": True,
            "stop_reason": stop_reason,
        },
    )
