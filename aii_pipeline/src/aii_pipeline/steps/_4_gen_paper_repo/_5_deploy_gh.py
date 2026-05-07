"""Step 5: deploy_gh — clone the GitHub repo and push everything.

Runs sequentially after gen_full_paper. Folds the three deploy phases that
used to be 2C / 3B / 4A into one substep: src files, demo notebooks +
per-artifact READMEs, paper PDF/LaTeX + root README, all grouped under
"deploy_gh".
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from aii_lib.run import emit
from aii_lib.run.artifact import Artifact
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import SingleTModule

from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import GistDeployment
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import rel_path

from ._gen_art_demo.gen_py_demo import github_to_colab_url
from .utils.deploy import (
    EXCLUDED_DIRS,
    EXCLUDED_EXTENSIONS,
    EXCLUDED_FILES,
    copy_paper_to_repo,
    make_copytree_ignore,
)
from .utils.readme import generate_readme_with_colab_links, write_artifact_readme

# ---------------------------------------------------------------------------
# Per-phase copy builders. Same logic as the old _2c/_3b/_4a files; folded
# here so the deploy step has one home and the file structure matches the
# new sequential numbering.
# ---------------------------------------------------------------------------


def _folder_for(art_or_prep) -> str:
    """Stable per-(iter, aid) GitHub repo folder name.

    Same shape as the on-disk prepared layout
    (``_3_gen_art_demo/iter_<N>/<aid>/``) so notebook URLs baked at
    prep time match the path deploy_gh writes them to. Title-derived
    slugs were the source of the iter-2-vs-iter-3 demo URL race
    (#3 in the errors doc) — using ``iter_<N>/<aid>`` keeps the
    identity stable regardless of how the agent retitles the artifact
    between iterations. ``iteration`` is stamped by make_artifact and
    must be >= 1; iter_0 means producer skipped invention_loop.
    """
    return f"iter_{art_or_prep.iteration}/{art_or_prep.id}"


def _build_src_copy_fn(
    artifacts: list,
    aid_to_folder: dict,
    files_added: list[str],
    max_file_size_mb: int,
):
    """Copy each artifact's workspace into {folder}/src/."""
    max_bytes = int(max_file_size_mb * 1024 * 1024)

    def _copy(clone_dir: Path) -> list[str]:
        _ignore = make_copytree_ignore(max_bytes)
        files: list[str] = []

        for artifact in artifacts:
            aid = artifact.id
            folder_name = _folder_for(artifact)
            aid_to_folder[(artifact.iteration, aid)] = folder_name

            artifact_dir = clone_dir / folder_name
            src_dir = artifact_dir / "src"
            (artifact_dir / "demo").mkdir(parents=True, exist_ok=True)
            src_dir.mkdir(parents=True, exist_ok=True)

            workspace_path = Path(artifact.workspace_path) if artifact.workspace_path else None
            if not workspace_path or not workspace_path.exists() or not workspace_path.is_dir():
                emit.status_public_warning(f"   Workspace not found for {aid}")
                continue

            src_count = 0
            skipped = 0
            for item in workspace_path.iterdir():
                if item.is_dir() and item.name in EXCLUDED_DIRS:
                    skipped += 1
                    continue
                if item.is_file() and (
                    item.name in EXCLUDED_FILES or item.suffix in EXCLUDED_EXTENSIONS
                ):
                    skipped += 1
                    continue
                dst = src_dir / item.name
                if item.is_file():
                    if item.stat().st_size > max_bytes:
                        skipped += 1
                        continue
                    shutil.copy(item, dst)
                    rel = f"{folder_name}/src/{item.name}"
                    files_added.append(rel)
                    files.append(rel)
                    src_count += 1
                elif item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True, ignore=_ignore)
                    for f in dst.rglob("*"):
                        if f.is_file():
                            if f.stat().st_size > max_bytes:
                                f.unlink()
                                skipped += 1
                                continue
                            rel = f"{folder_name}/src/{f.relative_to(src_dir)}"
                            files_added.append(rel)
                            files.append(rel)
                            src_count += 1
            if skipped:
                emit.status_public_info(f"   Skipped {skipped} temp/build items")
            emit.status_public_info(f"{folder_name}/src/ — {src_count} files")

        return files

    return _copy


def _build_demos_copy_fn(
    prepared_artifacts: list,
    artifacts: list,
    aid_to_folder: dict,
    files_added: list[str],
):
    """Copy demo notebooks/markdown into {folder}/demo/ and write per-artifact README."""

    def _copy(clone_dir: Path) -> list[str]:
        # Index by (iter, aid) — same aid recurs across iters.
        artifacts_by_key = {(a.iteration, a.id): a for a in artifacts}
        files: list[str] = []

        for prep in prepared_artifacts:
            aid = prep.id
            key = (prep.iteration, aid)
            original_artifact = artifacts_by_key.get(key)
            folder_name = _folder_for(prep)
            aid_to_folder[key] = folder_name

            artifact_dir = clone_dir / folder_name
            demo_dir = artifact_dir / "demo"
            demo_dir.mkdir(parents=True, exist_ok=True)

            demo_src = Path(prep.demo_path)
            if demo_src.exists():
                demo_count = 0
                if demo_src.is_dir():
                    for f in demo_src.iterdir():
                        if f.is_file():
                            shutil.copy(f, demo_dir / f.name)
                            rel = f"{folder_name}/demo/{f.name}"
                            files_added.append(rel)
                            files.append(rel)
                            demo_count += 1
                else:
                    shutil.copy(demo_src, demo_dir / demo_src.name)
                    rel = f"{folder_name}/demo/{demo_src.name}"
                    files_added.append(rel)
                    files.append(rel)
                    demo_count = 1
                emit.status_public_info(f"{folder_name}/demo/ — {demo_count} files")
            else:
                emit.status_public_warning(f"   Demo not found for {aid}: {demo_src}")

            write_artifact_readme(artifact_dir, original_artifact, prep, files_added)
            files.append(f"{folder_name}/README.md")

        return files

    return _copy


def _build_paper_copy_fn(
    repo_url: str,
    paper_pdf_path: Path | None,
    paper_latex_dir: Path | None,
    prepared_artifacts: list,
    artifacts: list,
    aid_to_folder: dict,
    files_added: list[str],
):
    """Copy paper PDF/LaTeX, write root README with Colab badges."""

    def _copy(clone_dir: Path) -> list[str]:
        files: list[str] = []
        has_pdf, has_latex = copy_paper_to_repo(
            repo_dir=clone_dir,
            paper_pdf_path=paper_pdf_path,
            paper_latex_dir=paper_latex_dir,
            files_added=files_added,
        )
        if has_pdf:
            files.append("paper.pdf")
        if has_latex:
            latex_dir = clone_dir / "paper_latex"
            if latex_dir.exists():
                for f in latex_dir.rglob("*"):
                    if f.is_file():
                        files.append(f"paper_latex/{f.relative_to(latex_dir)}")

        readme_content = generate_readme_with_colab_links(
            repo_url=repo_url,
            prepared_artifacts=prepared_artifacts,
            artifacts=artifacts,
            has_paper_pdf=has_pdf,
            has_paper_latex=has_latex,
            aid_to_folder=aid_to_folder,
        )
        (clone_dir / "README.md").write_text(readme_content)
        files_added.append("README.md")
        files.append("README.md")
        return files

    return _copy


# ---------------------------------------------------------------------------
# Module entry point — orchestrates clone + 3 sequential push phases.
# ---------------------------------------------------------------------------


@dataclass
class DeployGhCtx(ModuleCtx):
    """Substep ctx for deploy_gh."""

    repo_url: str = ""
    repo_name: str = ""
    repo_description: str = ""
    artifacts: list = field(default_factory=list)
    prepared_artifacts: list = field(default_factory=list)
    paper_pdf_path: Path | None = None
    paper_latex_dir: Path | None = None
    deploy_cfg: Any = None
    max_file_size_mb: int = 0
    parent_id: str = ""


class DeployGhModule(SingleTModule):
    """deploy_gh substep — clone the repo and push src/demos/paper to GitHub."""

    kind: Literal["deploy_gh_module"] = "deploy_gh_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["deploy_gh"] = "deploy_gh"

    def get_context(
        self,
        *,
        repo_url: str,
        repo_name: str,
        repo_description: str,
        output_dir: Path,
        artifacts: list,
        prepared_artifacts: list,
        paper_pdf_path: Path | None,
        paper_latex_dir: Path | None,
        deploy_cfg,
        max_file_size_mb: int,
        parent_id: str,
    ) -> DeployGhCtx:
        return DeployGhCtx(
            config=None,
            output_dir=output_dir,
            repo_url=repo_url,
            repo_name=repo_name,
            repo_description=repo_description,
            artifacts=list(artifacts) if artifacts else [],
            prepared_artifacts=list(prepared_artifacts) if prepared_artifacts else [],
            paper_pdf_path=paper_pdf_path,
            paper_latex_dir=paper_latex_dir,
            deploy_cfg=deploy_cfg,
            max_file_size_mb=max_file_size_mb,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        repo_url: str,
        repo_name: str,
        repo_description: str,
        output_dir: Path,
        artifacts: list,
        prepared_artifacts: list,
        paper_pdf_path: Path | None,
        paper_latex_dir: Path | None,
        deploy_cfg,
        github_cfg,
        max_file_size_mb: int,
        parent_id: str,
    ) -> list[GistDeployment]:
        with ctx_scope(
            self.get_context(
                repo_url=repo_url,
                repo_name=repo_name,
                repo_description=repo_description,
                output_dir=output_dir,
                artifacts=artifacts,
                prepared_artifacts=prepared_artifacts,
                paper_pdf_path=paper_pdf_path,
                paper_latex_dir=paper_latex_dir,
                deploy_cfg=deploy_cfg,
                max_file_size_mb=max_file_size_mb,
                parent_id=parent_id,
            )
        ):
            """Clone the repo, push src + demos + paper sequentially, return gist info.

            All three pushes share one "deploy_gh" group so logs interleave in
            time order under a single substep.
            """
            from .utils.incremental_deploy import PipelineDeployer

            mid = emit.start_single_module(
                name="deploy_gh",
                parent_id=parent_id,
            )

            aid_to_folder: dict[tuple[int, str], str] = {}
            files_added: list[str] = []
            deployments: list[GistDeployment] = []

            deployer = PipelineDeployer(
                repo_url=repo_url,
                repo_name=repo_name,
                repo_description=repo_description,
                output_dir=output_dir,
                max_file_size_mb=max_file_size_mb,
                chunk_max_bytes=deploy_cfg.chunk_max_mb * 1024 * 1024,
                push_timeout=deploy_cfg.push_timeout,
                min_push_interval=deploy_cfg.min_push_interval,
                commit_author_name=github_cfg.commit_author_name,
                commit_author_email=github_cfg.commit_author_email,
            )

            try:
                if not await deployer.start():
                    emit.status_public_warning("Repo deploy could not start; skipping push phases.")
                    return deployments

                # Phase 1: src
                await deployer.run_phase(
                    "src",
                    _build_src_copy_fn(artifacts, aid_to_folder, files_added, max_file_size_mb),
                )

                # Phase 2: demos + per-artifact READMEs
                await deployer.run_phase(
                    "demos",
                    _build_demos_copy_fn(prepared_artifacts, artifacts, aid_to_folder, files_added),
                )

                # Phase 3: paper PDF + LaTeX + root README
                await deployer.run_phase(
                    "paper",
                    _build_paper_copy_fn(
                        repo_url=repo_url,
                        paper_pdf_path=paper_pdf_path,
                        paper_latex_dir=paper_latex_dir,
                        prepared_artifacts=prepared_artifacts,
                        artifacts=artifacts,
                        aid_to_folder=aid_to_folder,
                        files_added=files_added,
                    ),
                )

                # Build GistDeployment records for each artifact's published demo
                for prep in prepared_artifacts:
                    aid = prep.id
                    folder_name = aid_to_folder.get((prep.iteration, aid)) or _folder_for(prep)
                    demo_path = Path(prep.demo_path)

                    if demo_path.is_dir():
                        if prep.type.value == "code":
                            ipynb_files = list(demo_path.glob("*.ipynb"))
                            main_file = ipynb_files[0].name if ipynb_files else "code_demo.ipynb"
                        else:
                            main_file = f"{aid}.md"
                        all_files = [f.name for f in demo_path.iterdir() if f.is_file()]
                    else:
                        main_file = demo_path.name
                        all_files = [main_file]

                    rel_path_str = f"{folder_name}/demo/{main_file}"
                    github_url = f"{repo_url}/blob/main/{rel_path_str}"
                    colab_url = (
                        github_to_colab_url(github_url) if prep.type.value == "code" else None
                    )

                    deployments.append(
                        GistDeployment(
                            artifact_id=aid,
                            iter=prep.iteration,
                            gist_url=github_url,
                            gist_id=folder_name,
                            files=all_files,
                            colab_url=colab_url,
                        )
                    )

                # Persist deployment record (step-scoped: _5_deploy_gh/gist_deployments.json)
                if output_dir:
                    step_dir = output_dir / "_5_deploy_gh"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    output_file = step_dir / "gist_deployments.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "deployments": [
                                    {
                                        "artifact_id": d.artifact_id,
                                        "iter": d.iter,
                                        "gist_url": d.gist_url,
                                        "gist_id": d.gist_id,
                                        "files": d.files,
                                        "colab_url": d.colab_url,
                                    }
                                    for d in deployments
                                ],
                                "metadata": {
                                    "generated_at": datetime.now(UTC).isoformat(),
                                    "module": "deploy_gh",
                                    "output_dir": str(output_dir),
                                },
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    emit.status_private_info(f"Saved to: {rel_path(output_file)}")

                from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import (
                    DeployGhOut,
                )

                emit.module_output(
                    module_id=mid,
                    name="deploy_gh",
                    output=DeployGhOut(deployments=deployments or []),
                )

                # Surface the run-level deliverables. The ``status_published``
                # event flows through the messages stream — FEs that want the
                # paper / repo URLs read them off the latest event with
                # ``kind`` matching ``paper_pdf`` / ``github_repo``.
                published_artifacts: list[Artifact] = [
                    Artifact(kind="github_repo", url=repo_url, title=repo_name),
                ]
                if paper_pdf_path is not None and paper_pdf_path.exists():
                    # ``/blob/main/paper.pdf`` opens GitHub's built-in PDF
                    # viewer — guaranteed inline display in every browser. The
                    # ``/raw/`` form serves ``application/octet-stream`` which
                    # Firefox/Safari interpret as a download.
                    published_artifacts.append(
                        Artifact(
                            kind="paper_pdf",
                            url=f"{repo_url}/blob/main/paper.pdf",
                            title="Paper PDF",
                        )
                    )
                emit.status_public_published(artifacts=published_artifacts)

                return deployments

            finally:
                deployer.cleanup()
                emit.end_module(parent_id=parent_id, module_id=mid)
