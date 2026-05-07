"""gen_repo — step 1 in gen_paper_repo. Resolve GitHub repository name and URL.

Determines the repo name, owner, and URL for later steps (gen_demos,
gen_viz, etc.) but does NOT create the repo on GitHub. Actual creation
is deferred to the deploy step so failed runs don't leave empty
GitHub repositories.

Uses aii_lib.utils.deploy_github for repo name generation and URL resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import SingleTModule
from aii_lib.utils.deploy_github import resolve_repo_url

from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig, rel_path

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GenRepoCtx(ModuleCtx):
    """Substep ctx for gen_repo — config + hypothesis + parent_id."""

    hypothesis: dict = None  # type: ignore[assignment]
    parent_id: str = ""


class GenRepoModule(SingleTModule):
    """gen_repo substep — resolve GitHub repository name + URL (no creation).

    Generates the repo name from the hypothesis title + a unique id and
    resolves the GitHub owner via ``gh api user``. Does not create the
    repo on GitHub — actual ``gh repo create`` is deferred to
    :class:`DeployGhModule` so failed runs don't leave empty repos.
    """

    kind: Literal["gen_repo_module"] = "gen_repo_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_repo"] = "gen_repo"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        hypothesis: dict,
        output_dir: Path | None = None,
        parent_id: str,
    ) -> GenRepoCtx:
        return GenRepoCtx(
            config=config,
            output_dir=output_dir,
            hypothesis=hypothesis,
            parent_id=parent_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        hypothesis: dict,
        output_dir: Path | None = None,
        parent_id: str,
    ) -> dict:
        with ctx_scope(
            self.get_context(
                config=config,
                hypothesis=hypothesis,
                output_dir=output_dir,
                parent_id=parent_id,
            )
        ):
            # Fork inheritance: if repo_info.json was copied in from a parent run,
            # reuse the parent's URL instead of minting a new random one. The
            # deployer's `ensure_repo_exists` will detect the repo on GitHub and
            # reuse it, so the fork's commits land on top of the parent's history
            # in the same repo.
            if output_dir:
                # Step-scoped output: 4_gen_paper_repo/_1_gh_repo/repo_info.json
                step_dir = output_dir / "_1_gh_repo"
                existing = step_dir / "repo_info.json"
                if existing.exists():
                    try:
                        cached = json.loads(existing.read_text(encoding="utf-8"))
                        if cached.get("repo_url"):
                            emit.status_private_info(
                                f"gen_repo: reusing inherited repo_url: {cached['repo_url']}",
                            )
                            return cached
                    except (OSError, json.JSONDecodeError) as e:
                        emit.status_public_warning(
                            f"gen_repo: could not read inherited repo_info.json ({e}); regenerating.",
                        )

            mid = emit.start_single_module(
                name="gen_repo",
                parent_id=parent_id,
            )

            try:
                # gh CLI's token comes from gh_paper_env() inside resolve_repo_url
                # (priority: AII_GH_TOKEN → GH_TOKEN → GITHUB_TOKEN). The pipeline
                # loads .env at startup so AII_GH_TOKEN is already in os.environ
                # by the time we reach this code path; no manual .env scrape
                # needed here.
                title = hypothesis.get("title", "research-project")
                description = hypothesis.get("hypothesis", "AI-generated research project")[:200]

                result = resolve_repo_url(
                    title,
                    prefix=config.gen_paper_repo.github.repo_prefix,
                )
                result["description"] = description
                # gen_repo only resolves the URL — it doesn't create the repo
                # (deploy_gh's ``ensure_repo_exists`` does the actual create).
                # ``created`` here was hardcoded False which read as "we
                # reused an existing repo" but actually meant "gen_repo
                # didn't create yet". Drop it; deploy_gh now surfaces the
                # real reuse-vs-create signal as a public warning at push
                # time (see ``repo_ops.ensure_repo_exists``).

                if result["error"]:
                    emit.status_public_warning(f"   {result['error']}")
                else:
                    emit.status_private_info(f"Repo URL: {result['repo_url']}")

                # Save output (step-scoped: _1_gh_repo/repo_info.json)
                if output_dir:
                    step_dir = output_dir / "_1_gh_repo"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    output_file = step_dir / "repo_info.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                **result,
                                "metadata": {
                                    "generated_at": datetime.now(UTC).isoformat(),
                                    "module": "gen_repo",
                                    "llm_provider": "gh_cli",
                                    "output_dir": str(output_dir) if output_dir else None,
                                },
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    emit.status_private_info(f"Saved to: {rel_path(output_file)}")

                from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import (
                    GenRepoOut,
                )

                emit.module_output(
                    module_id=mid,
                    name="gen_repo",
                    output=GenRepoOut(
                        repo_url=result.get("repo_url") or "",
                        repo_name=result.get("repo_name") or "",
                        description=result.get("description") or "",
                        created=bool(result.get("created")),
                        error=result.get("error") or "",
                    ),
                )
                return result

            finally:
                emit.end_module(parent_id=parent_id, module_id=mid)
