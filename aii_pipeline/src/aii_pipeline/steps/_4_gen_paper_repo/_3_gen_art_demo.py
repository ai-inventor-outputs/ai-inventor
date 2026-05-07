"""gen_demos — step 3 in gen_paper_repo. Generate demo versions of artifacts.

Routes each artifact to the appropriate demo converter based on file type:
- Python scripts (.py) → Jupyter notebooks (gen_py_demo)
- Lean proofs (.lean) → Markdown with playground link (gen_lean_demo)
- Research JSON (.json) → Formatted markdown (gen_md_demo)

Output structure:
- {artifact_id}/demo/: Self-contained demos (notebooks, markdown)
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from aii_lib.agent_backend import ExpectedFile
from aii_lib.run import emit
from aii_lib.run.context import ctx_scope
from aii_lib.run.module import ParallelTModule
from pydantic import TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.out_schema import (
    DatasetArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.out_schema import (
    EvaluationArtifact,
)

# Import artifact schemas to get demo_files and expected_files for each type
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.out_schema import (
    ExperimentArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.out_schema import (
    ProofArtifact,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.out_schema import (
    ResearchArtifact,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
    AnyDemo,
    BaseDemo,
    CodeDemo,
    DemoExpectedFiles,
    LeanDemo,
    MarkdownDemo,
)
from aii_pipeline.steps.base import ModuleCtx
from aii_pipeline.utils import PipelineConfig, rel_path

_demo_adapter = TypeAdapter(AnyDemo)

# Per-type demo converters
from ._gen_art_demo.gen_lean_demo import create_proof_markdown, lean_playground_url
from ._gen_art_demo.gen_md_demo import create_research_markdown
from ._gen_art_demo.gen_py_demo import (  # noqa: F401  (re-exported by utils.readme)
    convert_to_notebook,
    github_to_colab_url,
)

# Map artifact types to their schema classes
ARTIFACT_SCHEMAS = {
    "experiment": ExperimentArtifact,
    "evaluation": EvaluationArtifact,
    "dataset": DatasetArtifact,
    "proof": ProofArtifact,
    "research": ResearchArtifact,
}


def get_expected_out_files_for_type(artifact_type: str) -> list[ExpectedFile]:
    """Get expected output files for this artifact type.

    Used for copying dependencies and verification.

    Returns:
        List of ExpectedFile objects from the artifact schema.
    """
    schema = ARTIFACT_SCHEMAS.get(artifact_type)
    if schema:
        return schema.get_expected_out_files()
    return []


@dataclass
class GenArtDemoCtx(ModuleCtx):
    """Substep ctx for gen_art_demo."""

    artifacts: list = field(default_factory=list)
    artifact_workspaces: dict | None = None
    repo_url: str | None = None
    parent_module_id: str = ""


class GenArtDemoModule(ParallelTModule):
    """gen_art_demo substep — produce demo notebooks / markdowns / lean.

    Class name, file path, and wire-level ``name`` slug all use
    ``gen_art_demo`` as the canonical token.
    """

    kind: Literal["gen_art_demo_module"] = "gen_art_demo_module"
    """Per-subclass discriminator (see ``GenHypoModule.kind``)."""

    name: Literal["gen_art_demo"] = "gen_art_demo"

    def get_context(
        self,
        *,
        config: PipelineConfig,
        artifacts: list,
        output_dir: Path | None = None,
        artifact_workspaces: dict[str, Path] | None = None,
        repo_url: str | None = None,
        parent_module_id: str,
    ) -> GenArtDemoCtx:
        return GenArtDemoCtx(
            config=config,
            output_dir=output_dir,
            artifacts=list(artifacts) if artifacts else [],
            artifact_workspaces=artifact_workspaces,
            repo_url=repo_url,
            parent_module_id=parent_module_id,
        )

    async def execute(
        self,
        *,
        config: PipelineConfig,
        artifacts: list,
        output_dir: Path | None = None,
        artifact_workspaces: dict[str, Path] | None = None,
        repo_url: str | None = None,
        parent_module_id: str,
    ) -> list[BaseDemo]:
        with ctx_scope(
            self.get_context(
                config=config,
                artifacts=artifacts,
                output_dir=output_dir,
                artifact_workspaces=artifact_workspaces,
                repo_url=repo_url,
                parent_module_id=parent_module_id,
            )
        ):
            """Run the gen_demos step.

            No dependencies on prior gen_paper_repo substeps beyond
            gen_repo's URL. Collects files from artifact workspaces using
            ``get_github_deploy_outputs()`` and converts them to demo-ready
            formats. Output: ``{artifact_id}/src/`` and ``{artifact_id}/demo/``.
            """
            if not artifacts:
                emit.status_public_warning("No artifacts to prepare")
                return []

            if not output_dir:
                output_dir = Path("./prepared_artifacts")

            # Step-scoped output: 4_gen_paper_repo/_3_gen_art_demo/{artifact_id}/...
            # Mirrors the convention 3_invention_loop/iter_N/gen_art/{artifact_id}/
            # uses (parent step folder + per-task subfolder).
            step_dir = output_dir / "_3_gen_art_demo"
            step_dir.mkdir(parents=True, exist_ok=True)

            # Per-artifact folder structure: _3_gen_art_demo/{artifact_id}/{src,demo}/
            emit.status_public_info(f"Artifacts: {len(artifacts)}")
            emit.status_private_info(f"Output: {rel_path(step_dir)}")
            emit.status_private_info(
                "Structure: _3_gen_art_demo/iter_<N>/{artifact_id}/{src,demo}/"
            )

            def _per_iter_demo_dir(art) -> Path:
                """Per-iteration demo directory to avoid artifact id collisions.

                Per-iter folder so artifacts with the same id from
                different invention_loop iters don't collide. iteration
                is stamped by make_artifact and is always >= 1; if it's
                0 the artifact wasn't routed through invention_loop and
                that's a producer bug — surface it as iter_0/ rather
                than silently flattening.
                """
                return step_dir / f"iter_{art.iteration}" / art.id

            # =====================================================================
            # Per-artifact resume: load existing completed demos
            # =====================================================================
            existing_demos: dict[tuple[int, str], BaseDemo] = {}
            for artifact in artifacts:
                demo_result_path = _per_iter_demo_dir(artifact) / "demo_result.json"
                if demo_result_path.exists():
                    try:
                        with open(demo_result_path, encoding="utf-8") as f:
                            data = json.load(f)
                        demo = _demo_adapter.validate_python(data)
                        existing_demos[(artifact.iteration, artifact.id)] = demo
                    except Exception as e:
                        emit.status_public_warning(
                            f"   Failed to load existing demo for iter_{artifact.iteration}/{artifact.id}: {e}"
                        )

            if existing_demos:
                emit.status_public_info(
                    f"Resuming: {len(existing_demos)} demos already completed, skipping"
                )

            prepared: list[BaseDemo] = list(existing_demos.values())
            notebook_tasks = []

            for artifact in artifacts:
                # Skip already-completed artifacts (keyed by iter+id since
                # the same aid recurs across iters — see #14 in errors doc).
                key = (artifact.iteration, artifact.id)
                if key in existing_demos:
                    emit.status_public_info(
                        f"iter_{artifact.iteration}/{artifact.id} [RESUMED - already exists]"
                    )
                    continue
                aid = artifact.id
                artifact_type = (
                    artifact.type.value if hasattr(artifact.type, "value") else str(artifact.type)
                )

                # Get the demo file from artifact (first entry)
                demo_files = artifact.out_demo_files
                demo_file = demo_files[0].path if demo_files else None

                # Get workspace path for this artifact
                workspace_path = None
                if artifact_workspaces and aid in artifact_workspaces:
                    workspace_path = artifact_workspaces[aid]
                elif artifact.workspace_path:
                    workspace_path = Path(artifact.workspace_path)

                emit.status_public_info(
                    f"Processing iter_{artifact.iteration}/{aid} ({artifact_type}): {demo_file or 'no demo'}"
                )

                # Skip if no demo file
                if demo_file is None:
                    emit.status_public_info(f"Skipping {aid} - no demo file")
                    continue

                # Create per-iter demo output directory
                demo_dir = _per_iter_demo_dir(artifact)
                demo_dir.mkdir(parents=True, exist_ok=True)

                # Get workspace path and validate it exists
                if not workspace_path or not workspace_path.exists():
                    emit.status_public_warning(f"   No workspace for {aid}")
                    continue

                # Route to appropriate converter based on file extension
                if demo_file.endswith(".lean"):
                    # Lean proof -> Markdown with playground link
                    script_path = workspace_path / demo_file
                    if not script_path.exists():
                        emit.status_public_warning(f"   Demo file not found: {demo_file}")
                        continue

                    lean_code = script_path.read_text()
                    playground_url = lean_playground_url(lean_code)
                    md_content, _ = create_proof_markdown(aid, lean_code, artifact)

                    demo_path = demo_dir / f"{aid}.md"
                    demo_path.write_text(md_content)

                    lean_demo = LeanDemo(
                        id=aid,
                        iteration=artifact.iteration,
                        title=artifact.title or f"Lean proof: {aid}",
                        summary=artifact.summary or "Formal proof with Lean playground link",
                        original_path=str(workspace_path),
                        demo_path=str(demo_path),
                        playground_url=playground_url,
                    )
                    prepared.append(lean_demo)
                    # Save per-artifact result for incremental resume
                    with open(demo_dir / "demo_result.json", "w", encoding="utf-8") as f:
                        json.dump(lean_demo.model_dump(), f, indent=2)

                elif demo_file.endswith(".md"):
                    # Research artifact -> copy pre-generated markdown from workspace
                    source_md = workspace_path / demo_file
                    demo_path = demo_dir / "research_demo.md"
                    if source_md.exists():
                        shutil.copy(source_md, demo_path)
                    else:
                        emit.status_public_warning(
                            f"   Demo file not found: {demo_file}, generating from artifact"
                        )
                        md_content = create_research_markdown(
                            artifact, workspace_path=workspace_path
                        )
                        demo_path.write_text(md_content)

                    md_demo = MarkdownDemo(
                        id=aid,
                        iteration=artifact.iteration,
                        title=artifact.title or f"Research: {aid}",
                        summary=artifact.summary or "Research findings",
                        original_path=str(workspace_path),
                        demo_path=str(demo_path),
                    )
                    prepared.append(md_demo)
                    # Save per-artifact result for incremental resume
                    with open(demo_dir / "demo_result.json", "w", encoding="utf-8") as f:
                        json.dump(md_demo.model_dump(), f, indent=2)

                elif demo_file.endswith(".py"):
                    # Python script -> Jupyter notebook (queue for parallel conversion)
                    script_path = workspace_path / demo_file
                    if not script_path.exists():
                        emit.status_public_warning(f"   Demo file not found: {demo_file}")
                        continue

                    notebook_tasks.append(
                        {
                            "artifact_id": aid,
                            "artifact_type": artifact_type,
                            "artifact_name": demo_file,
                            "artifact": artifact,
                            "demo_dir": str(demo_dir),
                        }
                    )

            # Run notebook conversions in parallel with semaphore control
            if notebook_tasks:
                max_concurrent = (
                    config.gen_paper_repo.gen_art_demo.claude_agent.max_concurrent_agents
                )
                semaphore = asyncio.Semaphore(max_concurrent)
                emit.status_public_info(
                    f"Converting {len(notebook_tasks)} Python scripts to notebooks (max {max_concurrent} concurrent)..."
                )

                # Per-type 1-based counter assigned eagerly so each task knows
                # its slot in advance (we can't safely increment under
                # ``asyncio.gather`` without a lock — counters race across
                # concurrent ``convert_one`` calls). Stamping at queue-build
                # time gives stable, predictable ``gen_art_demo_<type>_<n>``
                # task names regardless of which task happens to start first.
                type_idx_of: dict[int, int] = {}
                type_counter: dict[str, int] = {}
                for i, t in enumerate(notebook_tasks):
                    atype = t["artifact_type"]
                    type_counter[atype] = type_counter.get(atype, 0) + 1
                    type_idx_of[i] = type_counter[atype]

                async def convert_one(task_data, task_seq):
                    aid = task_data["artifact_id"]
                    atype = task_data["artifact_type"]
                    name = task_data["artifact_name"]
                    art = task_data["artifact"]
                    demo_dir_str = task_data["demo_dir"]

                    result = await convert_to_notebook(
                        config=config,
                        artifact_id=aid,
                        artifact_name=name,
                        artifact_type=atype,
                        type_idx=type_idx_of[task_seq],
                        artifact=art,
                        output_dir=output_dir,
                        parent_module_id=parent_module_id,
                        repo_url=repo_url,
                    )

                    if result:
                        notebook_path, demo_expected_files = result

                        # Copy to artifact's demo directory
                        artifact_demo_dir = Path(demo_dir_str)
                        workspace_dir = notebook_path.parent

                        # Use descriptive name based on Python file
                        nb_name = name.replace(".py", "_code_demo.ipynb")
                        demo_path = artifact_demo_dir / nb_name
                        shutil.copy(notebook_path, demo_path)

                        # Also copy demo data file if it exists
                        demo_data_file = workspace_dir / "mini_demo_data.json"
                        if demo_data_file.exists():
                            shutil.copy(
                                demo_data_file,
                                artifact_demo_dir / "mini_demo_data.json",
                            )

                        code_demo = CodeDemo(
                            id=aid,
                            iteration=art.iteration,
                            title=art.title,
                            summary=art.summary,
                            original_path=str(art.workspace_path) if art.workspace_path else "",
                            demo_path=str(artifact_demo_dir),
                            notebook_path=str(demo_path),
                            out_expected_files=DemoExpectedFiles(**demo_expected_files),
                        )
                        # Save per-artifact result for incremental resume
                        with open(
                            artifact_demo_dir / "demo_result.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(code_demo.model_dump(), f, indent=2)
                        return code_demo
                    return None

                async def convert_with_semaphore(task_data, task_seq):
                    """Convert notebook with semaphore control."""
                    async with semaphore:
                        return await convert_one(task_data, task_seq)

                results = await asyncio.gather(
                    *[convert_with_semaphore(t, idx) for idx, t in enumerate(notebook_tasks)],
                    return_exceptions=True,
                )

                for r in results:
                    if isinstance(r, Exception):
                        emit.status_public_warning(f"   Notebook conversion failed: {r}")
                    elif r is not None:
                        prepared.append(r)

            # Emit completion through last notebook task's sequencer to avoid interleaving with concurrent tasks
            if notebook_tasks:
                emit.status_public_success(
                    f"gen_demos complete: {len(prepared)} artifacts prepared"
                )
            else:
                emit.status_public_success(
                    f"gen_demos complete: {len(prepared)} artifacts prepared"
                )

            # Save output (step-scoped: _3_gen_art_demo/prepared_artifacts.json)
            output_file = step_dir / "prepared_artifacts.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prepared": [p.model_dump() for p in prepared],
                        "folder_structure": "_3_gen_art_demo/iter_<N>/{artifact_id}/{src,demo}/",
                        "metadata": {
                            "generated_at": datetime.now(UTC).isoformat(),
                            "module": "gen_art_demo",
                            "llm_provider": "claude_agent",
                            "output_dir": str(step_dir),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            if notebook_tasks:
                emit.status_private_info(f"Saved to: {rel_path(output_file)}")
            else:
                emit.status_private_info(f"Saved to: {rel_path(output_file)}")

            return prepared
