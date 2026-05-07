"""Python script → Jupyter notebook demo conversion via Claude Agent.

Converts .py artifact scripts into self-contained Jupyter notebooks
with inlined JSON data and GitHub URL loading for Colab compatibility.

Used for: experiment, dataset, evaluation artifact types.
"""

from pathlib import Path

from aii_lib.run import emit

from aii_lib import (
    Agent,
    build_options,
    end_task_error,
    end_task_failure,
    end_task_success,
    end_task_timeout,
    setup_workspace,
    start_task,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.s_prompt_code import (
    get as get_notebook_sysprompt,
)
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
    CodeDemo,
)

# Import demo generation prompts
from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.u_prompt_code import (
    get_all_prompts as get_all_notebook_prompts,
)
from aii_pipeline.utils import PipelineConfig

from ..utils.naming import build_github_code_mini_demo_data_url


def github_to_colab_url(github_url: str) -> str:
    """Convert GitHub file URL to Google Colab URL.

    Example:
        github.com/user/repo/blob/main/demo_notebooks/exp/code_demo.ipynb
        -> colab.research.google.com/github/user/repo/blob/main/demo_notebooks/exp/code_demo.ipynb
    """
    if "github.com" in github_url:
        return github_url.replace("github.com", "colab.research.google.com/github")
    return github_url


async def convert_to_notebook(
    config: PipelineConfig,
    artifact_id: str,
    artifact_name: str,
    artifact_type: str,
    type_idx: int,
    artifact,
    output_dir: Path,
    parent_module_id: str,
    repo_url: str | None = None,
) -> tuple[Path, dict] | None:
    """Convert Python code to self-contained Jupyter notebook using Claude Agent.

    The agent reads source files from the artifact's workspace_path (in artifact_info)
    and writes output to its own workspace_dir.

    Args:
        config: Pipeline configuration
        artifact_id: Artifact identifier
        artifact_name: Name of the main script (e.g., "method.py") from demo_files
        artifact_type: Type of artifact (experiment, evaluation, dataset, etc.)
        type_idx: 1-based index for this ``artifact_type`` within the
            current ``gen_art_demo`` invocation. Drives the wire-level
            task name (``gen_art_demo_<type>_<idx>``) so the FE tree
            row reads "Dataset N" / "Experiment N" without needing a
            separate type lookup.
        artifact: Artifact object (BaseArtifact) with workspace_path for source files
        output_dir: Output directory
        repo_url: GitHub repo URL for constructing raw data URLs (for Colab compatibility)

    Returns:
        Path to the created notebook, or None on failure.
    """
    agent_cfg = config.gen_paper_repo.gen_art_demo.claude_agent
    # Wire-level task name reads ``gen_art_demo_<artifact_type>_<idx>``
    # so the FE tree row renders as "Dataset 1" / "Experiment 1" /
    # "Proof 2" — the type is encoded in the slug itself, no separate
    # channel needed. ``type_idx`` is 1-based, scoped per
    # ``gen_art_demo`` invocation by the caller in ``_3_gen_art_demo.py``.
    task_name = f"gen_art_demo_{artifact_type}_{type_idx}"

    # Per-iter workspace so artifacts that reuse the same id across iters
    # (e.g. gen_art_experiment_1 produced in iter_1 + iter_2 + iter_3) get
    # isolated CWDs.
    workspace_dir = (
        output_dir
        / "_3_gen_art_demo"
        / "notebook_workspaces"
        / f"iter_{artifact.iteration}"
        / artifact_id
    )
    setup_workspace(workspace_dir)
    task_id = start_task(task_name, parent_module_id)

    try:
        options = build_options(
            agent_cfg,
            workspace_dir,
            task_id=task_id,
            task_name=task_name,
            system_prompt=get_notebook_sysprompt(),
            output_format=CodeDemo.to_struct_output(),
            expected_files_field="out_expected_files",
            verify_retries=4,
        )

        # Get expected output files for this artifact type to include in prompt
        from .._3_gen_art_demo import get_expected_out_files_for_type

        available_files = get_expected_out_files_for_type(artifact_type)

        # Compute folder name — must match deploy_gh's destination so the
        # URL baked into the notebook resolves after publish. Stable
        # ``iter_<N>/<aid>/`` shape (no title-derived slug; the title can
        # change between iters and that was the source of #3 in the
        # errors doc — the demo URL race).
        folder_name = f"iter_{artifact.iteration}/{artifact_id}"
        github_code_mini_demo_data_url = (
            build_github_code_mini_demo_data_url(repo_url, folder_name) if repo_url else None
        )

        # Generate prompts — agent reads source files from artifact's workspace_path (in artifact_info), writes to workspace_dir
        demos_cfg = config.gen_paper_repo.gen_art_demo
        prompts = get_all_notebook_prompts(
            artifact_name=artifact_name,
            artifact=artifact,
            available_files=available_files,
            repo_url=repo_url,
            github_code_mini_demo_data_url=github_code_mini_demo_data_url,
            workspace_path=str(workspace_dir),
            max_notebook_total_runtime=demos_cfg.max_notebook_total_runtime,
        )

        # Run agent — route via exec_mode_router for RunPod support
        if config.execute_env.mode == "runpod":
            from aii_pipeline.steps._3_invention_loop.executors.exec_mode_router import (
                create_and_run_agent_simple,
            )

            agent, result = await create_and_run_agent_simple(
                options=options,
                prompts=prompts,
                config=config,
                compute_profile=agent_cfg.runpod_compute_profile,
                pod_timeout=agent_cfg.pod_timeout,
                pod_start_retries=agent_cfg.pod_start_retries,
            )
        else:
            agent = Agent(options)
            result = await agent.run(prompts)

        if result.failed:
            err = result.error_message or "unknown error"
            emit.status_public_error(f"Demo agent failed: {err}")
            end_task_failure(task_id, task_name, f"Agent failed: {err}")
            return None

        # Check if notebook was created
        notebook_path = workspace_dir / "code_demo.ipynb"
        if not notebook_path.exists():
            # Check for alternative names
            for nb_file in workspace_dir.glob("*.ipynb"):
                notebook_path = nb_file
                break

        # Extract expected_files from agent structured output
        demo_expected_files = {}
        if result.structured_output and isinstance(result.structured_output, dict):
            demo_expected_files = result.structured_output.get("out_expected_files", {})

        if notebook_path and notebook_path.exists():
            # Verify notebook contains the EXACT GitHub URL for Colab compatibility
            github_url_ok = False

            if github_code_mini_demo_data_url:
                try:
                    nb_content = notebook_path.read_text()
                    if github_code_mini_demo_data_url in nb_content:
                        github_url_ok = True
                        emit.status_private_info(
                            f"GitHub URL verified: {github_code_mini_demo_data_url}"
                        )
                    else:
                        # Try local text fix: replace any raw.githubusercontent URL
                        # pointing to mini_demo_data.json with the correct one
                        import re as _re

                        fixed_content, n_subs = _re.subn(
                            r'https://raw\.githubusercontent\.com/[^\s"\'\\]+/mini_demo_data\.json',
                            github_code_mini_demo_data_url,
                            nb_content,
                        )
                        if n_subs > 0:
                            notebook_path.write_text(fixed_content)
                            github_url_ok = True
                            emit.status_public_success(
                                f"GitHub URL fixed via local replace ({n_subs} substitutions)"
                            )
                        else:
                            emit.status_public_warning(
                                f"Notebook missing GitHub URL (no raw.githubusercontent pattern found to fix): {github_code_mini_demo_data_url}"
                            )

                except Exception as e:
                    emit.status_public_warning(f"Could not verify GitHub URL: {e}")

            status_msg = "Notebook created"
            if repo_url and github_url_ok:
                status_msg += " [GitHub URL ✓]"
            elif repo_url:
                status_msg += " [GitHub URL MISSING]"

            # Mirror raw structured_output to ``task.output`` so replay-
            # execute synthesis can recover it on a subsequent fork (see
            # task_output_replay_pattern.md / gen_paper_text). The
            # downstream notebook file is hardlinked via prep_fork on
            # replay, so the substep can re-read it; the replayed
            # ``response.structured_output`` lets the same
            # ``out_expected_files`` extraction succeed. Coerce through
            # ``CodeDemo(**dict)`` so the discriminator default
            # populates before assignment to ``task.output: AnyOutput``
            # — pydantic's tagged-union dispatch runs before field
            # defaults.
            if result.structured_output:
                parsed_output = CodeDemo.model_validate(result.structured_output)
                emit.task_output(task_id=task_id, output=parsed_output)
            emit.status_public_success(status_msg)
            end_task_success(task_id, task_name)
            return notebook_path, demo_expected_files

        emit.status_public_error("Notebook not created")
        end_task_failure(task_id, task_name, "No output")
        raise RuntimeError("Demo notebook was not created by the agent")

    except TimeoutError:
        emit.status_public_error(f"Timeout after {agent_cfg.agent_timeout}s")
        end_task_timeout(task_id, task_name, agent_cfg.agent_timeout)
        raise

    except Exception as e:
        emit.status_public_error(f"Exception: {e}")
        end_task_error(task_id, task_name, str(e))
        raise
