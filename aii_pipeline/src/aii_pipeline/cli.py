#!/usr/bin/env python3
"""
CLI interface for aii_pipeline.

Handles command-line argument parsing, configuration overrides, and entry point setup.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Unset CLAUDECODE early — prevents "nested session" error when pipeline
# spawns Claude Code agents via the SDK (SDK inherits os.environ as-is).
os.environ.pop("CLAUDECODE", None)

from aii_lib.dbos_app import init_dbos, shutdown_dbos
from aii_lib.run import current_run, emit
from dbos import SetWorkflowID
from loguru import logger

from aii_pipeline.pipeline import (
    PipelineWorkflowInput,
    cleanup_cache,
    run_end_already_emitted,
    run_pipeline,
    run_pipeline_workflow,
)
from aii_pipeline.utils import PipelineConfig, rel_path


def setup_argparser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Inventor - Pipeline Runner (internal — invoked via `python -m aii_pipeline.cli` by aii_launcher or the RunPod orchestrator entrypoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        metavar="DIR",
        action="append",
        default=[],
        help=(
            "Extra config directory layered on top of canonical "
            "aii_config/pipeline/. May be repeated; each dir mirrors the "
            "canonical layout (pipeline.yaml, io/, harness/) and any "
            "matching files deep-merge in order. Examples: per-user dir "
            "(aii_data/users/<u>/config/), experiment overlay "
            "(aii_eval/astabench/aii_config/pipeline/)."
        ),
    )

    # ---- Universal -------------------------------------------------------
    parser.add_argument(
        "--prompt",
        metavar="MSG",
        default="",
        help="Universal prompt: research prompt for fresh runs; user message for fork/resume target task.",
    )
    parser.add_argument(
        "--run-dir",
        metavar="PATH",
        default="",
        help="Parent dir where the run folder lands (default from pipeline.yaml's init.run_dir; typically aii_data/runs).",
    )
    parser.add_argument(
        "--uploads-from",
        metavar="DIR",
        default="",
        help="Pre-staged uploads dir to copy into <run>/user_uploads/ at startup.",
    )
    parser.add_argument(
        "--uploads-remove-source",
        action="store_true",
        help="After copying --uploads-from, delete the source dir (single-use staging).",
    )

    # ---- Resume mode (target module is the run's LAST module) -----------
    # Pick up an existing run dir whose pipeline died (or whose target
    # module finished but is still the last module). No truncation; the
    # existing clone log is appended to as the resumed pipeline continues.
    parser.add_argument(
        "--resume-run-id",
        metavar="ID",
        default="",
        help="Identity of the existing run to resume (run_dir = <run-dir>/<resume-run-id>).",
    )
    parser.add_argument(
        "--resume-moduleid",
        metavar="MODULE_ID",
        default="",
        help="Module id within the existing run dir whose children should resume with --prompt as the next user turn.",
    )
    # ---- Fork-spawn mode (server-initiated DBOS-native fork) ----------
    # The server's ``/fork`` endpoint pre-stages an ``aii_fork_overrides``
    # row + filesystem copytree, then spawns this CLI with the three
    # ``--fork-*`` flags below. The CLI inits DBOS, calls
    # ``DBOS.fork_workflow`` under a ``SetWorkflowID(fork-id)`` context,
    # and awaits the forked workflow's result. The forked workflow
    # body's fork-detection picks up the override row and applies the
    # new prompt + target module.
    parser.add_argument(
        "--fork-from-workflow",
        metavar="PARENT_WORKFLOW_ID",
        default="",
        help="Parent's DBOS workflow_id. Triggers fork-spawn mode.",
    )
    parser.add_argument(
        "--fork-start-step",
        metavar="N",
        type=int,
        default=0,
        help="Function id (step index) in parent's journal where the fork resumes execution.",
    )
    parser.add_argument(
        "--fork-id",
        metavar="ID",
        default="",
        help="Pre-picked DBOS workflow_id for the new fork (also the fork's run_id + run_dir name).",
    )
    parser.add_argument(
        "--execute-mode",
        choices=("local", "runpod"),
        default=None,
        help="Override execute_env.mode. Threaded down by aii_launcher (--local → 'local', --runpod → 'runpod') so the pipeline always matches the launcher's deployment intent. Falls back to ExecuteEnvConfig.mode default ('local') when omitted.",
    )
    return parser


def _tail_public_error(run: object | None) -> str:
    """Return the last ``status_public_error`` text from ``run``, or "".

    Used by the falsy-return branch of ``await run_pipeline(...)`` to
    surface the underlying reason inline in the worker log. Every
    intentional ``return None`` path in ``run_pipeline`` emits a
    ``status_public_error`` first, so reading the last one off the run
    is the cleanest way to expose what failed without changing the
    function's return contract.

    Best-effort: returns "" on any error (the caller falls back to a
    generic "no public-error event emitted" message). Walks the run's
    ``messages`` list back-to-front looking for the first
    ``status_public_error`` event.
    """
    if run is None:
        return ""
    try:
        msgs = getattr(run, "messages", None) or []
        for msg in reversed(msgs):
            if getattr(msg, "type", "") == "status_public_error":
                return str(getattr(msg, "text", "") or "").strip()
    except Exception:
        pass
    return ""


async def _run_fork_spawn(*, parent_workflow_id: str, start_step: int, fork_id: str) -> int:
    """Server-spawned fork: enqueue + await the forked workflow.

    The server has already pre-staged the ``aii_fork_overrides`` row and
    copied the parent's run_dir into the fork's run_dir. This cli boot
    path inits DBOS, calls ``DBOS.fork_workflow`` under a
    ``SetWorkflowID(fork_id)`` so the forked workflow_uuid equals the
    pre-picked fork id (the same id the FE already received from the
    server), and awaits the forked workflow's result. The fork executes
    in this process's DBOS executor — the subprocess is the workflow
    runner.
    """
    from dbos import DBOS

    init_dbos()
    try:
        with SetWorkflowID(fork_id):
            handle = await DBOS.fork_workflow_async(parent_workflow_id, start_step=start_step)
        try:
            await handle.get_result()
        except Exception as e:
            logger.exception(f"💥 Forked workflow {fork_id} failed: {e}")
            return 1
        logger.success(f"🎉 Forked workflow {fork_id} completed")
        return 0
    finally:
        shutdown_dbos()


async def main() -> int:
    """Main CLI entry point with configuration loading and pipeline execution."""
    # Pre-pipeline boot: ``logger.*`` already routes through loguru; the
    # pipeline.py boot path sets up the live Run + Run-side sinks once
    # config is loaded. Nothing to wire here.

    # Load .env file from project root (does NOT override existing env vars)
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(env_path)

    # Apply Claude SDK telemetry transport env vars BEFORE any
    # ClaudeSDKClient subprocess can spawn — the SDK reads
    # ``OTEL_EXPORTER_OTLP_*`` at subprocess-launch time and merges them
    # into the CLI child's env. Per-call on/off lives on
    # ``AgentOptions.telemetry`` (default true via
    # ``agent_backend.claude_agent_sdk.defaults`` in aii_config/pipeline/harness/<name>.yaml).
    from aii_lib.agent_backend.claude_agent_sdk.sdk_telemetry import configure_sdk_telemetry

    configure_sdk_telemetry()

    # Parse command-line arguments — strict (no unknowns).
    parser = setup_argparser()
    args = parser.parse_args()

    # ---- Fork-spawn mode -------------------------------------------------
    # When ``--fork-from-workflow`` is set, the cli's whole job is to call
    # ``DBOS.fork_workflow`` under a ``SetWorkflowID`` context and await
    # the forked workflow's result. No fresh-run / resume bootstrap. The
    # server's ``/fork`` endpoint has already pre-staged the override row
    # + filesystem state; this subprocess just executes the fork.
    if args.fork_from_workflow:
        if not args.fork_id:
            logger.error("--fork-from-workflow requires --fork-id")
            return 1
        if not args.fork_start_step or args.fork_start_step <= 0:
            logger.error("--fork-from-workflow requires --fork-start-step > 0")
            return 1
        return await _run_fork_spawn(
            parent_workflow_id=args.fork_from_workflow,
            start_step=args.fork_start_step,
            fork_id=args.fork_id,
        )

    # ---- Resume mode validation ----------------------------------------
    # Resume picks up an existing run_dir (target module flipped to
    # IN_PROGRESS, clone log appended). Forks no longer go through cli's
    # legacy clone-log path — the server's ``/fork`` endpoint calls
    # ``DBOS.fork_workflow`` via the fork-spawn mode above.
    if args.resume_moduleid and not args.resume_run_id:
        logger.error("--resume-moduleid requires --resume-run-id")
        return 1

    # Load configuration from YAML.
    # --config-dir is repeatable: each dir layers on top of the canonical
    # aii_config/pipeline/. Missing dirs are skipped with a warning.
    extra_config_dirs: list[Path] = []
    for d in args.config_dir or []:
        p = Path(d)
        if not p.exists():
            logger.warning(f"Config dir not found, skipping: {p}")
            continue
        extra_config_dirs.append(p)

    try:
        config = PipelineConfig.from_yaml(*extra_config_dirs)
        logger.info(f"📋 Config loaded (canonical + {len(extra_config_dirs)} extra layer(s))")

        # Initialize aii_lib global config
        from aii_lib.config import aii_config

        aii_config.init_from_pipeline_config(config)
    except FileNotFoundError as e:
        logger.error(f"❌ Config file not found: {e}")
        return 1
    except Exception as e:
        logger.error("❌ Error loading config", exc=e)
        return 1

    # No proxy auto-start needed. The LiteLLM proxy was the bridge for
    # claude_agent_sdk + openrouter, which ``PipelineConfig._validate_backend_pairings``
    # now rejects at load time (translation gaps made the bridge unreliable
    # in practice). The openrouter llm_backend's only consumption path
    # today is the direct ``OpenRouterClient.chat`` flow used by
    # ``_run_task_openrouter`` and friends — that talks to OpenRouter's
    # native API directly, no local proxy involved. When/if a non-Claude
    # agent_backend gains SDK support, this is the seam where its
    # transport bootstrap would land.

    # ---- Apply CLI args onto config --------------------------------------
    if args.execute_mode:
        # Threaded down from aii_launcher (--local / --runpod) so the
        # pipeline's exec env always matches the launcher's deployment
        # intent, no matter what yaml shipped in the config tarball.
        config.execute_env.mode = args.execute_mode
    if args.run_dir:
        config.init.run_dir = args.run_dir
    if args.uploads_from:
        config.init.user_uploads_copy_from = args.uploads_from
    if args.uploads_remove_source:
        config.init.user_uploads_remove_source = True
    # ``--prompt`` is mode-dependent: for fresh runs it's the AII research
    # prompt; for resume it's the user message that fans out across
    # the target module's session-bearing children (threaded through the
    # handoff below). We only land it on ``config.prompt`` for fresh runs
    # — resume paths consume it directly via ``args.prompt``.
    is_fresh_run = not args.resume_moduleid
    if is_fresh_run and args.prompt:
        config.prompt = args.prompt

    # ---- RunPod mode: ensure templates exist + capture their IDs --------
    # ``deploy_and_run`` does this on the host, but the IDs live only in
    # the in-memory ``runpod_cfg`` there — the tarball shipped to the
    # orchestrator pod is built from raw on-disk YAML, which has empty
    # ``template_ids.worker_*``. Without this call ``RunPodAgentDispatcher``
    # forwards ``template_id=""`` to ``aii_runpod__gen_pod``, which then
    # falls back to the ``aii-orchestrator`` template by name for every
    # worker pod (wrong template, wrong start command).
    if config.execute_env.mode == "runpod":
        from aii_runpod.pod_infra import ensure_all_templates

        from aii_pipeline.utils.config_models.infra import TemplateIdsConfig
        from aii_runpod import RunPodClient

        runpod_cfg = config.execute_env.runpod
        client = RunPodClient(
            data_center_id=runpod_cfg.data_center_id,
            cloud_type=runpod_cfg.cloud_type,
        )
        ids = await ensure_all_templates(
            client,
            getattr(runpod_cfg, "templates", {}) or {},
            runpod_cfg.docker_image,
            volume_mount_path=getattr(runpod_cfg, "volume_mount_path", None),
        )
        runpod_cfg.template_ids = TemplateIdsConfig(
            aii_server=ids.ids.get("aii_server", ""),
            orchestrator=ids.ids.get("orchestrator", ""),
            worker_gpu=ids.ids.get("worker_gpu", ""),
            worker_cpu_heavy=ids.ids.get("worker_cpu_heavy", ""),
            worker_cpu_light=ids.ids.get("worker_cpu_light", ""),
        )
        logger.info(
            f"Worker templates ready: gpu={runpod_cfg.template_ids.worker_gpu}, "
            f"cpu_heavy={runpod_cfg.template_ids.worker_cpu_heavy}, "
            f"cpu_light={runpod_cfg.template_ids.worker_cpu_light}"
        )

    # ---- Local mode: run pipeline directly ----
    # RunPod mode is owned by `aii_launcher --runpod` (which boots the local
    # aii_server first so RunPodClient can reach it). This entry point
    # is internal — it's invoked via `python -m aii_pipeline.cli` by the
    # launcher and by the orchestrator pod's entrypoint, both of which
    # only ever exercise the local-execution path here.
    # Resolve outputs directory based on execution mode
    output_base = config.outputs_directory
    # Resolve relative path against project root
    if not Path(output_base).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent  # repo root
        output_base = str(project_root / output_base)

    # ── Widen typed unions before any seed-driven model_validate ──
    # Each phase / substep subclass declares its own ``kind`` literal.
    # The pipeline-side discriminated union has to know about all of
    # them so ``CloneSink.load``'s ``fork_init`` seed branch picks
    # typed classes back out of ``model_validate`` instead of
    # collapsing to base. Idempotent — first call rebinds + rebuilds,
    # subsequent calls early-return.
    from aii_pipeline.run.typed_union import bind_pipeline_typed_unions

    bind_pipeline_typed_unions()

    # ── Pick the run id + run_dir + prepared Run based on cli flags ──
    # Fresh runs delegate to ``run_pipeline_workflow`` (the canonical
    # ``@DBOS.workflow`` entry), so we only pre-pick the id + dir here
    # and let the workflow body build ``Run.fresh`` itself.
    # Resume still constructs ``prepared_run`` here and calls
    # ``run_pipeline`` directly until that mode migrates too.
    from aii_pipeline.run import Run as _Run

    via_dbos = False
    prepared_run: _Run | None = None

    if args.resume_moduleid:
        # Resume in place — same run_dir as the existing run. Walks
        # DBOS's journal for the workflow id and dispatches each event
        # against a fresh Run; dispatch's typed-class resolvers (wired
        # in ``aii_pipeline.run.__init__``) construct typed phase /
        # module subclasses directly. No scaffold step.
        run_dir = Path(f"{output_base}/{args.resume_run_id}")
        run_id = run_dir.name
        try:
            prepared_run = _Run.from_journal(
                run_id,
                target_module_id=args.resume_moduleid,
                prompt=args.prompt or "",
            )
        except (ValueError, TypeError) as e:
            logger.error(f"resume: {e}")
            return 1
        logger.info(f"Resume mode: reusing existing run {run_id}")
    else:
        # Fresh run via DBOS workflow. Pre-pick the id so SetWorkflowID
        # can stamp it onto the workflow row before the body runs;
        # ``Run.fresh`` inside the workflow uses the same id.
        run_id = _Run.gen_id()
        run_dir = Path(output_base) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        via_dbos = True
        logger.info(f"Generated new run id: {run_id}")

    # Rename our tmux session so it matches the canonical
    # ``aii-<run_id>`` shape. Fresh-run starts from the dashboard
    # launch with an ``aii-pending-<uuid>`` placeholder because the
    # dashboard doesn't know the run id upfront (we just generated it).
    # /api/runs/{id}/stop relies on the canonical name to find + kill
    # the session, so this rename is the linchpin for graceful stops.
    # No-op when not in tmux (e.g. direct CLI launch from a terminal).
    from aii_lib.utils.tmux import (
        current_session_name as _cur_sess,
    )
    from aii_lib.utils.tmux import (
        pipeline_session_name as _sess_name,
    )
    from aii_lib.utils.tmux import (
        rename_session as _rename_sess,
    )

    _old_sess = _cur_sess()
    if _old_sess:
        _new_sess = _sess_name(run_id)
        if _old_sess != _new_sess:
            if _rename_sess(_old_sess, _new_sess):
                logger.info(f"Renamed tmux session: {_old_sess} → {_new_sess}")
            else:
                logger.warning(f"Failed to rename tmux session {_old_sess} → {_new_sess}")

    # Set up run-specific HuggingFace cache directory
    hf_cache_dir = run_dir / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")

    logger.info(f"🗂️ Run-specific HuggingFace cache in {rel_path(run_dir)}")
    logger.info(f"📁 Created run directory: {rel_path(run_dir)}")

    # Save resolved config (after CLI overrides) to run dir for reproducibility
    import yaml as _yaml

    config_snapshot_dir = run_dir / "config"
    config_snapshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        # pipeline.yaml — the fully resolved config
        with open(config_snapshot_dir / "pipeline.yaml", "w") as f:
            _yaml.dump(config.raw, f, default_flow_style=False, sort_keys=False)
        # Also save any other config files that were loaded. Snapshot the
        # canonical aii_config/pipeline/ tree as the audit reference; per-user
        # / experiment overlays already merged into ``config.raw`` above.
        source_dir = Path(__file__).parent.parent.parent.parent / "aii_config" / "pipeline"
        import shutil as _shutil

        for subdir in ("harness", "io"):
            src = source_dir / subdir
            if src.is_dir():
                _shutil.copytree(
                    src,
                    config_snapshot_dir / subdir,
                    dirs_exist_ok=True,
                )
    except Exception as e:
        # Config snapshot is a best-effort audit artifact — don't fail the run
        # if it can't be written (e.g. FS permissions), but log so the gap is
        # visible when investigating later.
        logger.warning(f"Config snapshot skipped: {e}")

    # Run the pipeline
    exit_code = 1
    try:
        if via_dbos:
            # Fresh runs go through the canonical ``@DBOS.workflow``
            # entry. ``init_dbos`` launches the DBOS runtime so the
            # journal mirror, agent step, and child workflows engage;
            # ``SetWorkflowID`` stamps our pre-picked ``run_id`` onto
            # the new workflow_status row so the run id and the DBOS
            # workflow_uuid stay equal.
            init_dbos()
            wf_input = PipelineWorkflowInput(
                mode="fresh",
                run_id=run_id,
                run_dir=str(run_dir),
                output_base=str(output_base),
                prompt=args.prompt or "",
                extra_config_dirs=[str(p) for p in extra_config_dirs],
            )
            try:
                with SetWorkflowID(run_id):
                    result_dict = await run_pipeline_workflow(wf_input)
                result = result_dict.get("status") == "completed"
            finally:
                shutdown_dbos()
        else:
            result = await run_pipeline(
                config,
                run_dir=run_dir,
                prepared_run=prepared_run,
            )

        if result:
            logger.success("🎉 Pipeline completed successfully!")
            exit_code = 0
        else:
            # ``run_pipeline`` returned None / falsy. Every legitimate
            # validation-failure path emits a ``status_public_error``
            # before returning, so dump the most recent one here so the
            # worker log has the actual reason inline (without forcing
            # an operator to grep slim_message.jsonl by hand). The
            # ``current_run`` lookup may miss for the DBOS path if the
            # workflow never set it (early failure) — guard with try.
            try:
                _live_run = current_run()
            except LookupError:
                _live_run = None
            tail = _tail_public_error(_live_run) if _live_run is not None else ""
            if tail:
                logger.error(f"💥 Pipeline failed: {tail}")
            else:
                logger.error("💥 Pipeline failed (no public-error event emitted)")
            exit_code = 1
            # Emit run_end on failure (the inner pipeline didn't reach its own emit)
            if not run_end_already_emitted() and _live_run is not None:
                emit.end_run(run_id=_live_run.node_id, status="failed")
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("⚠️  Pipeline interrupted by user")
        exit_code = 130
        # Emit run_end on interrupt — only if the inner pipeline didn't already
        if not run_end_already_emitted():
            try:
                emit.end_run(run_id=current_run().node_id, status="interrupted")
            except Exception:
                logger.opt(exception=True).debug(
                    "cli: emit.end_run on interrupt failed (no active Run?)"
                )
    except Exception as e:
        # ``logger.exception`` includes the full traceback under loguru —
        # critical for fork/resume crashes where ``e``'s message alone
        # ("KeyError: 'foo'") can't be located in pipeline.py without
        # the stack.
        logger.exception(f"💥 Pipeline crashed: {e}")
        exit_code = 1
        # Emit run_end on crash — only if the inner pipeline didn't already.
        if not run_end_already_emitted():
            try:
                emit.end_run(
                    run_id=current_run().node_id,
                    status="crashed",
                    text=f"Pipeline crashed: {e}",
                )
            except Exception:
                logger.opt(exception=True).debug(
                    "cli: emit.end_run on crash failed (no active Run?)"
                )
    finally:
        # Always clean up all caches at the end
        cleanup_cache(run_dir)

    return exit_code


_cleanup_done = False


def _kill_child_processes():
    """Kill all child processes (claude agents, subprocesses). Safe to call multiple times."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    try:
        import psutil

        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        if children:
            logger.info(f"Killing {len(children)} child processes...")
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # Wait briefly, then force-kill survivors
            _, alive = psutil.wait_procs(children, timeout=3)
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        logger.warning(f"Child process cleanup failed: {e}")


def _shutdown(exit_code: int):
    """Clean shutdown: kill child processes and exit."""
    _kill_child_processes()
    os._exit(exit_code)


def cli_main():
    """Synchronous entry point for console script.

    Just runs the pipeline. No tmux, no orchestration — aii_launcher handles that.
    """
    import atexit
    import signal

    # Register cleanup for ALL exit paths (normal, signal, atexit)
    atexit.register(_kill_child_processes)
    signal.signal(signal.SIGTERM, lambda *_: _shutdown(143))
    signal.signal(signal.SIGHUP, lambda *_: _shutdown(129))

    exit_code = 0
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        _shutdown(130)
    _kill_child_processes()
    sys.exit(exit_code)


if __name__ == "__main__":
    cli_main()
