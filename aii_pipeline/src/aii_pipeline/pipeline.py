#!/usr/bin/env python3
"""
AI Inventor - Pipeline Orchestration.

Core pipeline orchestration logic for executing research modules in sequence.

Boot flow (fresh / resume / fork all share this skeleton — fork enters
through DBOS's ``fork_workflow`` step-replay rather than a separate
constructor):

  1. cli.py picks a constructor (``Run.fresh`` for new runs,
     ``Run.from_resume`` for the legacy ``--resume-run-id`` cold path)
     and hands the prepared Run + run_dir to :func:`run_pipeline`.
     DBOS-native fork uses ``DBOS.fork_workflow`` directly without
     constructing a new ``Run`` aggregate.
  2. ``set_current_run`` + summary buffer + scaffold.
  3. :func:`wire_all_sinks` subscribes every output sink (Run-bus
     producers + JournalTailer consumers).
  4. Resume / fork: when the workflow has an ``aii_fork_overrides`` row
     OR ``run._pending_resume_target`` is set, the relevant phase /
     module substep picks up the override-prompt + session_ids when
     dispatching the resume target.
  5. Forward pipeline: phases run from ``first_step`` to ``last_step``;
     DBOS journal step-replay handles idempotent re-execution.
  6. ``run.end()`` + close sinks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from aii_lib.llm_backend.claude_max.autologin import ensure_oauth_token_fresh
from aii_lib.run import emit
from dbos import DBOS
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aii_lib.run.run import Run

from aii_lib import cleanup_run_caches
from aii_pipeline.utils import (
    DEFAULT_MIN_TOKEN_VALIDITY_SECONDS,
    PipelineConfig,
    rel_path,
)


class PipelineFailure(Exception):
    """Validation / structural failure inside :func:`run_pipeline`.

    Raised from paths that historically returned ``None`` silently — the
    public-status emitter is replay-skipped while ``playback_mode ==
    "replay"`` (the boot phase for fork / resume), so a bare emit drops
    on disk and the cli sees only "Pipeline failed!" with no surfaced
    reason. Raising instead routes the message through cli.py's
    ``except Exception`` handler (with full traceback via
    ``logger.exception``).
    """


# Module-level flag set by run_pipeline once it has emitted run_end. cli.py's
# except handlers check this before emitting their own run_end so a successful
# run that crashes during post-success cleanup doesn't get re-tagged "crashed".
_run_end_emitted: bool = False


def run_end_already_emitted() -> bool:
    """Check if run_end has already been emitted."""
    return _run_end_emitted


def cleanup_cache(run_dir: Path | str) -> None:
    """Clean up all cache directories created during this run."""
    result = cleanup_run_caches(run_dir, clear_venv=True, clear_hf=True)
    if result["removed"]:
        logger.info(f"🧹 Cleaning up caches: {len(result['removed'])} items")
        for item in result["removed"]:
            logger.debug(f"  - {item}")
        logger.success(f"✅ Removed {result['total_size_mb']:.1f} MB of cached data")


# ---------------------------------------------------------------------------
# Sink / source wiring helpers
# ---------------------------------------------------------------------------


def wire_all_sinks(run: Run, run_dir: Path, config: PipelineConfig) -> dict:
    """Subscribe every output sink config-says-enabled.

    Returns a dict of the constructed sinks keyed by short name; the
    pipeline holds onto these references for shutdown
    (:func:`close_all_sinks`). All-or-nothing — if a sink is disabled
    by config its slot stays absent.

    Single fan-out: every sink (CloneSink, SequencedCloneSink,
    TitleGeneratorSink, ConsoleRunSink, HealthSink, OTelRunSink)
    subscribes to one :class:`JournalTailer` keyed on
    ``run.node_id``. The tailer polls ``dbos.operation_outputs`` and
    dispatches each decoded :class:`BaseMessage` to every subscriber.

    Two write paths feed the same journal:

      * Direct ``emit.X`` calls (status / agent / *output) →
        ``journal_event_step`` directly.
      * Run-bus ``start_*`` / ``end_*`` lifecycle emits →
        :meth:`Run._record` mirrors via ``journal_event_step`` after
        dispatch (replay-skipped so legacy resume doesn't
        double-write).

    Subscription order doesn't matter: each sink polls the same
    cursor; a subscriber added later just starts at the next poll
    boundary.
    """
    sinks_cfg = config.sinks
    sinks: dict = {}

    # ── Journal tailer (single fan-out for every consumer below) ──
    from aii_lib.run.journal import JournalTailer

    tailer = JournalTailer(workflow_id=run.node_id)

    # ── Clone log + sequenced clone (journal consumers) ──
    from aii_lib.run.sinks.clone import (
        CLONE_RELATIVE,
        SEQUENCED_CLONE_RELATIVE,
        CloneSink,
        SequencedCloneSink,
    )

    sinks["clone"] = CloneSink(run_dir / CLONE_RELATIVE)
    tailer.subscribe(sinks["clone"])
    emit.status_private_info(
        f"CloneSink: writing to {Path(CLONE_RELATIVE).name}",
    )
    sinks["sequenced_clone"] = SequencedCloneSink(
        run_dir / SEQUENCED_CLONE_RELATIVE,
        sequence_lookup=run.task_sequence,
    )
    tailer.subscribe(sinks["sequenced_clone"])
    emit.status_private_info(
        f"SequencedCloneSink: writing to {Path(SEQUENCED_CLONE_RELATIVE).name}",
    )

    # ── Run-name generator (journal consumer; mutates run.name on
    #    RunTitleMessage and emits a fresh title via run._on) ──
    from aii_lib.run.sinks.title import TitleGeneratorSink

    sinks["title"] = TitleGeneratorSink(run, run_dir)
    tailer.subscribe(sinks["title"])

    # ── Console / health / OTel (journal consumers) ──
    from aii_lib.run.sinks.console import ConsoleRunSink

    sinks["console"] = ConsoleRunSink(
        truncation=sinks_cfg.console.msg_truncate,
        log_llm_messages=sinks_cfg.console.log_llm_messages,
        include_private_messages=sinks_cfg.console.include_private_messages,
        sequence_lookup=run.task_sequence,
    )
    tailer.subscribe(sinks["console"])
    emit.status_private_info(
        f"ConsoleRunSink started "
        f"(truncation={sinks['console'].truncation}, "
        f"log_llm={sinks['console'].log_llm_messages}, "
        f"include_private={sinks['console'].include_private_messages})",
    )

    from aii_lib.run.sinks.health import HealthSink

    sinks["health"] = HealthSink(
        run_dir,
        heartbeat_seconds=sinks_cfg.health.heartbeat_seconds,
    )
    tailer.subscribe(sinks["health"])

    # ── OpenTelemetry traces + metrics (opt-in) ──
    otel_cfg = sinks_cfg.otel
    if otel_cfg.enabled:
        from aii_lib.run.sinks.otel import OTelRunSink

        if OTelRunSink is not None:
            sinks["otel"] = OTelRunSink(
                run=run,
                traces_path=run_dir / otel_cfg.traces_file,
                metrics_path=run_dir / otel_cfg.metrics_file,
                metrics_interval_ms=otel_cfg.metrics_interval_ms,
                trace_export_interval_ms=otel_cfg.trace_export_interval_ms,
                sample_rate=otel_cfg.sample_rate,
                otlp_endpoint=otel_cfg.otlp_endpoint,
                otlp_insecure=otel_cfg.otlp_insecure,
                otlp_headers=(
                    {"Authorization": os.environ["GRAFANA_OTLP_AUTH"]}
                    if os.environ.get("GRAFANA_OTLP_AUTH")
                    else otel_cfg.otlp_headers
                ),
                resource_attrs={
                    "aii.run_dir": str(run_dir),
                    "deployment.environment": config.execute_env.mode,
                },
            )
            tailer.subscribe(sinks["otel"])

    # ── Start the tailer last (drives the consumer fan-out) ──
    # Keyed under ``_tailer`` so :func:`close_all_sinks` (reverse-LIFO)
    # tears it down FIRST — the synchronous drain pass inside
    # ``stop(drain=True)`` then dispatches the tail of the journal to
    # every subscribed sink before they close.
    tailer.start()
    sinks["_tailer"] = tailer

    return sinks


def close_all_sinks(sinks: dict) -> None:
    """Best-effort close every sink in reverse construction order.

    Reverse-LIFO closes the :class:`JournalTailer` first (added last in
    :func:`wire_all_sinks`), so its synchronous drain pass can still
    dispatch to console / health / OTel before they tear down. Standard
    resource-cleanup pattern: dispose in reverse construction order.
    """
    for name, sink in reversed(list(sinks.items())):
        if sink is None:
            continue
        try:
            sink.close()
        except Exception as e:
            logger.warning(f"sink close failed for {name}: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    config: PipelineConfig,
    run_dir: Path | None = None,
    prepared_run: Run | None = None,
) -> Run | None:
    """Run the pipeline.

    Args:
        config: PipelineConfig.
        run_dir: Run directory.
        prepared_run: ``Run`` aggregate from ``Run.fresh`` (new run) or
            ``Run.from_resume`` (legacy ``--resume-run-id`` cold path).
            Always required — cli.py picks a constructor and hands it
            through. DBOS-native fork builds its prepared_run via
            ``Run.fresh`` and carries the parent's session_ids via
            ``aii_fork_overrides``.

            For resume mode ``prepared_run._pending_resume_target`` is
            set to the target module's node_id. With
            ``prepared_run.prompt`` non-empty: this triggers the resume
            turn (FORK + prompt). With empty prompt: truncate-and-restart.
    """
    global _run_end_emitted
    if run_dir is None:
        raise ValueError("run_dir is required")
    if prepared_run is None:
        raise ValueError("prepared_run is required (use Run.fresh or Run.from_resume)")

    # Auto-create user_uploads/ in every run directory
    user_uploads_dir = run_dir / "user_uploads"
    user_uploads_dir.mkdir(exist_ok=True)

    # Copy pre-staged uploads into user_uploads/
    copy_from = config.init.user_uploads_copy_from
    if copy_from:
        import shutil

        src = Path(copy_from)
        if src.is_dir():
            shutil.copytree(src, user_uploads_dir, copy_function=shutil.copy2, dirs_exist_ok=True)
            logger.info(f"Copied user_uploads from {src}")
            if config.init.user_uploads_remove_source:
                shutil.rmtree(src, ignore_errors=True)
                logger.info(f"Removed user_uploads source: {src}")

    pipeline_cfg = config.init.pipeline
    summary_cfg = config.init.llm_gen_summary

    # ---- Live Run aggregate ----
    from aii_lib.run import set_current_run

    run = prepared_run
    set_current_run(run)

    # ---- Pre-record LLM-summary buffer (opt-in) ----
    if summary_cfg.enabled:
        run.enable_summary_buffer(
            min_chars=summary_cfg.min_chars,
            max_chars=summary_cfg.max_chars,
            max_concurrent=summary_cfg.max_concurrent,
        )

    # ---- v26 scaffold: pre-populate groups + iterations + modules ----
    from aii_pipeline.run.scaffold import scaffold_pipeline

    # Pre-populate phase MdGroups + iterations + modules. Idempotent
    # over re-entry. For forks (DBOS-native), the run_dir was copied
    # from parent by the server endpoint and the override row was
    # staged before the fork was enqueued — the workflow body just
    # needs an empty tree skeleton; the cached pre-fork operation
    # outputs DBOS auto-copies do the rest.
    scaffold_pipeline(run, config)

    # ---- Wire output sinks ----
    # Inbound channels (``send_message``, ``stop``) are now first-class
    # DBOS primitives — see ``run_pipeline_workflow`` for ``send_message``
    # via the recv-loop sibling, and the server's ``/stop`` endpoint
    # which calls ``DBOS.cancel_workflow_async`` directly.
    sinks = wire_all_sinks(run, run_dir, config)

    # Set ambient config so prompts/steps can read it without threading
    from aii_pipeline.utils.context import set_pipeline_config

    set_pipeline_config(config)

    # Bootstrap-source breadcrumb: resume already finished setting up
    # ``prepared_run`` before sinks existed, so its setup didn't hit the
    # bus. Emit a startup line here for visibility symmetric with
    # send_message / stop / sinks.
    if run._pending_resume_target is not None:
        emit.status_private_info(
            "ResumeSource: bootstrapped from clone log",
        )

    # RUN_START is emitted by the API endpoint (aii_server start_run) before
    # this subprocess boots, so the SSE/UI sees the event within milliseconds
    # instead of waiting for the pipeline to import. Standalone CLI
    # invocations (no API in front) won't have a run_start line —
    # acceptable for dev/debug runs.

    # Read auth config for token validity checks between steps
    _auth_cfg = (
        config.raw.get("agent_backend", {})
        .get("claude_agent_sdk", {})
        .get("llm_backend", {})
        .get("claude_max", {})
        .get("auth", {})
    )
    min_token_validity = _auth_cfg.get(
        "min_token_validity_seconds",
        DEFAULT_MIN_TOKEN_VALIDITY_SECONDS,
    )

    # Pipeline phase sequence — one entry per top-level phase, in
    # canonical order. ``run.children`` holds the typed phase MdGroups
    # in the same order (the scaffold builds them sequentially); we
    # iterate that directly and call ``await phase.execute()`` on
    # each. STEP_REGISTRY is gone — phase classes own their own
    # execution.
    from aii_lib.run.context import ctx_scope

    from aii_pipeline.steps.base import StepContext

    # ---- Resume / restart branch (replay-execute model) ----
    # When ``_pending_resume_target`` is set (legacy
    # ``--resume-run-id`` cold-path goes through ``Run.from_resume``),
    # the forward pipeline does NOT slice phases. Instead, the run
    # boots in ``_playback_mode="replay"`` and re-runs every phase's
    # ``execute()`` from ``first_step`` onward. Replay-execute
    # machinery handles the skip-ahead automatically: idempotent
    # dispatch + slot-claim reuse recorded node_ids, ``Agent.run``
    # synthesis short-circuits to the recorded response, and the mode
    # flips back to ``"live"`` at the resume target's
    # ``start_*_module`` boundary so the target's substep onward runs
    # in normal live mode.
    #
    # DBOS-native fork goes through ``DBOS.fork_workflow`` instead and
    # never enters this branch — its prepared_run is built by
    # ``Run.fresh`` with the parent's session_ids carried via
    # ``aii_fork_overrides``.
    #
    # The empty-prompt restart-from-target path still truncates the
    # target's children + re-scaffolds downstream phases.
    if run._pending_resume_target is not None:
        target = run.find_node(run._pending_resume_target)
        if target is None:
            # ``status_public_error`` is replay-skipped while ``playback_mode
            # == "replay"`` (which it still is at this point in the boot
            # path), so the emit silently drops on disk and the cli only
            # sees a bare "Pipeline failed!" — no surfaced reason. Raise
            # instead so the cli's exception handler logs the message +
            # traceback inline. Keep the emit too: it'll surface in live
            # mode (resume-from-already-live, no replay mode) where the
            # FE can show it; fork/cold-resume callers see the raise via
            # the cli log.
            msg = f"resume/fork target {run._pending_resume_target!r} not in tree"
            emit.status_public_error(msg)
            raise PipelineFailure(msg)

        if run.prompt:
            emit.status_public_progress(
                f"resume-with-prompt: module '{target.node_id}' will "
                f"FORK its session and replace the next user turn "
                f"(replay-execute walks prior phases first)",
            )
        else:
            emit.status_public_progress(
                f"restart-from-target: module '{target.node_id}' will "
                f"re-dispatch fresh after truncation "
                f"(replay-execute walks prior phases first)",
            )
            run.remove_children(target)
            # remove_children drops later phases (per its docstring);
            # re-scaffold so they're back as PENDING stubs, otherwise the
            # validation below ("Invalid last_step: gen_paper_repo") rejects
            # the run because gen_paper_repo is no longer in run.children.
            # scaffold_pipeline is idempotent (find_group_by_name guards).
            scaffold_pipeline(run, config)

    first_step = (pipeline_cfg.first_step or "seed_hypo").strip()
    last_step = (pipeline_cfg.last_step or "gen_paper_repo").strip()

    # Validate phase names against ``run.children`` (the live tree).
    # Each error path emits + raises (see resume-target validation above
    # for why the bare emit silently drops during replay mode).
    phase_names = [c.name for c in run.children]
    if first_step not in phase_names:
        msg = f"Invalid first_step: '{first_step}'. Valid: {', '.join(phase_names)}"
        emit.status_public_error(msg)
        raise PipelineFailure(msg)
    if last_step not in phase_names:
        msg = f"Invalid last_step: '{last_step}'. Valid: {', '.join(phase_names)}"
        emit.status_public_error(msg)
        raise PipelineFailure(msg)

    start_index = phase_names.index(first_step)
    end_index = phase_names.index(last_step)

    if start_index > end_index:
        msg = (
            f"Invalid configuration: first_step '{first_step}' comes after last_step '{last_step}'"
        )
        emit.status_public_error(msg)
        raise PipelineFailure(msg)

    step_ctx = StepContext(
        config=config,
        run_dir=run_dir,
    )

    # No prev_results hydration: phase results live on the run tree as
    # ``MdGroup.output`` (set by the per-phase ``mdgroup_output`` emit
    # below) — clone-log replay rebuilds those during legacy
    # ``Run.from_resume`` cold-path resume; DBOS-native fork inherits
    # them via the journal's step-output replay. Downstream phases
    # read via the run-tree accessors in their ``get_context`` methods.

    modules_to_run = phase_names[start_index : end_index + 1]
    emit.status_public_progress(
        f"Will run phases: {' -> '.join(modules_to_run)}",
    )

    # Run phases via the MdGroup ``execute`` contract. Each phase reads
    # the top-level StepContext via ``current_ctx()``, builds its own
    # narrower phase ctx in ``get_context()``, and pushes it inside
    # ``execute()``.
    #
    # /stop is now ``DBOS.cancel_workflow_async`` — the workflow body
    # gets cancelled at its next DBOS-step boundary, so there's no
    # cooperative bail-out for the loop to handle here.
    with ctx_scope(step_ctx):
        for i in range(start_index, end_index + 1):
            phase = run.children[i]
            step_name = phase.name
            emit.status_public_progress(
                f"Running phase: {step_name}",
            )

            result = await phase.execute()
            if not result:
                # ``status_public_error`` is replay-skipped during
                # the boot phase, so a phase that returns falsy
                # during replay-execute would otherwise drop on
                # disk and the cli would log only the generic
                # "Pipeline failed!" line. Raise so cli.py's
                # exception handler logs the message + traceback.
                msg = f"Phase {step_name!r} returned falsy result"
                emit.status_public_error(msg)
                raise PipelineFailure(msg)

            # Surface the phase's typed result on its MdGroup via
            # the ``mdgroup_output`` event so downstream readers and
            # fork-replay see the same state. Every phase now
            # produces a typed Pydantic model (SeedHypoOut /
            # HypoLoopOut / InventionLoopOut / GenPaperRepoOut).
            emit.mdgroup_output(
                group_id=phase.node_id,
                output=result,
            )

            # Wall-clock-dependent token refresh is gated on live
            # mode: in replay-execute the agents synth from
            # recorded state and never use the OAuth token, so a
            # check here would be wasted work (and on cold-fork
            # could trip on missing creds in the new run dir).
            if min_token_validity and i < end_index and run.playback_mode == "live":
                ensure_oauth_token_fresh(min_token_validity)

    # Drain pending LLM summaries before final status messages
    run.close_summary_buffer()

    emit.status_public_success("Pipeline completed")
    emit.status_private_info(f"Run directory: {rel_path(run_dir)}")
    gen_paper_group = run.find_group_by_name("gen_paper_repo")
    gen_paper_result = gen_paper_group.output if gen_paper_group is not None else None
    if gen_paper_result and getattr(gen_paper_result, "repo_url", None):
        emit.status_public_success(
            f"   GitHub repo:   {gen_paper_result.repo_url}",
        )

    # Emit run_end
    emit.end_run(run_id=run.node_id, status="completed")
    _run_end_emitted = True

    close_all_sinks(sinks)

    return run


# ---------------------------------------------------------------------------
# DBOS workflow entry point
# ---------------------------------------------------------------------------


class PipelineWorkflowInput(BaseModel):
    """JSON-safe input to :func:`run_pipeline_workflow`.

    Two modes share the shape:

      * ``fresh`` — brand-new run; ``Run.fresh`` constructs the Run +
        run_dir under ``output_base``.
      * ``resume`` — existing run resumed from ``target_module_id`` with
        a new ``prompt``; ``Run.from_resume`` rehydrates state from the
        on-disk clone log.

    Forking does NOT show up here — ``DBOS.fork_workflow`` clones the
    parent's input verbatim, so a forked workflow's ``wf_input`` is
    indistinguishable from the parent's. The fork is identified at the
    top of the body via the ``aii_fork_overrides`` side-table lookup
    keyed by ``DBOS.workflow_id``.

    The caller pre-picks ``run_id`` and stamps it via ``SetWorkflowID``
    so DBOS's ``workflow_uuid`` and the AII run id stay equal.
    """

    mode: Literal["fresh", "resume"]
    run_id: str
    run_dir: str = ""
    output_base: str = ""
    prompt: str = ""
    target_module_id: str | None = None
    extra_config_dirs: list[str] = Field(default_factory=list)


@DBOS.workflow()
async def run_pipeline_workflow(wf_input: PipelineWorkflowInput) -> dict:
    """Construct the Run from JSON-safe input + delegate to :func:`run_pipeline`.

    Decorated with ``@DBOS.workflow`` so every nested ``@DBOS.step``
    call (journal mirror, agent dispatch, child workflows) records
    under this workflow's ``workflow_uuid``. Caller sets the uuid via
    ``with SetWorkflowID(run_id):`` before the await so DBOS's
    workflow_id and the AII run_id stay equal.

    Fork detection: at the top of the body we read the
    ``aii_fork_overrides`` side table keyed by the **current**
    ``DBOS.workflow_id`` (not ``wf_input.run_id``, which carries the
    parent's id when ``DBOS.fork_workflow`` cloned the input). When a
    row exists, this is a forked workflow — apply the new prompt +
    target module + parent's task session ids onto the freshly
    constructed Run, then continue normally. The cached pre-fork
    operation outputs that DBOS auto-copies handle the rest.

    Background tasks (interim summary loop, send_message recv loop) are
    spawned inside the workflow body so their nested step calls inherit
    the workflow's ``ContextVar``s. They're cancelled in the finally
    block so they never outlive the workflow.

    Returns a JSON-safe summary dict so DBOS can journal it cleanly.
    """
    from aii_lib.run.fork_override import read_fork_override
    from aii_lib.run.run import Run as _Run

    config = PipelineConfig.from_yaml(*[Path(d) for d in wf_input.extra_config_dirs])

    # Use DBOS's workflow id rather than ``wf_input.run_id`` so a forked
    # workflow (which inherits the parent's input verbatim) builds its
    # own Run + run_dir under the fork's id.
    actual_run_id = DBOS.workflow_id or wf_input.run_id

    if wf_input.mode == "fresh":
        if not wf_input.output_base:
            raise PipelineFailure("fresh mode requires output_base")
        prepared_run, run_dir = _Run.fresh(
            Path(wf_input.output_base),
            prompt=wf_input.prompt,
            new_run_id=actual_run_id,
        )
    elif wf_input.mode == "resume":
        if not wf_input.target_module_id:
            raise PipelineFailure("resume mode requires target_module_id")
        if not wf_input.run_dir:
            raise PipelineFailure("resume mode requires run_dir")
        run_dir = Path(wf_input.run_dir)
        prepared_run = _Run.from_journal(
            actual_run_id,
            target_module_id=wf_input.target_module_id,
            prompt=wf_input.prompt,
        )
    else:
        raise PipelineFailure(f"Unknown mode: {wf_input.mode!r}")

    # Fork-override application — only present when the server's
    # ``/fork`` endpoint pre-staged a row for this ``workflow_id``
    # before invoking ``DBOS.fork_workflow``. Sets the same fields the
    # legacy ``Run.from_fork`` did, so the existing replay-execute
    # machinery (mode flip in ``start_*_module`` + agent FORK override)
    # carries the run forward without needing a separate fork code
    # path further down. ``_fork_session_ids`` carries parent's task →
    # session_id mapping so the agent backend can FORK-resume each
    # session-bearing child without walking the run tree.
    override = read_fork_override(actual_run_id)
    if override is not None:
        prepared_run.prompt = override["prompt"]
        prepared_run._pending_resume_target = override["target_module_id"]
        prepared_run._playback_mode = "replay"
        prepared_run._fork_session_ids = override["session_ids"]

    # ── Background sibling workflows ─────────────────────────────
    # Start two ``@DBOS.workflow`` siblings keyed off ``actual_run_id``:
    # one consumes ``DBOS.send_async`` injects from the server's
    # ``/send_message`` endpoint, the other periodically summarises
    # the journal. Each gets its own deterministic ``workflow_uuid``
    # via ``SetWorkflowID`` so the parent's body code can recompute
    # the same id on cancel without storing the handles. Each runs
    # concurrently but in **its own function-id space** — that's why
    # they're separate workflows rather than ``asyncio.create_task``
    # siblings (which would interleave step calls with the parent
    # body and break replay determinism with
    # ``DBOSUnexpectedStepError``).
    from dbos import SetWorkflowID

    from aii_pipeline.run.config import load_dbos_run_config
    from aii_pipeline.run.workflows._background import (
        interim_summary_workflow,
        recv_workflow_id,
        send_message_recv_workflow,
        summary_workflow_id,
    )

    dbos_run_cfg = load_dbos_run_config()
    recv_id = recv_workflow_id(actual_run_id)
    summary_id = summary_workflow_id(actual_run_id)
    summary_started = False

    with SetWorkflowID(recv_id):
        await DBOS.start_workflow_async(send_message_recv_workflow, actual_run_id)
    if dbos_run_cfg.interim_summary.enabled:
        with SetWorkflowID(summary_id):
            await DBOS.start_workflow_async(
                interim_summary_workflow,
                actual_run_id,
                dbos_run_cfg.interim_summary.model_dump(mode="json"),
            )
        summary_started = True

    try:
        result = await run_pipeline(
            config=config,
            run_dir=run_dir,
            prepared_run=prepared_run,
        )
    finally:
        # Cancel siblings so they don't outlive the parent. The async
        # variant is required because we're inside an async DBOS workflow
        # — the sync ``cancel_workflow`` raises when a loop is running.
        await DBOS.cancel_workflow_async(recv_id)
        if summary_started:
            await DBOS.cancel_workflow_async(summary_id)

    return {
        "run_id": prepared_run.node_id,
        "run_dir": str(run_dir),
        "status": "completed" if result is not None else "failed",
        "mode": wf_input.mode,
    }
