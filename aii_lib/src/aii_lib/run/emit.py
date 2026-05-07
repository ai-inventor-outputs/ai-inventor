"""Direct emit helpers — journal each event into DBOS's ``operation_outputs``.

Free-function API mirroring the legacy ``Run`` methods (``status_*``,
``start_seq_group``, ``start_task``, …). Each helper journals one
message dict via :func:`journal_event_step` so the events endpoint
can serve them via cursor pagination over DBOS's journal table — the
single source of truth for run telemetry.

Parent-id resolution:

  * Most ``status_*`` / ``agent_*`` / ``llm_*`` calls associate the
    event with the active workflow (= the run). Helpers default
    ``parent_id`` to ``DBOS.workflow_id()`` so callers don't need to
    thread it explicitly.
  * Lifecycle calls (``start_module``, ``end_task``, …) take ids
    explicitly — same shape as the legacy methods.

Usage::

    from aii_lib.run import emit
    emit.status_public_info("running gen_strat")
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from loguru import logger

from aii_lib.run.journal import journal_event_step


def _resolve_parent_id(parent_id: str | None) -> str:
    """Default unset ``parent_id`` to the current DBOS workflow id."""
    if parent_id:
        return parent_id
    from dbos import DBOS

    return DBOS.workflow_id or ""


def _emit(msg_type: str, **fields: Any) -> None:
    """Journal one event under the active workflow + apply Run-tree side effects.

    Two writes per call:

      1. ``journal_event_step(payload)`` records the event in DBOS's
         ``operation_outputs`` — the canonical wire. Outside a DBOS
         workflow context the step decorator raises; we log at debug
         level and drop so test / manual-script imports don't blow up.

      2. When a :class:`~aii_lib.run.run.Run` is in scope and not in
         replay mode, we re-hydrate the payload into a typed
         :class:`~aii_lib.run.messages.BaseMessage` and call
         :func:`~aii_lib.run.dispatch.dispatch_event` so the in-memory
         tree mutates in lockstep (state transitions, task session-id
         capture, per-node ``messages`` list append). This keeps
         ``emit.X`` semantically equivalent to the legacy
         ``Run.X`` → ``self._on(msg)`` path without going through the
         Run bus / sink fan-out.
    """
    fields["type"] = msg_type
    if "parent_id" in fields:
        fields["parent_id"] = _resolve_parent_id(fields.get("parent_id"))
    payload = {k: v for k, v in fields.items() if v is not None}
    try:
        journal_event_step(payload)
    except Exception:
        logger.opt(exception=True).debug(
            f"emit._emit: journal_event_step failed (msg_type={msg_type!r}) — "
            f"expected only outside a DBOS workflow context"
        )

    # Apply to the active Run's in-memory tree (state transitions +
    # per-node messages list) THEN fan out to Run-bus subscribers.
    # Skipped in replay mode — the tree was rebuilt from disk during
    # boot and sinks listen for live events only; re-applying here
    # would duplicate work and emit ghosts.
    from aii_lib.run.context import get_current_run

    run = get_current_run()
    if run is None or run._playback_mode == "replay":
        return
    try:
        from aii_lib.run.dispatch import dispatch_event
        from aii_lib.run.messages import parse_message

        msg = parse_message(payload)
        dispatch_event(run, msg)
    except Exception:
        logger.opt(exception=True).debug(
            f"emit._emit: dispatch_event failed (msg_type={msg_type!r})"
        )
        return

    # Run-bus fan-out for legacy in-process subscribers
    # (agent_worker BufferSink, test sinks). No-op in production
    # pipeline now that prod sinks consume the journal via JournalTailer
    # — only out-of-process / agent-worker contexts where DBOS isn't
    # writable rely on this path. Wrapped in try so a bad sink can't
    # block the rest of emit's caller.
    try:
        run._notify(msg)
    except Exception:
        logger.opt(exception=True).debug(f"emit._emit: run._notify failed (msg_type={msg_type!r})")


def _gen_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _derive_path_and_id(*, parent_id: str, name: str) -> tuple[str, str]:
    """Compute the deterministic path-derived id for a new structural node.

    Mirrors :meth:`aii_lib.run.run.Run._emit_path_and_id` so live emit,
    boot-time scaffold, and replay all converge on the same id for a
    node at the same structural address (``{parent_path}/{name}[{idx}]``).
    Without this, ``emit.start_loop_group("hypo_loop")`` would generate
    a random uuid that never matches the scaffold's pre-created node id,
    crashing the pipeline at the first ``assert loop_gid == self.node_id``.

    Returns ``(path, node_id)``. When no :class:`Run` is in scope (tests
    / non-workflow contexts) we can't compute a path — fall back to
    ``("", random_id)`` so unit tests of emit.* helpers stay stand-alone.
    """
    from aii_lib.run.context import get_current_run

    run = get_current_run()
    if run is None:
        return "", _gen_id(name)
    return run._emit_path_and_id(parent_id=parent_id, name=name)


# ─── Status messages ────────────────────────────────────────────────


def status_public_info(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_info`` — FE-visible, LLM-summarized."""
    _emit("status_public_info", parent_id=parent_id, text=text, **extra)


def status_public_warning(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_warning`` — FE-visible."""
    _emit("status_public_warning", parent_id=parent_id, text=text, **extra)


def status_public_error(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_error`` — FE-visible, LLM-summarized."""
    _emit("status_public_error", parent_id=parent_id, text=text, **extra)


def status_public_success(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_success`` — FE-visible."""
    _emit("status_public_success", parent_id=parent_id, text=text, **extra)


def status_public_progress(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_progress`` — FE-visible, LLM-summarized."""
    _emit("status_public_progress", parent_id=parent_id, text=text, **extra)


_ARTIFACT_KIND_LABELS: dict[str, str] = {
    "paper_pdf": "Paper",
    "github_repo": "Repo",
}


def _format_artifact_published_text(artifacts: list[Any] | None) -> str:
    """Human-readable summary embedding each artifact's URL — wire ``text``."""
    if not artifacts:
        return "Published 0 artifacts"
    parts = [f"{_ARTIFACT_KIND_LABELS.get(a.kind, a.kind)}: {a.url}" for a in artifacts]
    return "Published — " + "  ·  ".join(parts)


def status_public_published(
    *,
    run_id: str | None = None,
    artifacts: Any = None,
    parent_id: str | None = None,
    text: str | None = None,
    **extra: Any,
) -> None:
    """Emit ``status_public_published`` — run-level deliverables (artifacts list)."""
    if text is None:
        text = _format_artifact_published_text(artifacts)
    _emit(
        "status_public_published",
        parent_id=parent_id,
        run_id=run_id,
        artifacts=artifacts,
        text=text,
        **extra,
    )


def status_public_interim_summary(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_public_interim_summary`` — periodic LLM narrative."""
    _emit("status_public_interim_summary", parent_id=parent_id, text=text, **extra)


def status_private_info(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_private_info`` — pipeline-internal log line."""
    _emit("status_private_info", parent_id=parent_id, text=text, **extra)


def status_private_debug(text: str, *, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``status_private_debug`` — dev-only diagnostic."""
    _emit("status_private_debug", parent_id=parent_id, text=text, **extra)


# ─── Group / iteration / module / task lifecycle ───────────────────


def start_seq_group(name: str, *, parent_id: str | None = None, **extra: Any) -> str:
    """Emit ``mdgroup_start`` for a sequential MdGroup; return the new group id.

    Mirrors legacy ``Run.start_seq_group``: the new group's ``parent_id``
    on the wire is the group's own id (the message owner is the group
    itself), while the structural parent is the run/group that was
    active before this start. We default ``parent_id`` to the run's
    workflow id via :func:`_resolve_parent_id` when callers don't pass
    it, then derive a deterministic ``path`` + ``group_id`` together.
    """
    structural_parent = _resolve_parent_id(parent_id)
    path, gid = _derive_path_and_id(parent_id=structural_parent, name=name)
    _emit(
        "mdgroup_start",
        group_id=gid,
        group_type="seq",
        name=name,
        parent_id=gid,
        group=gid,
        text=name,
        path=path,
        **extra,
    )
    return gid


def start_loop_group(name: str, *, parent_id: str | None = None, **extra: Any) -> str:
    """Emit ``mdgroup_start`` for a loop MdGroup; return the new group id."""
    structural_parent = _resolve_parent_id(parent_id)
    path, gid = _derive_path_and_id(parent_id=structural_parent, name=name)
    _emit(
        "mdgroup_start",
        group_id=gid,
        group_type="loop",
        name=name,
        parent_id=gid,
        group=gid,
        text=name,
        path=path,
        **extra,
    )
    return gid


def end_group(group_id: str, *, status: str = "done", **extra: Any) -> None:
    """Emit ``mdgroup_end`` for ``group_id`` with the given terminal status."""
    _emit("mdgroup_end", group_id=group_id, parent_id=group_id, status=status, **extra)


def start_iteration(*, group_id: str, iteration: int, **extra: Any) -> str:
    """Emit ``iteration_start`` under a loop group; return the iteration node id.

    ``iter{N}`` (no underscore) matches both ``Run.start_iteration`` and
    the scaffold (``aii_pipeline.run.scaffold._ensure_loop_group``) so
    the path-derived id under ``group_id`` resolves to the same node the
    scaffold pre-created.
    """
    iter_name = f"iter{iteration}"
    path, iid = _derive_path_and_id(parent_id=group_id, name=iter_name)
    _emit(
        "iteration_start",
        iteration_id=iid,
        iteration=iteration,
        parent_id=group_id,
        group_id=group_id,
        path=path,
        **extra,
    )
    return iid


def end_iteration(*, group_id: str, iteration: int, **extra: Any) -> None:
    """Emit ``iteration_end`` for the given loop iteration."""
    _emit(
        "iteration_end",
        group_id=group_id,
        iteration=iteration,
        parent_id=group_id,
        **extra,
    )


def start_single_module(*, name: str, parent_id: str, **extra: Any) -> str:
    """Emit ``module_start`` for a single-T module; return the new module id."""
    path, mid = _derive_path_and_id(parent_id=parent_id, name=name)
    _emit(
        "module_start",
        module_id=mid,
        module_type="single",
        name=name,
        parent_id=mid,
        attach_under_id=parent_id,
        path=path,
        **extra,
    )
    return mid


def start_parallel_module(*, name: str, parent_id: str, **extra: Any) -> str:
    """Emit ``module_start`` for a parallel module; return the new module id."""
    path, mid = _derive_path_and_id(parent_id=parent_id, name=name)
    _emit(
        "module_start",
        module_id=mid,
        module_type="parallel",
        name=name,
        parent_id=mid,
        attach_under_id=parent_id,
        path=path,
        **extra,
    )
    return mid


def end_module(*, parent_id: str, module_id: str, status: str = "done", **extra: Any) -> None:
    """Emit ``module_end`` for ``module_id`` under ``parent_id``."""
    _emit(
        "module_end",
        module_id=module_id,
        parent_id=module_id,
        attach_under_id=parent_id,
        status=status,
        **extra,
    )


def start_task(
    *,
    name: str,
    parent_module_id: str,
    model: str | None = None,
    session: str | None = None,
    module: str | None = None,
    group: str | None = None,
    **extra: Any,
) -> str:
    """Emit ``task_start`` under a module; return the new task id.

    ``model`` / ``session`` are folded into the message ``extras`` dict
    (where dispatch + OTel readers expect them). ``module`` / ``group``
    become top-level fields on :class:`TaskStartMessage`. ``name`` is
    written as both ``task_name`` and ``agent_context`` to match the
    legacy ``Run.start_task`` wire shape.
    """
    path, tid = _derive_path_and_id(parent_id=parent_module_id, name=name)
    ev_extras: dict[str, Any] = {}
    if model:
        ev_extras["model"] = model
    if session:
        ev_extras["session_id"] = session

    fields: dict[str, Any] = {
        "task_id": tid,
        "task_name": name,
        "agent_context": name,
        "parent_id": tid,
        "attach_under_id": parent_module_id,
        "path": path,
    }
    if module:
        fields["module"] = module
    if group:
        fields["group"] = group
    if ev_extras:
        fields["extras"] = ev_extras
    fields.update(extra)
    _emit("task_start", **fields)
    return tid


def end_task(
    task_id: str,
    *,
    status: str = "done",
    cost_usd: float | None = None,
    text: str | None = None,
    module: str | None = None,
    group: str | None = None,
    name: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit ``task_end`` with terminal status + display text.

    Mirrors legacy ``Run.end_task``: ``cost_usd`` → ``extras.total_cost``,
    ``session_id`` → ``extras.session_id``, ``name`` → ``task_name`` +
    ``agent_context``. Default ``text`` follows the status token
    (``"OK"`` / ``"FAILED"`` / ``"STOPPED"``).
    """
    if status not in ("done", "failed", "stopped"):
        raise ValueError(f"end_task status must be 'done', 'failed', or 'stopped'; got {status!r}")
    token = {"done": "OK", "failed": "FAILED", "stopped": "STOPPED"}[status]
    text_value = text or token

    ev_extras: dict[str, Any] = {}
    if cost_usd is not None:
        ev_extras["total_cost"] = cost_usd
    if session_id:
        ev_extras["session_id"] = session_id

    fields: dict[str, Any] = {
        "task_id": task_id,
        "parent_id": task_id,
        "status": status,
        "text": text_value,
    }
    if name:
        fields["task_name"] = name
        fields["agent_context"] = name
    if module:
        fields["module"] = module
    if group:
        fields["group"] = group
    if ev_extras:
        fields["extras"] = ev_extras
    fields.update(extra)
    _emit("task_end", **fields)


# ─── Output emits ───────────────────────────────────────────────────


def run_output(*, output: Any = None, parent_id: str | None = None, **extra: Any) -> None:
    """Emit ``run_output`` — the run's final aggregate result."""
    _emit("run_output", parent_id=parent_id, output=output, **extra)


def mdgroup_output(*, group_id: str, output: Any = None, **extra: Any) -> None:
    """Emit ``mdgroup_output`` — phase-aggregate result for an MdGroup."""
    _emit("mdgroup_output", parent_id=group_id, group_id=group_id, output=output, **extra)


def module_output(*, module_id: str, name: str = "", output: Any = None, **extra: Any) -> None:
    """Emit ``module_output`` — structured output of a module."""
    _emit(
        "module_output",
        parent_id=module_id,
        module_id=module_id,
        name=name,
        output=output,
        **extra,
    )


def task_output(*, task_id: str, output: Any = None, **extra: Any) -> None:
    """Emit ``task_output`` — single-task structured result."""
    _emit("task_output", parent_id=task_id, task_id=task_id, output=output, **extra)


# ─── Agent + LLM messages ───────────────────────────────────────────


def agent_start(task_id: str, **extra: Any) -> None:
    """Emit ``agent_start`` — LLM-call open bracket inside a task."""
    _emit("agent_start", parent_id=task_id, task_id=task_id, **extra)


def agent_end(task_id: str, *, session_id: str | None = None, text: str = "", **extra: Any) -> None:
    """Emit ``agent_end`` — LLM-call close bracket; ``session_id`` for fork/resume."""
    _emit(
        "agent_end",
        parent_id=task_id,
        task_id=task_id,
        session_id=session_id,
        text=text,
        **extra,
    )


def agent_retry(
    task_id: str,
    *,
    attempt: int = 0,
    reason: str = "",
    **extra: Any,
) -> None:
    """Emit ``agent_retry`` — agent loop retry attempt."""
    _emit(
        "agent_retry",
        parent_id=task_id,
        task_id=task_id,
        attempt=attempt,
        reason=reason,
        **extra,
    )


def agent_hook(task_id: str, **extra: Any) -> None:
    """Emit ``agent_hook`` — SDK hook callback fired (PostToolUse, …)."""
    _emit("agent_hook", parent_id=task_id, task_id=task_id, **extra)


def agent_user_prompt(task_id: str, text: str = "", **extra: Any) -> None:
    """Emit ``agent_user_prompt`` — user-side message into the agent stream."""
    _emit(
        "agent_user_prompt",
        parent_id=task_id,
        task_id=task_id,
        text=text,
        **extra,
    )


# ─── Run lifecycle ──────────────────────────────────────────────────


def run_start(*, run_id: str, **extra: Any) -> None:
    """Emit ``run_start`` — the run's open bracket event."""
    _emit("run_start", run_id=run_id, parent_id=run_id, **extra)


def run_end(*, run_id: str, status: str = "completed", **extra: Any) -> None:
    """Emit ``run_end`` — the run's close bracket with terminal status.

    Default ``text`` is ``"Pipeline {status}"`` and ``extras.status``
    mirrors the top-level field so legacy consumers that look only in
    extras keep working. Callers can override either via ``**extra``.
    """
    extra.setdefault("text", f"Pipeline {status}")
    extras = extra.setdefault("extras", {})
    if isinstance(extras, dict):
        extras.setdefault("status", status)
    _emit("run_end", run_id=run_id, parent_id=run_id, status=status, **extra)


def finalize_orphans(*, status: str = "stopped") -> None:
    """Walk the active Run's tree and end every IN_PROGRESS descendant.

    Defense-in-depth for crash / early-return paths that bypass
    per-phase ``end_*`` calls. Walks leaves-first so parents see their
    children already terminal. Each per-node emit is wrapped so one bad
    node (missing parent in the index, dispatch rejection) doesn't
    abort the rest of the walk.

    Outside a Run scope this is a no-op.
    """
    from aii_lib.run.context import get_current_run
    from aii_lib.run.loop_iteration import LoopIteration
    from aii_lib.run.mdgroup import LoopMdGroup, MdGroup
    from aii_lib.run.module import Module
    from aii_lib.run.node import NodeStatus
    from aii_lib.run.task import Task

    run = get_current_run()
    if run is None:
        return

    in_progress = [
        n
        for n in run._node_index.nodeid_to_node.values()
        if getattr(n, "status", None) == NodeStatus.IN_PROGRESS and n is not run
    ]

    def _depth(node: Any) -> int:
        d = 0
        cur: Any = node
        while getattr(cur, "parent_id", None):
            d += 1
            cur = run._node_index.nodeid_to_node.get(cur.parent_id)
            if cur is None or cur is run:
                break
        return d

    in_progress.sort(key=lambda n: -_depth(n))

    for node in in_progress:
        try:
            if isinstance(node, Task):
                end_task(node.node_id, status=status)
            elif isinstance(node, Module):
                end_module(parent_id=node.parent_id, module_id=node.node_id, status=status)
            elif isinstance(node, LoopIteration):
                parent = run._node_index.nodeid_to_node.get(node.parent_id)
                if isinstance(parent, LoopMdGroup):
                    iter_n = parent.iteration_number(node)
                    if iter_n is not None:
                        end_iteration(
                            group_id=parent.node_id,
                            iteration=iter_n,
                            status=status,
                        )
            elif isinstance(node, MdGroup):
                end_group(node.node_id, status=status)
        except Exception:
            logger.opt(exception=True).warning(
                f"emit.finalize_orphans: end_* failed for {type(node).__name__} {node.node_id!r}"
            )


def end_run(*, run_id: str, status: str = "completed", **extra: Any) -> None:
    """Finalize any in-progress descendants then emit ``run_end``.

    Top-level public API for terminating a run cleanly. Combines
    :func:`finalize_orphans` (defense-in-depth for crash / early-return
    paths) and :func:`run_end` (the actual close bracket), preserving
    the legacy ``Run.end()`` semantics on a free-function entry point.
    """
    try:
        finalize_orphans(status="stopped")
    except Exception:
        logger.opt(exception=True).debug(
            "emit.end_run: finalize_orphans raised — emitting run_end anyway"
        )
    run_end(run_id=run_id, status=status, **extra)
