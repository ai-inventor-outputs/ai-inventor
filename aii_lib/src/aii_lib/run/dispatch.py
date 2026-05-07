"""Event → Run-state dispatch (the v26 single-write reducer).

``dispatch_event(run, event)`` is the apply step in ``Run._on``: it
mutates ``run`` to reflect ``event``'s effect. It carries no telemetry
side-effects — those are the subscriber list's job.

It accepts both:

  - typed ``BaseMessage`` instances (the live-bus path), and
  - dict-shape events (the JSONL replay path), via
    ``aii_lib.run.messages.parse_message``.

The match-statement keeps each case tiny: each branch reads only the
fields it needs and calls the matching domain primitive on ``run`` /
its child objects. The four ``*_output`` events
(``run_output`` / ``mdgroup_output`` / ``module_output`` / ``task_output``)
each set the matching node's :attr:`AIINode.output` attribute — found
via ``run.find_node(event.parent_id)`` — so live execution and replay
both reach the same state.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aii_lib.run.loop_iteration import LoopIteration
from aii_lib.run.mdgroup import LoopMdGroup, SeqMdGroup
from aii_lib.run.messages import (
    _MESSAGE_CLASSES,
    AgentConfigMessage,
    AgentEndMessage,
    AgentMessageDeltaMessage,
    AgentStartMessage,
    AgentSummaryMessage,
    BaseMessage,
    GroupEndMessage,
    GroupStartMessage,
    IterationEndMessage,
    IterationStartMessage,
    LlmSummaryMessage,
    MdGroupOutputMessage,
    ModuleEndMessage,
    ModuleOutputMessage,
    ModuleStartMessage,
    RunEndMessage,
    RunOutputMessage,
    RunStartMessage,
    RunTitleMessage,
    StatusPublicPublishedMessage,
    TaskEndMessage,
    TaskOutputMessage,
    TaskStartMessage,
    parse_message,
)
from aii_lib.run.module import Module, ParallelTModule
from aii_lib.run.node import NodeStatus

if TYPE_CHECKING:
    from aii_lib.run.messages import AgentMessage
    from aii_lib.run.node import AIINode
    from aii_lib.run.run import Run
    from aii_lib.run.task import ClaudeAgentTask, Task


# ---------------------------------------------------------------------------
# Helpers (shared with serializer)
# ---------------------------------------------------------------------------


def _parse_ts(s: str | datetime | None) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo is not None else s.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _ensure_task_in_module(
    run: Run,
    module: Module,
    *,
    task_id: str,
    name: str = "",
    path: str = "",
) -> ClaudeAgentTask:
    from aii_lib.run.task import ClaudeAgentTask

    cached = run.find_task(task_id)
    if isinstance(cached, ClaudeAgentTask):
        return cached

    # Tasks keep their canonical step-supplied name verbatim. Identity
    # is the path-derived ``node_id``; same-named slots inside a parallel
    # module are disambiguated structurally by the ``[N]`` suffix on
    # their structural path (``parent.path/name[idx]``).
    task = ClaudeAgentTask(
        node_id=task_id,
        name=name,
        parent_id=module.node_id,
        path=path,
    )
    module.add_task(task)
    run._index(task)
    return task


def _ensure_task_for_id(
    run: Run,
    task_id: str,
    name: str = "",
) -> Task | None:
    """Pure cache lookup for a task by ID.

    Task created via ``Run.start_task`` (live) or ``_ensure_task_in_module``
    from ``_apply_task_lifecycle`` (replay via ``parent_id``). No tree
    walking, no name-prefix matching: every routing path goes through
    ``parent_id`` now.
    """
    if not isinstance(task_id, str) or not task_id:
        return None
    return run.find_task(task_id)


# ---------------------------------------------------------------------------
# Per-event apply functions
# ---------------------------------------------------------------------------


def _apply_run_start(run: Run, e: RunStartMessage) -> None:
    # Idempotent: already started → no-op (re-running execute() in
    # replay-execute mode emits this twice for the same run).
    if run.status != NodeStatus.PENDING:
        return
    run._apply_start(ts=_parse_ts(e.end_at))


def _apply_run_end(run: Run, e: RunEndMessage) -> None:
    """Map status token to NodeStatus and apply.

    Unrecognized tokens collapse to STOPPED — covers crashes / interrupts
    that don't fit a clean DONE/FAILED.
    """
    # Idempotent: terminal status already set → preserve original (and
    # its end_at). Replay-execute may re-emit this event during the
    # re-run of execute() over a tree already finalized at boot.
    if run.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.STOPPED):
        return
    mapping = {
        "completed": NodeStatus.DONE,
        "complete": NodeStatus.DONE,
        "failed": NodeStatus.FAILED,
        "failure": NodeStatus.FAILED,
        "crashed": NodeStatus.FAILED,
        "stopped": NodeStatus.STOPPED,
        "interrupted": NodeStatus.STOPPED,
    }
    status = mapping.get(e.status.lower(), NodeStatus.STOPPED)
    run._apply_end(status=status, ts=_parse_ts(e.end_at))


def _apply_run_title(run: Run, e: RunTitleMessage) -> None:
    """Slugify title and store as run name.

    Converts to lowercase_underscore form and matches the naming convention
    every other tree node follows. No separate human-form field; the slug
    IS the canonical label.
    """
    # Idempotent: title already set → keep original. The first
    # successful title set wins; replay re-emits don't overwrite.
    if run.name:
        return
    text = (e.text or "").strip()
    run.name = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() if text else ""


def _apply_mdgroup_start(run: Run, e: GroupStartMessage) -> None:
    group_id = e.group_id
    if not group_id:
        return
    # Direct id hit — same instance already in the tree. Live boot's
    # scaffold pre-created the group with this id (start_seq_group /
    # start_loop_group reuse the scaffold's node_id); a duplicate
    # replay would also land here.
    existing = run.find_group(group_id)
    if existing is not None:
        # Idempotent: only flip PENDING → IN_PROGRESS. If already
        # past PENDING (live scaffold-boot has flipped it; replay-
        # execute is re-emitting), leave as-is.
        if existing.status == NodeStatus.PENDING:
            existing._apply_start()
        return
    # Pure-replay (from_resume / from_fork) or test-fixture path —
    # ask the pipeline-side resolver for a typed subclass (registered
    # via :func:`set_group_class_resolver`); fall back to the generic
    # base when no subclass is registered for this name.
    from aii_lib.run.run import resolve_group_class

    default_cls = LoopMdGroup if e.group_type == "loop" else SeqMdGroup
    typed_cls = resolve_group_class(e.name or "", e.group_type) if e.name else None
    cls = typed_cls if typed_cls is not None else default_cls
    g = cls(
        node_id=group_id,
        name=e.name or "",
        parent_id=run.node_id,
        path=getattr(e, "path", "") or "",
    )
    g._apply_start()
    run.children.append(g)
    run._index(g)


def _apply_mdgroup_end(run: Run, e: GroupEndMessage) -> None:
    if not e.group_id:
        return
    g = run.find_group(e.group_id)
    if g is None:
        return
    # Idempotent: terminal status already set → preserve original.
    if g.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.STOPPED):
        return
    override = NodeStatus.STOPPED if e.status == "stopped" else None
    g._apply_end(status_override=override)


def _apply_iteration_start(run: Run, e: IterationStartMessage) -> None:
    from aii_lib.run.loop_iteration import LoopIteration
    from aii_lib.run.node_id import generate_short_id

    if not e.group_id or e.iteration is None:
        return
    g = run.find_group(e.group_id)
    if not isinstance(g, LoopMdGroup):
        return
    existing = g.find_iteration(e.iteration)
    if existing is not None:
        # Pre-existing iteration (boot-time scaffold or duplicate event).
        # ``start_iteration`` reuses the scaffold's node_id when emitting
        # the message, so the iteration_id always matches and no
        # reindex is needed — just flip status PENDING → IN_PROGRESS.
        # Idempotent: only flip if PENDING. Replay-execute re-emits
        # against an already-IN_PROGRESS / terminal iteration → no-op.
        if existing.status == NodeStatus.PENDING:
            existing._apply_start()
        return
    # Pure-replay path: honor the producer's iteration_id so the
    # caller-held id matches the indexed node.
    iid = e.iteration_id or generate_short_id()
    it = LoopIteration(
        node_id=iid,
        parent_id=g.node_id,
        path=getattr(e, "path", "") or "",
    )
    it._apply_start()
    g._apply_iteration_started(it)
    run._index(it)


def _apply_iteration_end(run: Run, e: IterationEndMessage) -> None:
    if not e.group_id or e.iteration is None:
        return
    g = run.find_group(e.group_id)
    if not isinstance(g, LoopMdGroup):
        return
    # Idempotent: skip if iteration is already terminal.
    existing_iter = g.find_iteration(e.iteration)
    if existing_iter is not None and existing_iter.status in (
        NodeStatus.DONE,
        NodeStatus.FAILED,
        NodeStatus.STOPPED,
    ):
        return
    override = NodeStatus.STOPPED if e.status == "stopped" else None
    g._apply_iteration_ended(e.iteration, status_override=override)


def _apply_module_start(run: Run, e: ModuleStartMessage) -> None:
    from aii_lib.run.module import SingleTModule

    if not e.module_type or e.module_type not in ("single", "parallel"):
        return
    if not e.name or not e.module_id or not e.attach_under_id:
        return

    existing = run.find_module(e.module_id)
    if existing is not None:
        # Live boot path: scaffold pre-created the module with this
        # exact id (``start_*_module`` reuses the scaffold's node_id).
        # Just flip status PENDING → IN_PROGRESS.
        # Idempotent: only flip if PENDING. Replay-execute re-emits
        # against an already-IN_PROGRESS / terminal module → no-op.
        if existing.status == NodeStatus.PENDING:
            existing._apply_start()
        return

    # Resolve the structural parent. ``attach_under_id`` is the
    # auto-gen node_id of a SeqMdGroup or LoopIteration — looked up
    # directly in the flat node index. (``parent_id`` on the message
    # is the *owner* of the message — the module being started — not
    # the structural parent; see :class:`ModuleStartMessage`.)
    direct_parent = run.find_node(e.attach_under_id)
    if isinstance(direct_parent, (SeqMdGroup, LoopIteration)):
        parent_node_id = direct_parent.node_id
    else:
        parent_node_id = None

    # Construct as PENDING (the AIINode default) and let ``_apply_start``
    # flip the status. Ask the pipeline-side resolver (registered via
    # :func:`set_module_class_resolver`) for the typed substep
    # subclass — falls back to the generic base if no subclass is
    # registered for this name.
    from aii_lib.run.run import resolve_module_class

    typed_cls = resolve_module_class(e.name, e.module_type)
    base_cls = ParallelTModule if e.module_type == "parallel" else SingleTModule
    cls = typed_cls if typed_cls is not None else base_cls
    module: Module = cls(
        node_id=e.module_id,
        name=e.name,
        parent_id=parent_node_id,
        path=getattr(e, "path", "") or "",
    )
    module._apply_start()
    run._attach_module(parent_id=e.attach_under_id, module=module)


def _apply_module_end(run: Run, e: ModuleEndMessage) -> None:
    if not e.parent_id or not e.module_id:
        return
    m = run.find_module(e.module_id)
    if m is not None:
        # Idempotent: terminal status already set → preserve original.
        # Resume cleanup logic still fires (see below) so the marker
        # clears even if the end event arrives twice.
        if m.status not in (
            NodeStatus.DONE,
            NodeStatus.FAILED,
            NodeStatus.STOPPED,
        ):
            override = NodeStatus.STOPPED if e.status == "stopped" else None
            m._apply_end(status_override=override)
    # Resume cleanup: when the resume target's module_end fires DURING
    # LIVE EXECUTE, the resume turn is over — every downstream substep
    # dispatch is fresh. Clear the marker so the agent backend stops the
    # FORK override (which gates on ``_pending_resume_target``).
    #
    # During REPLAY, the target's module_end is just history being
    # replayed onto the rebuilt tree — live execution hasn't reached
    # the target yet, so the marker MUST stay armed. The mode-flip in
    # :meth:`Run.start_*_module` consumes the marker at the live
    # boundary; this same handler then fires once more in live mode
    # (when forward execute() emits the post-resume module_end) and
    # clears it on that pass.
    if run._playback_mode == "live" and run._pending_resume_target == e.module_id:
        run._pending_resume_target = None


_TASK_END_STATUS_MAP = {
    "done": NodeStatus.DONE,
    "failed": NodeStatus.FAILED,
    "stopped": NodeStatus.STOPPED,
}


def _apply_task_lifecycle(run: Run, e: BaseMessage, *, end: bool) -> None:
    tid = getattr(e, "task_id", "") or ""
    if not tid:
        return
    name = getattr(e, "task_name", "") or ""

    # ``attach_under_id`` is the routing signal for fresh task_start —
    # emitters always set it via ``Run.start_task(..., parent_module_id=...)``.
    # task_end has no attach decision (the task already exists).
    # ``parent_id`` on the message is the *owner* (= the task itself),
    # not the structural parent; see :class:`TaskStartMessage`.
    attach_under_id = getattr(e, "attach_under_id", "") or ""
    cached = run.find_task(tid)
    if cached is not None:
        task = cached
    elif attach_under_id:
        parent = run.find_node(attach_under_id)
        if not isinstance(parent, Module):
            return
        path = getattr(e, "path", "") or ""
        task = _ensure_task_in_module(run, parent, task_id=tid, name=name, path=path)
    else:
        return
    if task is None:
        return
    # Pre-existing task — backfill ``name`` if event carries one and the
    # task didn't have it yet (e.g. task_end arrives after task_start
    # populated only ``node_id``).
    if name and not task.name:
        task.name = name

    ts = _parse_ts(getattr(e, "ts", None))
    module = run.find_module_for_task(tid)
    if module is None:
        return
    if end:
        # Idempotent: terminal status already set → preserve original
        # (and its end_at). Replay-execute re-emit lands here.
        if task.status in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.STOPPED):
            return
        status_str = getattr(e, "status", None) or "done"
        new_status = _TASK_END_STATUS_MAP.get(status_str, NodeStatus.DONE)
        task._apply_end(status=new_status, ts=ts)
        module._apply_task_ended()
    else:
        # Idempotent: skip if task already past PENDING. Re-applying
        # _apply_start would flip a DONE/IN_PROGRESS task back to
        # IN_PROGRESS (corruption); guard explicitly.
        if task.status != NodeStatus.PENDING:
            return
        task._apply_start(ts=ts)
        module._apply_task_started()

    if any(t.is_active for t in run.all_tasks):
        if run.status not in (NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.STOPPED):
            run.status = NodeStatus.IN_PROGRESS


def _apply_task_start(run: Run, e: TaskStartMessage) -> None:
    _apply_task_lifecycle(run, e, end=False)


def _apply_task_end(run: Run, e: TaskEndMessage) -> None:
    _apply_task_lifecycle(run, e, end=True)


def _session_id_from_event(e: BaseMessage) -> str | None:
    """Pull a non-empty SDK ``session_id`` off any agent event.

    Checks the typed top-level field first (``AgentEndMessage`` exposes
    one) then falls back to ``extras.session_id`` — agent_config carries
    it there, since ``BaseMessage`` has ``extra="allow"`` so unknown
    fields land in the auto-extras dict.
    """
    sid = getattr(e, "session_id", None)
    if isinstance(sid, str) and sid:
        return sid
    extras = getattr(e, "extras", None)
    if isinstance(extras, dict):
        sid = extras.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    return None


def _capture_task_session_id(run: Run, e: BaseMessage) -> None:
    """Set ``task.session_id`` from an agent event when present.

    Used for both ``agent_start`` / ``agent_config`` (in-flight capture so
    a worker killed before ``agent_end`` still leaves the session_id on
    its Task) and ``agent_end`` (canonical capture). Idempotent: writes
    only when the event carries a non-empty session_id and the task is
    findable. ``Task._apply_agent_end`` is the actual setter — agent_end
    semantics are "set if non-empty," matching what we want here.
    """
    tid = getattr(e, "task_id", "") or ""
    if not tid:
        return
    sid = _session_id_from_event(e)
    if not sid:
        return
    task = run.find_task(tid)
    if task is not None:
        task._apply_agent_end(session_id=sid)


def _apply_agent_start(run: Run, e: AgentMessage) -> None:
    """Capture session_id from event if available.

    Older event shapes sometimes attach it here, and capturing it eagerly
    means a worker killed between agent_start and agent_end still leaves a
    resumable session on its Task.
    """
    _capture_task_session_id(run, e)


def _apply_agent_config(run: Run, e: AgentMessage) -> None:
    """Capture session_id from the SDK init snapshot.

    The SDK emits agent_config the moment a session opens (``extras.
    session_id`` carries the SDK's UUID), so this is the earliest event
    that lets us bind a session to its Task. Without this handler, in-flight
    tasks (killed before agent_end) lose their session_id at replay time
    and fork/resume can't reattach to the conversation.
    """
    _capture_task_session_id(run, e)


def _apply_agent_end(run: Run, e: AgentEndMessage) -> None:
    _capture_task_session_id(run, e)


def _apply_status_published(run: Run, e: BaseMessage) -> None:
    """Reducer for ``status_published`` events.

    No-op against ``run`` state — the event is recorded on
    ``Run.events`` (the source of truth) by ``Run._on``. The published
    artifacts are surfaced on the wire via the to_app mapper which
    derives ``AppRun.published`` from the latest ``status_published``
    event scoped to a completed gen_paper_repo group.
    """
    return


def _apply_output(run: Run, e: BaseMessage) -> None:
    """Reducer for the four ``*_output`` events.

    Resolves the owning node via ``parent_id`` (every output message
    is emitted with ``parent_id`` set to the node that produced the
    output: run / mdgroup / module / task) and assigns
    ``event.output`` to ``node.output``.
    """
    owner_id = e.parent_id
    node = run if owner_id == run.node_id else run.find_node(owner_id)
    if node is None:
        return
    # Idempotent: first non-None output wins. Replay-execute re-emit
    # of an output that's already set is a no-op so the original
    # typed model survives untouched (preserves identity for any
    # downstream consumer that captured it).
    if getattr(node, "output", None) is not None:
        return
    node.output = e.output


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _ancestors_of(run: Run, node: AIINode) -> set[str]:
    """Collect all ancestor node IDs from a node up to the run.

    Walk from ``node`` up via ``parent_id`` collecting every ancestor's
    ``node_id`` (including ``node`` itself and the run). Used by
    ``Run.node_index`` so a scope-membership check is one set lookup. The
    walk uses :meth:`Run.find_node` which is kept current with the tree.
    """
    ancestors: set[str] = {run.node_id}
    cur: AIINode | None = node
    seen: set[str] = set()
    while cur is not None and cur.node_id not in seen:
        seen.add(cur.node_id)
        ancestors.add(cur.node_id)
        if cur is run:
            break
        pid = cur.parent_id
        if pid is None:
            break
        cur = run.find_node(pid)
    return ancestors


def _route_message(run: Run, event: BaseMessage) -> None:
    """Append message to node and add to run index.

    Appends ``event`` to the per-node ``messages`` list of its owning
    node AND adds it to ``run.node_index``. Owner = the node identified by
    ``event.parent_id`` (a required field on every :class:`BaseMessage`).
    Falls back to the run when the id is unknown — defensive only; emitters
    always set a real owner. The message is also indexed for fast cursor/
    scope lookups by the to_app sink, and its timestamp is pushed up the
    parent chain via :func:`update_derived_stats_from_message` so each
    ancestor's :attr:`NodeStats.runtime_seconds` and
    :attr:`NodeStats.total_messages` stay current.

    Stage 5 — replay-execute gate: when the run is in
    ``_playback_mode == "replay"`` we skip the per-node messages
    append and node-index add. Clone-log boot replay already
    populated both during the parent-events replay before
    ``execute()`` started; re-appending would duplicate every event
    on the in-memory tree. Stats updates also skipped since
    NodeStats was already accumulated to its final value at boot.
    """
    if run._playback_mode == "replay":
        return
    owner = run.find_node(event.parent_id) or run
    owner.messages.append(event)
    run.node_index.add_message(event, ancestors=_ancestors_of(run, owner))
    ts = event.end_at or event.start_at
    if ts is not None:
        from .node_stats_aggregator import update_derived_stats_from_message

        update_derived_stats_from_message(owner, ts)


def dispatch_event(run: Run, event: BaseMessage | dict | Any) -> None:
    """Apply ``event``'s state transition to ``run``.

    Accepts a typed ``BaseMessage`` (the live-bus path) or a dict /
    Pydantic model (the JSONL replay path — re-hydrated through
    ``parse_message``).

    Also routes the event onto the ``messages`` list of its
    scope-owning node (Run / MdGroup / LoopIteration / Module / Task)
    via :func:`_route_message`. ``Run.events`` is still maintained
    by ``Run._record`` alongside this routing during the migration.
    """
    if not isinstance(event, BaseMessage):
        event = parse_message(event)
    else:
        # Upgrade to the typed subclass when the runtime class doesn't
        # match what the wire ``type`` should map to. Covers two
        # ad-hoc construction patterns:
        #   1. ``BaseMessage(type="status_public_warning", ...)``
        #   2. ``SummarizedMessage(type="status_public_info", ...)``
        # — both used in tests / dynamic emit paths. Without this, the
        # class-pattern ``match event:`` arms below would skip them
        # (no matching ``StatusPublicWarningMessage`` / etc. case).
        # Modern production emit always uses the typed subclass
        # directly, so this is a no-op on the hot path.
        expected_cls = _MESSAGE_CLASSES.get(event.type)
        if expected_cls is not None and type(event) is not expected_cls:
            event = parse_message(event.model_dump())

    # Match-statement keeps the routing flat & easy to read.
    # Apply the state transition FIRST so structural events
    # (mdgroup_start, iteration_start, module_start, task_start) have
    # already created their scope-node by the time ``_route_message``
    # tries to find it. In live mode this is a no-op (the task etc.
    # was pre-created by the facade method); in replay mode it ensures
    # the routing target exists.
    match event:
        case RunStartMessage():
            _apply_run_start(run, event)
        case RunEndMessage():
            _apply_run_end(run, event)
            from .node_stats_aggregator import format_node_summary

            event.text = format_node_summary(f"Run {run.name or run.node_id}", run)
        case RunTitleMessage():
            _apply_run_title(run, event)
        case GroupStartMessage():
            _apply_mdgroup_start(run, event)
        case GroupEndMessage():
            _apply_mdgroup_end(run, event)
            g = run.find_group(event.group_id)
            if g is not None:
                from .node_stats_aggregator import format_node_summary

                event.text = format_node_summary(f"Group {g.name or g.node_id}", g)
        case IterationStartMessage():
            _apply_iteration_start(run, event)
        case IterationEndMessage():
            _apply_iteration_end(run, event)
            g = run.find_group(event.group_id)
            it = g.find_iteration(event.iteration or 0) if isinstance(g, LoopMdGroup) else None
            if it is not None:
                from .node_stats_aggregator import format_node_summary

                event.text = format_node_summary(f"Iter {event.iteration}", it)
        case ModuleStartMessage():
            _apply_module_start(run, event)
        case ModuleEndMessage():
            _apply_module_end(run, event)
            m = run.find_module(event.module_id)
            if m is not None:
                from .node_stats_aggregator import format_node_summary

                event.text = format_node_summary(f"Module {m.name or m.node_id}", m)
        case (
            ModuleOutputMessage()
            | RunOutputMessage()
            | MdGroupOutputMessage()
            | TaskOutputMessage()
        ):
            _apply_output(run, event)
        case TaskStartMessage():
            _apply_task_start(run, event)
        case TaskEndMessage():
            _apply_task_end(run, event)
            t = run.find_node(event.task_id)
            if t is not None:
                from .node_stats_aggregator import format_node_summary

                event.text = format_node_summary(f"Task {t.name or t.node_id}", t)
        case AgentSummaryMessage() | LlmSummaryMessage():
            from .node_stats_aggregator import apply_leaf_summary

            apply_leaf_summary(
                task_id=event.task_id,
                total_cost=event.total_cost,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cache_read_tokens=event.cache_read_tokens,
                cache_write_tokens=event.cache_write_tokens,
            )
        case AgentMessageDeltaMessage():
            # Mid-stream usage update — SET (overwrite) the live
            # ``current_all_input_tokens`` / ``current_all_output_tokens``
            # on the owning Task. NOT propagated to ancestors and NOT
            # added to ``cum_*`` (those are driven by ``agent_summary``
            # at end-of-call). Per-task only — Module / Group / Run
            # don't carry a meaningful "current call" value.
            #
            # ``current_all_input_tokens`` =
            #     input_tokens + cache_read_input_tokens + cache_creation_input_tokens
            # i.e. the FULL context window size at this moment (uncached
            # new input PLUS cache-read PLUS cache-write portions).
            if event.task_id:
                t = run.find_node(event.task_id)
                if t is not None and hasattr(t, "stats"):
                    t.stats.current_all_input_tokens = (
                        event.input_tokens
                        + event.cache_read_input_tokens
                        + event.cache_creation_input_tokens
                    )
                    t.stats.current_all_output_tokens = event.output_tokens
        case AgentStartMessage():
            _apply_agent_start(run, event)
        case AgentConfigMessage():
            _apply_agent_config(run, event)
        case AgentEndMessage():
            _apply_agent_end(run, event)
        case StatusPublicPublishedMessage():
            _apply_status_published(run, event)
        case _:
            # Unknown type — no state transition, but still route the
            # message below for forward-compat (a future event type
            # carrying a known scope id should still land on its node).
            pass

    # Route AFTER apply so structural events that create their scope
    # node (task_start, module_start, iteration_start, mdgroup_start)
    # find the just-created node. Routing must never break dispatch —
    # state has already been applied above; log and continue on any
    # routing-side error.
    try:
        _route_message(run, event)
    except Exception:
        from loguru import logger

        logger.exception("dispatch._route_message failed for {}", event.type)


__all__ = [
    "_ensure_task_for_id",
    "_parse_ts",
    "dispatch_event",
]
