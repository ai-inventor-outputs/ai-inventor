"""Run — top-level domain object for one pipeline execution (generic).

In v26 the tree is::

    Run > MdGroup (Loop|Seq) > [LoopIteration >] Module > tasks/slots

A Run holds a flat list of MdGroups. Each group is either a SeqMdGroup
(flat module list) or a LoopMdGroup (N LoopIterations).

Every node carries an auto-generated 12-char ``node_id`` (from
:func:`aii_lib.run.node_id.generate_short_id`). Identity is opaque —
node_ids are not constructible from outside, only returned by the
``start_*`` methods. A module's ``parent_id`` is the parent node's
auto-gen ``node_id`` (a SeqMdGroup's or LoopIteration's). For
LoopIterations, the iteration *index* lives on the
:attr:`LoopIteration.iteration` field; the parent_id stays opaque.

The pipeline call sites declare each ceremony via flat methods on Run,
capturing the returned id and flowing it into subsequent calls:

    gid = run.start_seq_group(name="gen_paper_repo")
    mid = run.start_single_module(name="gen_repo", parent_id=gid)
    ...
    run.end_module(parent_id=gid, module_id=mid)
    run.end_group(id=gid)

    loop = run.start_loop_group(name="invention_loop")
    iter_id = run.start_iteration(group_id=loop, iteration=1)
    mid = run.start_single_module(name="gen_strat", parent_id=iter_id)
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from aii_lib.timestamp import Timestamp

from .loop_iteration import LoopIteration
from .mdgroup import AnyMdGroup, LoopMdGroup, MdGroup, SeqMdGroup
from .messages import (
    AgentEndMessage,
    AgentHookMessage,
    AgentMessage,
    AgentRetryMessage,
    BaseMessage,
    GroupEndMessage,
    GroupStartMessage,
    IterationEndMessage,
    IterationStartMessage,
    MdGroupOutputMessage,
    ModuleEndMessage,
    ModuleOutputMessage,
    ModuleStartMessage,
    RunEndMessage,
    RunOutputMessage,
    RunStartMessage,
    SummarizedMessage,
    TaskEndMessage,
    TaskOutputMessage,
    TaskStartMessage,
)
from .module import AnyModule, Module
from .node import AIINode, NodeStats, NodeStatus
from .node_id import NodeID, gen_path_id, generate_short_id
from .node_index import NodeIndex
from .task import Task

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    # RunSink lives in a module that imports Run — true circular
    # import. Keep it under TYPE_CHECKING; pydantic resolves the field
    # type via model_rebuild() after both modules load. (TC004 wants it
    # moved out, but doing so re-creates the cycle and crashes at boot.)
    from .sink import RunSink  # noqa: TC004


def _now_dt() -> datetime:
    return Timestamp.now().dt


# ---------------------------------------------------------------------------
# Pluggable hook points
# ---------------------------------------------------------------------------


_DISPATCH_FN: Callable[[Run, Any], None] | None = None
_ENSURE_FOR_TASK_FN: Callable[[Run, str, str], Any] | None = None
_GROUP_CLASS_RESOLVER_FN: Callable[[str, str], type | None] | None = None


def set_dispatch(fn: Callable[[Run, Any], None]) -> None:
    """Register the typed-event dispatcher for ``Run._on``.

    The pipeline supplies this on import (see
    ``aii_pipeline.run.__init__``). Live runs route every event through
    ``_DISPATCH_FN(run, event)`` after constructing the typed event;
    replay of dict-shape events lands here too via ``parse_message``.
    """
    global _DISPATCH_FN
    _DISPATCH_FN = fn


def set_ensure_for_task(fn: Callable[[Run, str, str], Any]) -> None:
    """Register a task-id → Task resolver."""
    global _ENSURE_FOR_TASK_FN
    _ENSURE_FOR_TASK_FN = fn


def set_group_class_resolver(fn: Callable[[str, str], type | None]) -> None:
    """Register a phase-name → typed group class resolver.

    aii_pipeline supplies this on import so dispatch.py's
    ``_apply_mdgroup_start`` can construct the registered subclass
    (``GenPaperRepoGroup`` / ``HypoLoopGroup`` / …) when start_*_group
    is called dynamically (test fixtures, ad-hoc runs) without going
    through the boot scaffold. The resolver takes ``(name,
    group_type)`` where ``group_type`` is ``"seq"`` or ``"loop"`` and
    returns the typed subclass — or ``None`` to fall back to the
    generic base.
    """
    global _GROUP_CLASS_RESOLVER_FN
    _GROUP_CLASS_RESOLVER_FN = fn


def resolve_group_class(name: str, group_type: str) -> type | None:
    """Look up the typed group class for ``name`` (or ``None``)."""
    if _GROUP_CLASS_RESOLVER_FN is None:
        return None
    return _GROUP_CLASS_RESOLVER_FN(name, group_type)


_MODULE_CLASS_RESOLVER_FN: Callable[[str, str], type | None] | None = None


def set_module_class_resolver(fn: Callable[[str, str], type | None]) -> None:
    """Register a substep-name → typed module class resolver.

    aii_pipeline supplies this so ``_apply_module_start`` can
    construct the registered subclass (``GenHypoModule`` /
    ``GenStratModule`` / …) directly during clone-log replay,
    without needing the boot scaffold to pre-create instances. Takes
    ``(name, module_type)`` where ``module_type`` is ``"single"`` or
    ``"parallel"`` and returns the typed subclass — or ``None`` to
    fall back to the generic ``SingleTModule`` / ``ParallelTModule``
    base.
    """
    global _MODULE_CLASS_RESOLVER_FN
    _MODULE_CLASS_RESOLVER_FN = fn


def resolve_module_class(name: str, module_type: str) -> type | None:
    """Look up the typed module class for ``name`` (or ``None``)."""
    if _MODULE_CLASS_RESOLVER_FN is None:
        return None
    return _MODULE_CLASS_RESOLVER_FN(name, module_type)


# ---------------------------------------------------------------------------
def _depth(node: Any) -> int:
    """Tree depth of ``node`` for leaves-first traversal in :meth:`Run._finalize_orphans`.

    Tasks (no .children list) → 0. Anything with a ``children`` list
    contributes 1 + max(child_depth). Used only as a sort key, so the
    exact metric doesn't matter — just that descendants come first.
    """
    children = getattr(node, "children", None)
    if not children:
        return 0
    return 1 + max((_depth(c) for c in children), default=0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run(AIINode):
    """Top-level pipeline run.

    Composed of MdGroups (Seq or Loop). LoopMdGroups own LoopIterations,
    which in turn own Modules. SeqMdGroups own Modules directly.

    Inherits from ``AIINode`` so it carries ``node_id``,
    ``start_at`` / ``end_at``, and ``status`` (a :class:`NodeStatus`)
    uniformly with every other tree node. Carries its own
    :attr:`stats` aggregate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def gen_id(cls) -> str:
        """Generate a fresh run ID.

        Single source of truth shared by the cli's auto-generation path and
        the server's fork endpoint pre-pick — both must produce the same
        shape. Uses the uniform ``<name>_<random>`` shape every node
        follows; the ``"run"`` prefix self-documents the id and matches the
        directory's role on disk.
        """
        return f"run_{generate_short_id()}"

    # ── Constructors (factories) — sync state-prep, no I/O beyond clone replay
    # and fork seed-event write. The actual SDK / agent dispatches happen later
    # in ``run_pipeline`` after sinks are wired.

    @classmethod
    def fresh(
        cls,
        run_dir_parent: Path,
        *,
        prompt: str = "",
        new_run_id: str | None = None,
    ) -> tuple[Run, Path]:
        """Construct a fresh Run with a freshly-generated id + run_dir.

        Returns ``(run, run_dir)``. ``prompt`` is the research topic that
        ``seed_hypo`` reads as its input; left empty when seed_hypo isn't
        the first phase or the topic comes from elsewhere.

        ``new_run_id`` lets the caller pre-pick the id (matching
        :meth:`from_fork`'s signature). Used by the DBOS workflow entry
        so ``run_id == DBOS.workflow_id``: the caller stamps the id
        with ``SetWorkflowID`` before invoking the workflow, then
        passes it here so the on-disk run_dir name matches. ``None``
        (default) preserves the legacy auto-generate behaviour for
        direct callers.
        """
        run_id = new_run_id or cls.gen_id()
        run_dir = run_dir_parent / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run = cls(node_id=run_id)
        run.prompt = prompt
        return run, run_dir

    @classmethod
    def from_journal(
        cls,
        workflow_id: str,
        *,
        target_module_id: NodeID,
        prompt: str,
    ) -> Run:
        """Reconstruct an existing run from DBOS's journal + arm for resume.

        Replaces the legacy clone-log replay path with a journal-walk:
        :func:`query_events` paginates ``dbos.operation_outputs`` for
        the workflow chain (resolved via ``forked_from``), each row's
        output is decoded into a typed
        :class:`~aii_lib.run.messages.BaseMessage` and dispatched against
        a fresh :class:`Run` to rebuild the in-memory tree. Same on-tree
        result as the old :meth:`CloneSink.load` path, but the journal
        is the single source of truth so a worker can resume from any
        node that has DBOS access — no clone log required.

        Stashes ``prompt`` + ``target_module_id`` so ``run_pipeline``
        can either fire a resume turn (prompt non-empty) or
        truncate-and-restart (prompt empty) once sinks are wired.

        Sync. Requires ``init_dbos()`` to have been called by the
        caller (cli.py / server). Doesn't subscribe sinks. Doesn't
        write anything to disk.
        """
        from aii_lib.run.context import get_current_run, set_current_run
        from aii_lib.run.dispatch import dispatch_event
        from aii_lib.run.journal import (
            decode_output,
            query_events,
            resolve_workflow_chain,
        )
        from aii_lib.run.module import Module

        chain = resolve_workflow_chain(workflow_id)

        # Cursor-paginate to handle runs with more events than a single
        # query batch can return. ``query_events`` advances by
        # (started_at_epoch_ms, function_id) — a strictly increasing
        # tuple per row, so the loop drains the whole journal.
        events: list[BaseMessage] = []
        after_ts = 0
        after_fid = 0
        BATCH = 5000
        while True:
            rows = query_events(
                chain,
                after_ts_ms=after_ts,
                after_function_id=after_fid,
                limit=BATCH,
            )
            if not rows:
                break
            for _wf_id, fid, ts, raw in rows:
                after_ts = ts
                after_fid = fid
                msg = decode_output(raw)
                if msg is not None:
                    events.append(msg)
            if len(rows) < BATCH:
                break

        run = cls(node_id=workflow_id)
        prev_run = get_current_run()
        set_current_run(run)
        try:
            for event in events:
                dispatch_event(run, event)
        finally:
            set_current_run(prev_run)

        target = run.find_node(target_module_id)
        if target is None:
            raise ValueError(
                f"resume: module {target_module_id!r} not found in run {run.node_id!r}",
            )
        if not isinstance(target, Module):
            raise TypeError(
                f"resume: node {target_module_id!r} is not a Module "
                f"(got {type(target).__name__})",
            )
        run.prompt = prompt
        run._pending_resume_target = target_module_id
        run._playback_mode = "replay"
        return run

    stats: NodeStats = Field(default_factory=NodeStats)

    prompt: str = ""
    """User-supplied prompt for this run. Two meanings depending on
    constructor used:

      * :meth:`fresh` — research topic that drives ``seed_hypo``.
      * :meth:`from_resume` — user message that fans out to the resume
        target module's session-bearing children as the next user turn.

    For DBOS-native forks, ``run_pipeline_workflow`` reads the
    ``aii_fork_overrides`` row at boot and writes the override prompt
    here so the existing resume-turn / replay-execute machinery
    carries the fork forward.

    Empty string for resume in "restart-from-target" mode (no user
    message; truncate target's children + redispatch fresh)."""

    _pending_resume_target: NodeID | None = PrivateAttr(default=None)
    """Target module id for a pending resume / restart action. Set by
    :meth:`from_resume` / :meth:`from_fork` at boot; consumed by
    ``run_pipeline`` after sinks are wired (fires the resume turn or
    truncates the tree). ``None`` for fresh runs and after consumption.

    Excluded from the disk mirror — purely a boot-time signal."""

    _playback_mode: Literal["live", "replay"] = PrivateAttr(default="live")
    """Active execution mode for the v27 replay-execute architecture.

    - ``"live"`` (default) — normal forward execution; every emitter
      fires; sinks write; agent.run dispatches the SDK.
    - ``"replay"`` — re-running ``execute()`` over a tree rebuilt from
      the clone log (fork / cold-resume boot). Diagnostic emitters
      (``status_*``, ``agent_*``) early-return; structural emitters
      (``start_*``, ``end_*``, ``*_output``) flow through idempotent
      dispatch handlers (existing nodes claimed; output preserves
      first-wins). ``Agent.run`` short-circuits with a synthesized
      ``AgentResponse`` from the recorded task.

    Set ``"replay"`` by :meth:`from_resume` / :meth:`from_fork` at
    boot. Flipped back to ``"live"`` at the resume target's module
    boundary (in ``start_*_module`` when the claimed node id matches
    :attr:`_pending_resume_target`) so the target substep onward
    executes normally.

    Excluded from the disk mirror — purely a runtime concept."""

    _fork_session_ids: dict[str, str] = PrivateAttr(default_factory=dict)
    """Map of ``task_id → parent's session_id`` for the target module's
    children, populated by ``run_pipeline_workflow`` from the
    ``aii_fork_overrides`` row when the workflow body detects a fork.

    The agent backend's FORK override looks up session ids here first
    (rather than walking the run tree via ``find_task``) because for
    DBOS-native forks the fork's tree starts empty — only the cached
    pre-fork operation outputs are inherited, not the in-process Run
    aggregate state. Empty dict for fresh runs and resumes."""

    _emit_counter: dict[tuple[NodeID, str], int] = PrivateAttr(default_factory=dict)
    """Per-``(parent_id, name)`` monotonic counter that gives each
    structural emit (``start_seq_group`` / ``start_loop_group`` /
    ``start_iteration`` / ``start_*_module`` / ``start_task``) the
    sibling-position index used to compute its path.

    Increments on every emit, reset per Run instance. Replay re-walks
    in the same emit order as the original execution → same counter
    values → same paths → same path-derived UUIDs → existing slots
    resolve via :meth:`find_node`. Replaces the v27-stage-3 slot-claim
    cursor (``_resume_claimed_tasks`` + ``_find_unclaimed_resume_slot``)
    with a stateless deterministic-id lookup."""

    # ── content ───────────────────────────────────────────────────────────
    # Forward-reference annotation: ``"AnyMdGroup"`` resolves at
    # ``model_rebuild`` time against module globals. aii_pipeline rebinds
    # the symbol to a wider union (including phase subclasses) at process
    # boot via :func:`aii_pipeline.run.typed_union.bind_pipeline_typed_unions`,
    # then forces ``Run.model_rebuild`` so the rebind takes effect for
    # ``model_validate``-driven seed hydration. Without the rebind, base
    # ``SeqMdGroup | LoopMdGroup`` is what pydantic sees — fine for tests
    # that don't load pipeline classes.
    children: list[AnyMdGroup] = Field(default_factory=list)
    """Top-level MdGroups (Seq or Loop). Uniform ``children`` field name
    across every parent in the tree (Run/MdGroup/LoopIteration/Module)."""

    # NB: ``Run.artifacts`` was retired in favour of a derived projection
    # on ``AppRun.published`` (computed by the to_app mapper from the
    # latest ``status_published`` event when the gen_paper_repo group is
    # done). Cascade-truncate doesn't have to clear an accumulator any
    # more — the canonical source of truth is ``Run.events``.
    #
    # Reviews live in Django (``RunReview`` table), NOT on the Run
    # domain object. The dashboard reads them off the DB directly when
    # rendering the review surface; the pipeline never reads or emits
    # them.

    # ── per-node event log: inherited from AIINode ────────────────────────
    # ``messages`` holds run-scoped events (``run_start`` / ``run_end`` and
    # any pipeline-level diagnostics that don't fit a child node). Every
    # other node (MdGroup, LoopIteration, Module, Task) carries its own
    # ``messages`` list via the same inherited field; consumers that want
    # a global timeline walk the tree themselves and merge-sort by ``ts``.

    # ── Combined index (private; not serialized) ─────────────────────────
    _node_index: NodeIndex = PrivateAttr(default_factory=NodeIndex)
    """Two indices on the run aggregate:

      * ``nodeid_to_node`` — flat ``node_id → AIINode`` lookup over
        every node in the tree. Populated by :meth:`_index`.
      * ``id_to_ancestors`` — per-message ancestor map maintained
        across the tree.

    See :class:`aii_lib.run.node_index.NodeIndex`."""

    _sinks: list[RunSink] = PrivateAttr(default_factory=list)
    """Registered :class:`RunSink` instances — write-side subscribers.

    Live runs add ``CloneSink`` / ``ConsoleRunSink`` / ``OTelRunSink``
    / sequenced-clone variants; replay leaves this empty so the same
    ``_on`` code path produces no side-effects when reconstructing
    state from disk.

    Stored as a private attr (with a public ``sinks`` property view)
    because sinks wrap asyncio queues, FastAPI hub handles, and other
    non-picklable runtime state — they're excluded from the disk mirror.
    The state-only mirror is enough for resume; on ``load_run_clone``
    the resume path re-subscribes whatever sinks the new pipeline needs.
    """

    _summary_buffer: Any | None = PrivateAttr(default=None)
    """Optional :class:`SummaryBuffer` — pre-record LLM-summary hop.

    When set (via :meth:`enable_summary_buffer`), eligible messages
    take a detour through the buffer before reaching :meth:`_record`:
    the buffer holds them until the LLM-generated short summary is
    attached, then forwards in submission order. Replay (no executor
    wired) leaves this ``None`` so events flow straight through.
    """

    _summary_executor: Any | None = PrivateAttr(default=None)
    """Backing :class:`ThreadPoolExecutor` for the summary buffer.

    Tracked separately so :meth:`close_summary_buffer` can shut it
    down on pipeline teardown."""

    @property
    def playback_mode(self) -> Literal["live", "replay"]:
        """Public read-accessor for :attr:`_playback_mode`.

        ``"live"`` for normal forward execution and for replay-execute
        once the resume target's module boundary is reached. ``"replay"``
        while re-running ``execute()`` over a tree rebuilt from the
        clone log (fork / cold-resume boot phase). Consumers introspect
        this to gate their own side effects (e.g. external HTTP calls,
        wall-clock-dependent retries) when they're outside Run's
        emit-side gating.
        """
        return self._playback_mode

    def _replay_skip(self) -> bool:
        """True when diagnostic emitters should early-return.

        Used by every status_* / agent_* emitter to no-op during the
        replay-execute boot phase — the events are already on disk
        from the original run, so re-emitting would duplicate them
        in per-node ``messages`` lists. Structural emitters
        (``start_*`` / ``end_*`` / ``*_output``) DO NOT use this gate;
        they flow through idempotent dispatch handlers instead.
        """
        return self._playback_mode == "replay"

    @property
    def sinks(self) -> list[RunSink]:
        """The registered :class:`RunSink` instances on this Run.

        Read-only view; mutate via :meth:`subscribe` (and the closure
        returned from it for unsubscribe).
        """
        return self._sinks

    @property
    def node_index(self) -> NodeIndex:
        """Per-message ancestor map keyed by message ``node_id``.

        See :class:`aii_lib.run.node_index.NodeIndex`.
        """
        return self._node_index

    def model_post_init(self, _ctx: Any, /) -> None:
        """Index the run so find_node resolves it."""
        # Self-index so find_node(run.node_id) returns the run.
        self._index(self)

    def __getstate__(self) -> dict:
        """Exclude non-picklable sink state.

        Exclude ``_sinks`` so any pickle round-trip doesn't choke on
        non-picklable runtime state (asyncio queues, file handles).
        The state-only snapshot is enough for resume; sinks are
        re-attached by the new pipeline boot path.
        """
        state = super().__getstate__()
        priv = state.get("__pydantic_private__")
        if isinstance(priv, dict) and "_sinks" in priv:
            priv = {**priv, "_sinks": []}
            state = {**state, "__pydantic_private__": priv}
        return state

    # ── O(1) node index ───────────────────────────────────────────────────

    def _index(self, node: AIINode) -> None:
        """Add node to the flat index.

        Idempotent — re-indexing replaces the prior entry.
        """
        self._node_index.index_node(node)

    def attach(self, child: AIINode, *, parent: AIINode | None = None) -> None:
        """Attach child to parent and index it.

        Appends ``child`` to ``parent.children`` (default: this Run) and
        registers it in the run-level node index. Public scaffold-time API
        for bulk tree assembly — replaces ``parent.children.append(child);
        run._index(child)`` pairs at the boundary of ``aii_lib/run/`` so
        external callers don't reach into the private index.
        """
        target = parent if parent is not None else self
        target.children.append(child)
        self._index(child)

    def find_node(self, node_id: str) -> AIINode | None:
        """Look up any node in this Run by its ``node_id``."""
        if not node_id:
            return None
        return self._node_index.nodeid_to_node.get(node_id)

    # ── derived properties ────────────────────────────────────────────────

    @property
    def all_tasks(self) -> list[Task]:
        """Flatten across the full tree → tasks. O(n) walk."""
        out: list[Task] = []
        for g in self.children:
            if isinstance(g, SeqMdGroup):
                for m in g.children:
                    out.extend(m.tasks)
            elif isinstance(g, LoopMdGroup):
                for it in g.children:
                    for m in it.children:
                        out.extend(m.tasks)
        return out

    # ── lookups (thin typed wrappers over the flat node index) ──────────

    def find_group(self, group_id: str) -> MdGroup | None:
        """Look up a group by ID."""
        n = self._node_index.nodeid_to_node.get(group_id)
        return n if isinstance(n, MdGroup) else None

    def find_group_by_name(self, name: str) -> MdGroup | None:
        """Locate a group by its human-readable ``name``.

        Used by callers that don't have the auto-generated node_id but
        know the canonical phase label (``"invention_loop"`` / ``"gen_hypo"``
        / etc.). Returns the first match or None. Names are unique per
        phase so collisions don't happen in practice.
        """
        for child in self.children:
            if isinstance(child, MdGroup) and child.name == name:
                return child
        return None

    def find_iteration(self, group_id: str, iteration: int) -> LoopIteration | None:
        """Look up an iteration by group ID and iteration number."""
        g = self.find_group(group_id)
        if not isinstance(g, LoopMdGroup):
            return None
        return g.find_iteration(iteration)

    def find_module(self, module_id: str) -> AnyModule | None:  # type: ignore[valid-type]
        """Locate a module by id (ids are unique tree-wide)."""
        n = self._node_index.nodeid_to_node.get(module_id)
        # Module is abstract — runtime instances are always SingleTModule
        # or ParallelTModule (the AnyModule union members), so the
        # isinstance(Module) narrowing is correct even though ty can't
        # see through the abstract base to the concrete subclasses.
        return n if isinstance(n, Module) else None  # ty: ignore[invalid-return-type]

    def find_task(self, task_id: str) -> Task | None:
        """Look up a task by ID."""
        n = self.find_node(task_id)
        return n if isinstance(n, Task) else None

    def find_module_for_task(self, task_id: str) -> AnyModule | None:  # type: ignore[valid-type]
        """Find the module owning a task."""
        t = self.find_task(task_id)
        if t is None or t.parent_id is None:
            return None
        n = self._node_index.nodeid_to_node.get(t.parent_id)
        return n if isinstance(n, Module) else None  # ty: ignore[invalid-return-type]

    def task_sequence(self, task_id: str) -> int | None:
        """Return the task's 0-based index within its parent module.

        The position in :attr:`Module.children` is the canonical sequence
        — sequencing sinks (``TaskSequencer`` consumers: console +
        sequenced clone) read it via this method instead of pulling
        an extra ``sequence`` field off the ``task_start`` event.

        Returns ``None`` when the task isn't in the tree, has no parent
        module, or isn't actually one of that module's children.
        """
        m = self.find_module_for_task(task_id)
        if m is None:
            return None
        return m.task_sequence(task_id)

    # ── Resume / restart navigation ──────────────────────────────────────

    def remove_children(self, target: AnyModule) -> None:  # type: ignore[valid-type]
        """Remove target's children and everything after it.

        Removes ``target``'s children + everything that ran after ``target``
        in execution order. ``target`` itself stays (empty + IN_PROGRESS),
        ready for a fresh forward-pipeline pass to redispatch. Used by
        no-prompt resume ("restart from this module"). After the call:

          * ``target.children`` is empty (target's own subtree dropped)
          * substeps after ``target`` in its parent (iter or seq-group)
            are removed
          * if ``target``'s parent is a LoopIteration, later iterations
            in the LoopMdGroup are removed
          * later phases (top-level groups under Run.children) are
            removed
          * ``target`` + its ancestors are flipped IN_PROGRESS,
            ``end_at`` cleared
        """
        target.truncate_self()
        self._remove_post_target_subtrees(target)
        self._flip_ancestors_in_progress(target)

    def _remove_post_target_subtrees(self, target: AnyModule) -> None:  # type: ignore[valid-type]
        """Drop substeps after target and later iterations.

        Helper for :meth:`remove_children`.
        """
        parent = self.find_node(target.parent_id) if target.parent_id else None
        if parent is None:
            return
        # Substeps after target in its parent (iter or seq-group)
        if hasattr(parent, "children") and target in parent.children:
            idx = parent.children.index(target)
            for d in parent.children[idx + 1 :]:
                self._unindex_subtree(d)
            parent.children = parent.children[: idx + 1]
        # Later iters in same LoopMdGroup
        if isinstance(parent, LoopIteration):
            loop_group = self.find_node(parent.parent_id) if parent.parent_id else None
            if isinstance(loop_group, LoopMdGroup) and parent in loop_group.children:
                iter_idx = loop_group.children.index(parent)
                for d in loop_group.children[iter_idx + 1 :]:
                    self._unindex_subtree(d)
                loop_group.children = loop_group.children[: iter_idx + 1]
        # Later phases under Run
        phase = self._top_phase_of(target)
        if phase is not None and phase in self.children:
            phase_idx = self.children.index(phase)
            for d in self.children[phase_idx + 1 :]:
                self._unindex_subtree(d)
            self.children = self.children[: phase_idx + 1]

    def _flip_ancestors_in_progress(self, target: AnyModule) -> None:  # type: ignore[valid-type]
        """Set ancestors to IN_PROGRESS and clear end_at.

        Helper for :meth:`remove_children` so the FE shows an active run
        after the removal. Dispatcher self-flips on subsequent ``*_start``
        events too, but flipping eagerly here keeps the snapshot consistent.
        """
        target.status = NodeStatus.IN_PROGRESS
        target.end_at = None
        parent_id = target.parent_id
        while parent_id:
            node = self.find_node(parent_id)
            if node is None:
                break
            node.status = NodeStatus.IN_PROGRESS
            node.end_at = None
            parent_id = getattr(node, "parent_id", None)
        self.status = NodeStatus.IN_PROGRESS
        self.end_at = None

    def _top_phase_of(self, target: AnyModule) -> MdGroup | None:  # type: ignore[valid-type]
        """Find the top-level phase containing a module."""
        node: Any = target
        while node is not None:
            parent_id = getattr(node, "parent_id", None)
            if parent_id is None or parent_id == self.node_id:
                # We've walked up to the run root or hit a node without
                # a parent — the current node is the top-level child.
                if isinstance(node, MdGroup):
                    return node
                return None
            parent = self.find_node(parent_id)
            if parent is None:
                return None
            node = parent
        return None

    def _unindex_subtree(self, node: Any) -> None:
        """Remove subtree nodes from the index."""
        stack = [node]
        while stack:
            n = stack.pop()
            self._node_index.nodeid_to_node.pop(getattr(n, "node_id", ""), None)
            stack.extend(getattr(n, "children", []) or [])

    # ── retry context (last-N message lines for prompt rebuild) ──────────

    def get_recent_message_text(
        self,
        *,
        task_id: str | None = None,
        n: int = 20,
        types: set[str] | None = None,
        char_cap: int = 300,
    ) -> list[str]:
        """Format the last ``n`` typed events as retry-context lines.

        Each entry is ``"[type] tool: text…"`` (with ``ERROR`` suffix
        when the event's ``is_error`` flag is set). Reads directly off
        the in-memory event log (every Task carries its own
        ``messages`` list), so there's no disk hop.

        Args:
            task_id:  Restrict to one task's event log. When ``None``
                      walks every task in the tree.
            n:        Cap on the number of lines returned. The most
                      recent ``n`` are kept.
            types:    Wire-string filter set. Default is the set the
                      retry rebuilder cares about (tool calls/results,
                      public warnings/errors, think events).
            char_cap: Per-line text truncation.
        """
        if types is None:
            # Include the LLM's output (``agent_response`` /
            # ``llm_response``) and the user turn (``agent_user_prompt``
            # / ``llm_user_prompt``) so the retry agent sees both what
            # it was asked to do AND what it produced — the most
            # relevant signal for "this is what failed the schema."
            # Skipped: ``agent_message_delta`` (token stream / cost
            # ticks, no semantic content), ``agent_summary`` (cost
            # accounting), ``agent_config`` / ``agent_start`` /
            # ``agent_end`` (lifecycle). Tool calls / results stay in
            # so the retry agent inherits the prior tool-use trail.
            types = {
                "agent_tool_call",
                "agent_tool_result",
                "llm_tool_call",
                "llm_tool_result",
                "status_public_warning",
                "status_public_error",
                "agent_think",
                "llm_think",
                "agent_response",
                "llm_response",
                "agent_user_prompt",
                "llm_user_prompt",
            }

        if task_id:
            task = self.find_task(task_id)
            events = list(task.messages) if task is not None else []
        else:
            events = [ev for t in self.all_tasks for ev in t.messages]

        out: list[str] = []
        for ev in events:
            msg_type = getattr(ev, "type", "") or ""
            if msg_type not in types:
                continue
            text = (getattr(ev, "text", "") or "")[:char_cap]
            tool = getattr(ev, "tool", "") or ""
            is_err = getattr(ev, "is_error", False)
            label = f"[{msg_type}] {tool}" if tool else f"[{msg_type}]"
            if is_err:
                label += " ERROR"
            out.append(f"{label}: {text}")

        return out[-n:] if len(out) > n else out

    # ── fork helpers ──────────────────────────────────────────────────────

    # ── domain mutations (run-level events) ──────────────────────────────

    def _apply_start(self, *, ts: datetime | None) -> None:
        if self.start_at is None and ts is not None:
            self.start_at = ts
        self.status = NodeStatus.IN_PROGRESS

    def _apply_end(
        self,
        *,
        status: NodeStatus,
        ts: datetime | None,
    ) -> None:
        if ts is not None:
            self.end_at = ts
        self.status = status

    # ── pluggable structure-ensure hook ──────────────────────────────────

    def ensure_for_task(self, task_id: str, name: str = "") -> Task | None:
        """Ensure a task exists in a module (resume/replay hook).

        Called from :meth:`end_task` so re-emitted ``task_end`` events
        on a tree that doesn't yet have the task get the task created
        before the end attaches. The resolver is registered by
        ``aii_pipeline.run.__init__`` via :func:`set_ensure_for_task`.
        """
        if _ENSURE_FOR_TASK_FN is None:
            raise RuntimeError("Run.ensure_for_task requires set_ensure_for_task(...).")
        return _ENSURE_FOR_TASK_FN(self, task_id, name)

    # ── v26 single-write path: _on / _dispatch / _notify ─────────────────

    def subscribe_sink(self, sink: RunSink) -> Callable[[], None]:
        """Register a :class:`RunSink` and return an unsubscribe closure.

        Sinks' ``flush(event)`` is called once per ``_on(event)``
        invocation, in registration order. Each call is wrapped in
        try/except so a buggy sink doesn't block the others. The
        returned closure removes the sink when called; it's safe to
        invoke twice (idempotent).
        """
        self._sinks.append(sink)

        def unsub() -> None:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass

        return unsub

    def _dispatch(self, event: BaseMessage) -> None:
        """Apply ``event``'s state mutation to ``self``.

        Routes to the pipeline-supplied dispatcher (registered via
        ``set_dispatch``). When no dispatcher is registered (very early
        boot, before ``aii_pipeline.run`` is imported), the event is
        recorded but no state change happens — the dispatch is purely
        structural and replay-safe to repeat once the dispatcher is
        wired.

        Dispatch errors are NOT swallowed: a domain-mutation failure is
        a real bug that must surface. The previous best-effort swallow
        masked drift between dispatcher and domain method names.
        """
        if _DISPATCH_FN is None:
            return
        _DISPATCH_FN(self, event)

    def _notify(self, event: BaseMessage) -> None:
        """Call every registered sink's ``flush(event)``.

        Per-sink try/except: one bad sink must not block the rest
        or cause a partial state where some sinks saw the event and
        others didn't.

        Replay-mode sink gating: when ``_playback_mode == "replay"``
        every sink whose :attr:`RunSink.replay_policy` is
        :attr:`~aii_lib.run.sink.ReplayPolicy.SKIP` (the only value
        today) is silenced — the on-disk projection files are already
        complete, so re-emitting would double-write.
        """
        from .sink import ReplayPolicy

        in_replay = self._playback_mode == "replay"
        # Snapshot the list so sinks that unsubscribe themselves don't
        # mutate the iterable mid-walk.
        for sink in list(self._sinks):
            if in_replay and sink.replay_policy == ReplayPolicy.SKIP:
                continue
            try:
                sink.flush(event)
            except Exception:
                # Swallow — sinks are best-effort. A persistent failure
                # surfaces in their own logs.
                continue

    def _on(self, event: BaseMessage) -> None:
        """The single write path — every event passes through here.

        When a :class:`SummaryBuffer` is enabled, ALL events are routed
        through ``buf.submit`` so submission order is preserved at the
        sink boundary. The buffer holds each event in a FIFO queue and
        only fires an LLM summary when ``is_eligible`` says so —
        non-eligible events still enqueue but mark themselves ready
        immediately, so they drain as soon as they reach the head
        (after any earlier eligible event's summary lands).

        Without a buffer, events go straight to :meth:`_record`.
        """
        buf = self._summary_buffer
        if buf is not None:
            buf.submit(event, on_ready=self._record)
            return
        self._record(event)

    def _record(self, event: BaseMessage) -> None:
        """Inner write path: dispatch → journal mirror → notify.

        Order matters:
          1. ``_dispatch`` mutates Run state AND routes the event onto
             its scope-owning node's ``messages`` list (via
             ``aii_lib.run.dispatch._route_message``) so
             subscribers see the post-event domain.
          2. Mirror to ``dbos.operation_outputs`` so the
             :class:`~aii_lib.run.journal.JournalTailer`-driven
             consumers (clone, sequenced clone, console, health, otel,
             title) see this Run-bus event alongside direct ``emit.X``
             writes. Replay-skipped — the on-disk projection files are
             already complete, re-emitting would double-write. Errors
             are logged at debug level and dropped: outside a DBOS
             workflow context the step decorator raises, and we don't
             want test/manual-script paths spamming the log.
          3. ``_notify`` fans out to every :class:`RunSink` still on
             the Run-bus (today: only the in-process BufferSink in
             ``worker/server.py``).

        Called either directly from :meth:`_on` (passthrough events)
        or from the summary buffer's drain (after summary attached).
        """
        self._dispatch(event)
        if self._playback_mode != "replay":
            try:
                from .journal import journal_event_step

                journal_event_step(event.model_dump(mode="json"))
            except Exception:
                from loguru import logger

                logger.opt(exception=True).debug(
                    "Run._record: journal_event_step raised — "
                    "expected only outside a DBOS workflow context"
                )
        self._notify(event)

    # ── LLM summary buffer (opt-in) ──────────────────────────────────────

    def enable_summary_buffer(
        self,
        *,
        min_chars: int = 30,
        max_chars: int = 50,
        max_concurrent: int = 10,
    ) -> None:
        """Enable the pre-record LLM-summary buffer for this Run.

        Eligible messages (``agent_*`` / ``llm_*`` / specific
        ``status_public_*`` types) are held in a FIFO queue, an LLM
        summary is generated in a thread executor, and the message is
        released to :meth:`_record` once the summary is attached.

        If every fallback tier in the summarizer chain returns empty
        (or the 20-second drain safety valve fires), the message
        drains *without* a summary and a ``status_public_warning`` is
        emitted so the failure is visible.

        Pipeline.py calls this once per live run; replay/test paths
        leave the buffer disabled so events flow straight through.
        """
        from concurrent.futures import ThreadPoolExecutor

        from .llm_summary import SummaryBuffer, SummaryBufferConfig

        executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="llm_summary",
        )
        config = SummaryBufferConfig(
            min_chars=min_chars,
            max_chars=max_chars,
            max_concurrent=max_concurrent,
        )
        self._summary_buffer = SummaryBuffer(
            executor=executor,
            config=config,
            on_summary_failed=self._emit_summary_warning,
        )
        self._summary_executor = executor

    def close_summary_buffer(self) -> None:
        """Drain pending summaries and shut the executor down.

        Pipeline.py calls this on run teardown so the executor's
        worker threads exit cleanly. Idempotent: a no-op when no
        buffer was enabled.
        """
        if self._summary_buffer is not None:
            self._summary_buffer.flush(timeout=5.0)
            self._summary_buffer = None
        if self._summary_executor is not None:
            self._summary_executor.shutdown(wait=False, cancel_futures=True)
            self._summary_executor = None

    def _emit_summary_warning(self, original: BaseMessage) -> None:
        """Surface that the LLM-summary chain failed for ``original``.

        The warning carries a non-empty ``summary`` of its own so it
        is NOT itself eligible for the buffer (no recursive re-entry).
        ``status_public_warning`` is a plain :class:`BaseMessage`
        (not :class:`SummarizedMessage`) so it would skip the buffer
        regardless, but we use :class:`SummarizedMessage` here with
        a pre-filled summary for belt-and-braces safety in case the
        eligibility set ever expands to include warnings.
        """
        original_type = getattr(original, "type", "?")
        original_node = getattr(original, "node_id", "") or ""
        warn = SummarizedMessage(
            type="status_public_warning",
            parent_id=getattr(original, "parent_id", None) or self.node_id,
            text=(
                f"LLM summary unavailable for {original_type}"
                + (f" ({original_node})" if original_node else "")
            ),
            summary="LLM summary unavailable",
        )
        # Bypass _on: a SummarizedMessage with a pre-filled summary
        # is not eligible, so _on would just call _record anyway —
        # but going direct removes any future ambiguity if the
        # eligibility predicate changes.
        self._record(warn)

    # ── run-level operations ──────────────────────────────────────────────

    def start(self, **extra: Any) -> None:
        """Emit ``run_start`` — flips the Run from PENDING to IN_PROGRESS."""
        ts = _now_dt()
        # _on dispatches → applies _start; we don't pre-call _start any
        # more so the v26 single-write path is the only mutator.
        ev = RunStartMessage(
            ts=ts,
            run_id=self.node_id,
            parent_id=self.node_id,
            text=self.node_id,
            **extra,
        )
        self._on(ev)

    def _finalize_orphans(self, *, orphan_status: str = "stopped") -> None:
        """Walk every IN_PROGRESS descendant and emit a typed end-event, leaves first.

        Idempotent: nodes already at a terminal status
        are skipped (the IN_PROGRESS filter excludes them up front).

        Used by both the /stop walk (:meth:`mark_stopped`) and ``Run.end``
        as defense-in-depth — a clean exit calls each phase's own
        ``end_group`` first, so this is a no-op then. A crash or early-
        return that bypasses per-phase terminals would otherwise leave
        children stuck at IN_PROGRESS in ``node_status.jsonl`` forever.

        ``orphan_status`` defaults to ``"stopped"`` so an orphan never
        gets tagged ``"done"`` (which would lie about completion). The
        Run itself takes the caller's chosen status separately.

        **Resilience**: each per-node ``end_*`` is wrapped so a single
        bad node (missing parent in the index, dispatcher rejection,
        sink hiccup) doesn't abort the rest of the walk.
        """
        from aii_lib.run.loop_iteration import LoopIteration
        from aii_lib.run.mdgroup import LoopMdGroup, MdGroup
        from aii_lib.run.module import Module
        from aii_lib.run.task import Task

        # Walk leaves-first so parents see their children already terminal.
        in_progress = [
            n
            for n in self._node_index.nodeid_to_node.values()
            if getattr(n, "status", None) == NodeStatus.IN_PROGRESS and n is not self
        ]
        in_progress.sort(key=lambda n: -_depth(n))

        for node in in_progress:
            try:
                if isinstance(node, Task):
                    self.end_task(node.node_id, status=orphan_status)
                elif isinstance(node, Module):
                    # parent_id on Module is the iteration / seq-group node_id.
                    self.end_module(
                        parent_id=node.parent_id,
                        module_id=node.node_id,
                        status=orphan_status,
                    )
                elif isinstance(node, LoopIteration):
                    # Iterations live under LoopMdGroup; recover the 1-based
                    # index from the parent's children list.
                    parent = self._node_index.nodeid_to_node.get(node.parent_id)
                    if isinstance(parent, LoopMdGroup):
                        iter_n = parent.iteration_number(node)
                        if iter_n is not None:
                            self.end_iteration(
                                group_id=parent.node_id,
                                iteration=iter_n,
                                status=orphan_status,
                            )
                elif isinstance(node, MdGroup):
                    self.end_group(id=node.node_id, status=orphan_status)
            except Exception as e:
                # One node failure must not abort the rest of the walk —
                # parents above this node still need their terminal
                # events. Surface the failure as a warning so it's
                # visible in the run feed without stopping the walk.
                try:
                    self.status_public_warning(
                        f"_finalize_orphans: end_* failed for "
                        f"{type(node).__name__} {node.node_id!r}: {e}"
                    )
                except Exception:
                    pass

    def end(self, *, status: str = "completed", **extra: Any) -> None:
        """Emit ``run_end`` and seal the Run at the given terminal status."""
        # Defense-in-depth: finalize any IN_PROGRESS descendants before
        # emitting run_end. On a clean exit each phase already called
        # its own ``end_group``, so this is a no-op. On a crash or
        # early-return path (cli.py's ``except Exception`` handlers,
        # ``return None`` from a phase that bypasses ``end_group``) it
        # prevents children from sticking at IN_PROGRESS in
        # ``node_status.jsonl`` after the run finishes. Wrapped so
        # ``run_end`` is always emitted even if the walk fails — the FE
        # gates "is the run live?" on this event.
        try:
            self._finalize_orphans(orphan_status="stopped")
        except Exception:
            pass

        ts = _now_dt()
        ev = RunEndMessage(
            ts=ts,
            run_id=self.node_id,
            parent_id=self.node_id,
            status=status,
            text=f"Pipeline {status}",
            extras={"status": status},
            **extra,
        )
        self._on(ev)

    # ── group-level operations ────────────────────────────────────────────

    def start_seq_group(self, *, name: str, **extra: Any) -> str:
        """Construct a SeqMdGroup and emit ``mdgroup_start``.

        Returns the node_id.

        ``name`` is the human-readable canonical label (e.g.
        ``"gen_paper_repo"``). If the boot scaffold already created a
        group with this name, its existing node_id is reused so caller-
        held ids stay consistent with the indexed tree; otherwise a new
        id is auto-generated.

        """
        path, gid = self._emit_path_and_id(parent_id=self.node_id, name=name)
        ev = GroupStartMessage(
            group_type="seq",
            group_id=gid,
            parent_id=gid,
            name=name,
            group=gid,
            text=name,
            path=path,
            **extra,
        )
        self._on(ev)
        return gid

    def start_loop_group(self, *, name: str, **extra: Any) -> str:
        """Construct a LoopMdGroup and emit ``mdgroup_start``.

        Returns the node_id. Reuses the scaffold's node_id when one exists;
        see :meth:`start_seq_group` for naming semantics.

        """
        path, gid = self._emit_path_and_id(parent_id=self.node_id, name=name)
        ev = GroupStartMessage(
            group_type="loop",
            group_id=gid,
            parent_id=gid,
            name=name,
            group=gid,
            text=name,
            path=path,
            **extra,
        )
        self._on(ev)
        return gid

    def end_group(self, *, id: str, status: str = "done", **extra: Any) -> None:  # noqa: A002 — keyword-only public API; matches GroupEndMessage.group_id
        """Emit ``mdgroup_end``.

        Without ``status``, the dispatcher rolls the group up from children;
        ``status="stopped"`` overrides to STOPPED (used by :meth:`mark_stopped`).
        """
        ev = GroupEndMessage(
            group_id=id,
            parent_id=id,
            group=id,
            text=id,
            status=status,
            **extra,
        )
        self._on(ev)

    # ── iteration-level operations (LoopMdGroup only) ─────────────────────

    def start_iteration(
        self,
        *,
        group_id: str,
        iteration: int,
        **extra: Any,
    ) -> str:
        """Create a LoopIteration under a LoopMdGroup and emit ``iteration_start``.

        Returns the iteration's node_id.

        Callers must capture the returned id and pass it as
        ``parent_id`` to subsequent ``start_*_module`` / ``end_*``
        calls. When the boot scaffold has already pre-created this
        iteration, its existing node_id is reused so the caller-held
        id matches the indexed tree; otherwise a new id is auto-
        generated.

        """
        iter_name = f"iter{iteration}"
        path, iid = self._emit_path_and_id(parent_id=group_id, name=iter_name)
        ts = _now_dt()
        ev = IterationStartMessage(
            ts=ts,
            group_id=group_id,
            iteration=iteration,
            iteration_id=iid,
            parent_id=iid,
            text=iter_name,
            path=path,
            **extra,
        )
        self._on(ev)
        return iid

    def end_iteration(
        self,
        *,
        group_id: str,
        iteration: int,
        status: str = "done",
        **extra: Any,
    ) -> None:
        """Emit ``iteration_end``.

        ``status="stopped"`` overrides the rollup (used by :meth:`mark_stopped`);
        default rolls up from the iteration's modules.
        """
        ts = _now_dt()
        # Recover the iteration id from the group so the message owner
        # (the iteration itself) is set explicitly. mark_stopped + the
        # phase loops both pass group_id+iteration; iteration_id isn't
        # otherwise on the wire by name.
        iter_node = self.find_iteration(group_id, iteration)
        iter_id = iter_node.node_id if iter_node is not None else group_id
        ev = IterationEndMessage(
            ts=ts,
            group_id=group_id,
            iteration=iteration,
            parent_id=iter_id,
            text=f"iter{iteration}",
            status=status,
            **extra,
        )
        self._on(ev)

    # ── module-level operations ───────────────────────────────────────────

    def _attach_module(self, *, parent_id: str, module: AnyModule) -> None:  # type: ignore[valid-type]
        """Attach a module to its parent (SeqMdGroup or LoopIteration).

        ``parent_id`` is the auto-generated node_id of the parent —
        looked up directly via the flat node index. The parent (group
        or iteration) MUST already exist — its ``start_*`` event has
        to fire before any of its modules' ``module_start`` events.
        Raises if the lookup fails: that's a sequencing bug, not
        something to paper over.
        """
        parent = self._node_index.nodeid_to_node.get(parent_id)
        if isinstance(parent, SeqMdGroup):
            parent._apply_module_added(module)
            self._index(module)
            return
        if isinstance(parent, LoopIteration):
            parent._apply_module_added(module)
            self._index(module)
            return
        raise ValueError(
            f"_attach_module: parent_id {parent_id!r} does not resolve to "
            f"a SeqMdGroup or LoopIteration "
            f"(module={module.node_id!r}, parent={type(parent).__name__})"
        )

    def _next_emit_index(self, parent_id: NodeID, name: str) -> int:
        """Return the next sibling-position index for ``(parent_id, name)``.

        Monotonic per Run instance — increments on every call. Used by
        structural emitters to compute paths so live emit, resume
        re-walk, and fork re-walk all generate the same path-derived
        node_ids in the same emit order. See :attr:`_emit_counter`.
        """
        key = (parent_id, name)
        idx = self._emit_counter.get(key, 0)
        self._emit_counter[key] = idx + 1
        return idx

    def _compute_emit_path(self, *, parent_id: NodeID, name: str) -> str:
        """Compute the structural path a new ``name`` child would occupy.

        Advances :attr:`_emit_counter` by one. Path shape:
        ``"{parent.path}/{name}[{idx}]"``. Empty parent path (root) keeps
        the leading ``/``. Caller hashes the result through
        :func:`gen_path_id` to derive the deterministic id and stamps both
        ``path`` and ``node_id`` on the constructed node.
        """
        parent = self._node_index.nodeid_to_node.get(parent_id)
        parent_path = "" if (parent is None or parent is self) else parent.path
        idx = self._next_emit_index(parent_id, name)
        return f"{parent_path}/{name}[{idx}]"

    def _emit_path_and_id(self, *, parent_id: NodeID, name: str) -> tuple[str, NodeID]:
        """Convenience: compute path + derived id together.

        Returns ``(path, node_id)``. Live emitters use this once per
        ``start_*`` call to get both fields for the new (or claimed)
        node. The id is deterministic from ``path`` so a subsequent
        ``find_node(node_id)`` resolves any pre-existing slot at the
        same structural address — covers the resume-warm-tree and
        scaffold-prebuilt cases without separate special-case lookups.
        """
        path = self._compute_emit_path(parent_id=parent_id, name=name)
        return path, gen_path_id(name, path)

    def _maybe_flip_to_live_at_resume_target(self, claimed_module_id: str) -> None:
        """Stage 3 mode-flip helper: replay → live at the resume target boundary.

        Replay-execute re-runs ``execute()`` over a tree pre-built from
        the clone log. When a structural emitter claims an existing
        module whose id matches :attr:`_pending_resume_target`, we've
        reached the resume boundary — flip ``_playback_mode`` to
        ``"live"`` so the target's substep dispatches normally (agent
        FORK turn fires; sinks resume; status emitters un-gate).
        Idempotent: subsequent calls with already-cleared marker no-op.
        """
        if (
            self._playback_mode == "replay"
            and self._pending_resume_target is not None
            and claimed_module_id == self._pending_resume_target
        ):
            self._playback_mode = "live"

    def start_single_module(
        self,
        *,
        name: str,
        parent_id: str,
        **extra: Any,
    ) -> str:
        """Construct a SingleTModule and emit ``module_start`` through ``_on``.

        Returns the module's node_id.

        Callers must capture the returned id and pass it as
        ``module_id`` to subsequent ``end_module`` / ``module_output``
        calls. Reuses the scaffold's id if a same-named module already
        exists under ``parent_id``; otherwise auto-generates.

        """
        path, mid = self._emit_path_and_id(parent_id=parent_id, name=name)
        # Stage 3: replay→live mode flip at the resume target boundary,
        # BEFORE emitting so the start event lands in live mode (so any
        # sinks gated on replay see the right transition).
        self._maybe_flip_to_live_at_resume_target(mid)
        ev = ModuleStartMessage(
            module_type="single",
            name=name,
            module_id=mid,
            parent_id=mid,
            attach_under_id=parent_id,
            module=mid,
            text=name,
            path=path,
            **extra,
        )
        self._on(ev)
        return mid

    def start_parallel_module(
        self,
        *,
        name: str,
        parent_id: str,
        **extra: Any,
    ) -> str:
        """Construct a ParallelTModule and emit ``module_start`` through ``_on``.

        Returns the module's node_id. See :meth:`start_single_module` for id/name
        semantics.

        """
        path, mid = self._emit_path_and_id(parent_id=parent_id, name=name)
        # Stage 3: replay→live mode flip at the resume target boundary.
        self._maybe_flip_to_live_at_resume_target(mid)
        ev = ModuleStartMessage(
            module_type="parallel",
            name=name,
            module_id=mid,
            parent_id=mid,
            attach_under_id=parent_id,
            module=mid,
            text=name,
            path=path,
            **extra,
        )
        self._on(ev)
        return mid

    def end_module(
        self,
        *,
        parent_id: str,
        module_id: str,
        status: str = "done",
        **extra: Any,
    ) -> None:
        """Emit ``module_end``.

        ``status="stopped"`` overrides the rollup (used by :meth:`mark_stopped`);
        default rolls up from tasks.
        """
        ev = ModuleEndMessage(
            parent_id=module_id,
            module_id=module_id,
            status=status,
            module=module_id,
            text=module_id,
            **extra,
        )
        self._on(ev)

    # ── task-level operations ─────────────────────────────────────────────

    def start_task(
        self,
        *,
        name: str,
        parent_module_id: str,
        model: str | None = None,
        session: str | None = None,
        module: str | None = None,
        group: str | None = None,
        **extra: Any,
    ) -> str:
        """Construct a Task and emit ``task_start`` through ``_on``.

        ``name`` is the role label (``"gen_plan"``, ``"gen_plan_1"``,
        ``"upd_hypo"``); ``parent_module_id`` is the owning Module's
        node_id. node_id is ``gen_path_id(name, path)`` — deterministic
        across runs / forks / resume re-walks, so a recorded task's
        ``session_id`` survives for the agent backend's FORK override
        without any explicit slot-claim cursor (replay re-walk
        regenerates the same path → same id → :meth:`find_node` resolves
        the existing slot).

        """
        from aii_lib.run.module import Module
        from aii_lib.run.task import ClaudeAgentTask

        ts = _now_dt()

        # Direct parent lookup — no name-matching, no tree walking.
        parent = self._node_index.nodeid_to_node.get(parent_module_id)
        if not isinstance(parent, Module):
            raise TypeError(
                f"start_task: parent_module_id {parent_module_id!r} does not "
                f"resolve to a Module (got {type(parent).__name__})"
            )

        # Compute path + deterministic id. Same path on replay re-walk
        # → same id → :meth:`find_node` returns the existing slot, so
        # session_id captured during the original execution survives
        # for the agent backend's FORK override. In live mode ``existing``
        # is ``None`` and we attach a fresh task at the computed path.
        path, task_id = self._emit_path_and_id(parent_id=parent_module_id, name=name)
        existing = self._node_index.nodeid_to_node.get(task_id)
        if existing is None:
            task = ClaudeAgentTask(
                node_id=task_id,
                name=name,
                parent_id=parent_module_id,
                path=path,
            )
            parent.add_task(task)
            self._index(task)

        ev_extras: dict[str, Any] = {}
        if model:
            ev_extras["model"] = model
        if session:
            ev_extras["session_id"] = session

        ev_kwargs: dict[str, Any] = {
            "task_name": name,
            "agent_context": name,
            "parent_id": task_id,
            "attach_under_id": parent_module_id,
        }
        if module:
            ev_kwargs["module"] = module
        if group:
            ev_kwargs["group"] = group
        if ev_extras:
            ev_kwargs["extras"] = ev_extras

        ev = TaskStartMessage(
            ts=ts,
            task_id=task_id,
            path=path,
            **ev_kwargs,
            **extra,
        )
        self._on(ev)
        return task_id

    def end_task(
        self,
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
        """Emit ``task_end`` for ``task_id`` at the given terminal status."""
        if status not in ("done", "failed", "stopped"):
            raise ValueError(
                f"end_task status must be 'done', 'failed', or 'stopped'; got {status!r}"
            )
        ts = _now_dt()
        token = {"done": "OK", "failed": "FAILED", "stopped": "STOPPED"}[status]

        # Pre-ensure the task so dispatch can find it.
        self.ensure_for_task(task_id, name=name or "")

        text_value = text or token

        ev_extras: dict[str, Any] = {}
        if cost_usd is not None:
            ev_extras["total_cost"] = cost_usd
        if session_id:
            ev_extras["session_id"] = session_id

        ev_kwargs: dict[str, Any] = {}
        if name:
            ev_kwargs["task_name"] = name
            ev_kwargs["agent_context"] = name
        if module:
            ev_kwargs["module"] = module
        if group:
            ev_kwargs["group"] = group
        if ev_extras:
            ev_kwargs["extras"] = ev_extras

        ev = TaskEndMessage(
            ts=ts,
            task_id=task_id,
            parent_id=task_id,
            status=status,
            text=text_value,
            **ev_kwargs,
            **extra,
        )
        self._on(ev)

    def module_output(
        self,
        *,
        module_id: str,
        name: str,
        output: Any = None,
        **extra: Any,
    ) -> None:
        """Emit a module_output message — structured output of a module.

        ``module_id`` is the module's node_id (returned from
        ``start_*_module``); the message lives on that module's
        ``messages`` list (parent_id internally set to ``module_id``).
        ``name`` is the canonical module name (``"gen_strat"``, ``"gen_viz"``,
        …). ``output`` is a typed Pydantic model — dispatch assigns it
        to ``Module.output``.
        """
        self._on(
            ModuleOutputMessage(
                parent_id=module_id,
                name=name,
                output=output,
                **extra,
            )
        )

    def run_output(self, *, output: Any, **extra: Any) -> None:
        """Emit a run_output message — the run's final aggregate result.

        Test-side facade only; production pipeline emits via direct
        ``emit.run_output(...)``. Kept because the dispatch
        handler + idempotency contract still need exercising via
        :class:`tests.runtime.test_dispatch_idempotent`.
        """
        self._on(
            RunOutputMessage(
                parent_id=self.node_id,
                output=output,
                **extra,
            )
        )

    def mdgroup_output(
        self,
        *,
        group_id: str,
        output: Any,
        **extra: Any,
    ) -> None:
        """Emit a mdgroup_output message — phase aggregate result.

        ``group_id`` is the MdGroup's node_id (e.g. ``"hypo_loop"``,
        ``"invention_loop"``, ``"gen_paper_repo"``). Dispatch assigns
        ``output`` to that group's :attr:`AIINode.output`.
        """
        self._on(
            MdGroupOutputMessage(
                parent_id=group_id,
                output=output,
                **extra,
            )
        )

    def task_output(
        self,
        *,
        task_id: str,
        output: Any,
        **extra: Any,
    ) -> None:
        """Emit a task_output message — single-task structured result.

        ``task_id`` is the Task's node_id. Dispatch assigns ``output``
        to that task's :attr:`AIINode.output`.
        """
        self._on(
            TaskOutputMessage(
                parent_id=task_id,
                output=output,
                **extra,
            )
        )

    @staticmethod
    def _reject_task_name(kind: str, extra: dict) -> None:
        """Sub-task events (``agent_*``) MUST NOT carry ``task_name``.

        Names are a property of the structural tree (Task and above);
        sub-task events link via ``task_id`` and inherit their display
        name from the live Task node. Letting callers stuff a
        ``task_name`` into the wire payload here desyncs from the
        suffixed Task name (parallel-module slot ``_NN``) and produces
        duplicate-looking activity-feed rows.
        """
        if "task_name" in extra:
            raise TypeError(
                f"{kind}: task_name is not accepted on sub-task events; "
                f"the linked Task carries the canonical name."
            )

    def agent_user_prompt(
        self,
        task_id: str,
        text: str,
        *,
        prompt_source: str = "pipeline",
        prompt_index: int = 0,
        **extra: Any,
    ) -> None:
        """Emit an agent_user_prompt — user-facing or pipeline-supplied."""
        if self._replay_skip():
            return
        self._reject_task_name("agent_user_prompt", extra)
        from .messages import AgentUserPromptMessage

        self._on(
            AgentUserPromptMessage(
                task_id=task_id,
                parent_id=task_id,
                text=text,
                prompt_source=prompt_source,
                prompt_index=prompt_index,
                **extra,
            )
        )

    # ── agent emits ──────────────────────────────────────────────────────

    def agent_start(self, task_id: str, **extra: Any) -> None:
        """Emit an agent_start message — LLM-call open bracket inside a task."""
        if self._replay_skip():
            return
        self._reject_task_name("agent_start", extra)
        self._on(AgentMessage(type="agent_start", task_id=task_id, parent_id=task_id, **extra))

    def agent_end(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Emit an agent_end message — LLM-call close bracket inside a task."""
        if self._replay_skip():
            return
        self._reject_task_name("agent_end", extra)
        self._on(
            AgentEndMessage(
                task_id=task_id,
                parent_id=task_id,
                session_id=session_id,
                **extra,
            )
        )

    def agent_retry(
        self,
        task_id: str,
        *,
        attempt: int = 0,
        reason: str = "",
        **extra: Any,
    ) -> None:
        """Emit an agent_retry message — agent loop retry."""
        if self._replay_skip():
            return
        self._reject_task_name("agent_retry", extra)
        self._on(
            AgentRetryMessage(
                task_id=task_id,
                parent_id=task_id,
                attempt=attempt,
                reason=reason,
                **extra,
            )
        )

    def agent_hook(
        self,
        task_id: str,
        *,
        hook_type: str,
        text: str = "",
        **extra: Any,
    ) -> None:
        """Emit an agent_hook message — SDK hook callback fired.

        ``hook_type`` is the SDK event name (``"PostToolUse"``,
        ``"PreToolUse"``, ``"UserPromptSubmit"``, …). ``text`` is a
        human-readable description of what the hook did (e.g. the
        time-remaining warning string for ``_TimeRemainingHook``).
        """
        if self._replay_skip():
            return
        self._reject_task_name("agent_hook", extra)
        self._on(
            AgentHookMessage(
                task_id=task_id,
                parent_id=task_id,
                hook_type=hook_type,
                text=text,
                **extra,
            )
        )


__all__ = [
    "Run",
    "set_dispatch",
    "set_ensure_for_task",
]
