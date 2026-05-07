"""Module — one substep within a parent (SeqMdGroup or LoopIteration).

Two concrete subtypes share an identical shape and differ only in
dispatch semantics:

  - ``SingleTModule``:   holds exactly one Task (single-task module).
  - ``ParallelTModule``: tasks fan out concurrently (multi-task).

Both subclasses store their tasks in :attr:`Module.children` — a flat
``list[Task]`` on the base class. ``SingleTModule`` enforces
``len(children) <= 1`` via a pydantic validator; ``ParallelTModule``
holds N concurrent branches.

Identity is :attr:`AIINode.node_id` — pipeline call sites pass a
canonical name (``"gen_hypo_it1"`` / ``"gen_plan_it2"`` / ``"gen_repo"``) when
constructing modules. ``name`` carries the canonical step token
(``"gen_hypo"`` / ``"gen_plan"`` / ``"gen_repo"``) used by the mapper for UI
labels.

Resume / restart treatment lives at the agent-dispatch site, not on
Module:

  * Resume-with-prompt is handled by the agent backend's FORK
    override — at :meth:`Agent.run` time, when the dispatched task
    lives under :attr:`Run._pending_resume_target` and carries a
    captured ``session_id``, the backend swaps to ``session_type=FORK``
    + ``resume_session_id`` and replaces prompts with ``[run.prompt]``.
    The forward pipeline runs identically; the resume turn just rides
    the normal substep dispatch.
  * Restart-from-target is handled by :meth:`Run.remove_children` at boot
    + :meth:`Module.truncate_self` (clears children + flips
    IN_PROGRESS) — followed by the forward pipeline re-dispatching
    target's substep fresh.

:meth:`Module.prep_fork` is fork-specific side prep (SDK session
bucket copy) that runs once during :meth:`Run.from_fork` setup.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from .node import AIINode, NodeStats, NodeStatus

# AnyTask is used in the ``children: list[AnyTask]`` annotation below;
# pydantic's ``model_rebuild()`` resolves it against THIS module's
# globals at boot. ``Task`` is the un-annotated alias for the rest of
# this module. DO NOT move under TYPE_CHECKING (ruff's TC001 will be
# tempted) — pydantic schema build needs the runtime symbol.
from .task import AnyTask, Task


class Module(AIINode):
    """Concrete base for one substep instance.

    Holds a flat list of child Tasks (:attr:`children`, typed override
    of :attr:`AIINode.children`). Subclasses differ only in dispatch
    semantics — the storage shape is identical.

    Inherits from :class:`AIINode`: identity (``node_id`` /
    ``parent_id``), lifecycle, and the per-node event log
    (``messages``). Carries its own :attr:`stats` aggregate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stats: NodeStats = Field(default_factory=NodeStats)

    children: list[AnyTask] = Field(default_factory=list)
    """Tasks under this module. ``SingleTModule`` holds exactly one;
    ``ParallelTModule`` runs them concurrently. No sub-branches — each
    child Task IS one branch (parallel) or the lone task (single).
    Discriminated union over Task subclasses so seed-driven rehydrate
    routes back to the concrete subclass (``ClaudeAgentTask`` →
    preserves ``session_id``)."""

    @property
    def tasks(self) -> list[Task]:
        """Alias for :attr:`children`."""
        return self.children

    def computed_status(self) -> NodeStatus:
        """Status rolled up from tasks (ground truth when present).

        Order: any IN_PROGRESS → IN_PROGRESS; any FAILED → FAILED;
        any STOPPED → STOPPED; otherwise (all DONE) → DONE.
        Empty children → keep current status.
        """
        ts = self.tasks
        if not ts:
            return self.status
        if any(t.is_active for t in ts):
            return NodeStatus.IN_PROGRESS
        if any(t.status == NodeStatus.FAILED for t in ts):
            return NodeStatus.FAILED
        if any(t.status == NodeStatus.STOPPED for t in ts):
            return NodeStatus.STOPPED
        return NodeStatus.DONE

    def _apply_start(self) -> None:
        """module_start — flip PENDING → IN_PROGRESS."""
        self.status = NodeStatus.IN_PROGRESS

    def _apply_end(self, status_override: NodeStatus | None = None) -> None:
        """Module-end transition.

        If ``status_override`` is set (e.g. :meth:`Run._finalize_orphans`
        passes ``STOPPED``), use it directly. Otherwise: empty children → DONE;
        otherwise roll up via :meth:`computed_status`.
        """
        if status_override is not None:
            self.status = status_override
            return
        if not self.tasks:
            self.status = NodeStatus.DONE
            return
        self.status = self.computed_status()

    def _apply_task_started(self) -> None:
        """Ensure module is IN_PROGRESS when a task starts.

        Covers a pre-declared parallel module whose children kick off
        without an explicit module_start.
        """
        if self.status == NodeStatus.PENDING:
            self.status = NodeStatus.IN_PROGRESS

    def _apply_task_ended(self) -> None:
        self.status = self.computed_status()

    def add_task(self, task: Task) -> None:
        """Attach ``task`` as a child of this module.

        Subclasses may override to enforce structural constraints
        (e.g. :class:`SingleTModule` caps at one).
        """
        self.children.append(task)

    def task_sequence(self, task_id: str) -> int | None:
        """Return the task's 0-based index in :attr:`children`.

        For ``ParallelTModule`` this is the slot the task occupies in
        the fan-out — used by sequencing sinks (the console + sequenced
        clone) to decide display/archive order. For ``SingleTModule``
        the lone child is at index 0.

        Returns ``None`` when the task is not in this module's children.
        """
        if not task_id:
            return None
        for i, t in enumerate(self.children):
            if t.node_id == task_id:
                return i
        return None

    # ── resume / fork polymorphism ──────────────────────────────────────

    def truncate_self(self) -> None:
        """Discard child Tasks and reset to IN_PROGRESS.

        Used by :meth:`Run.remove_children` for no-prompt resume ("restart
        from this module") so the next forward pipeline pass dispatches
        fresh tasks under the empty module. Default implementation works
        for both subclasses since both store children in ``self.children``.
        Subclass overrides preserved as extension points if a future Module
        type needs different truncation semantics.
        """
        self.children.clear()
        self.status = NodeStatus.IN_PROGRESS
        self.end_at = None


class SingleTModule(Module):
    """Single-task substep. Holds exactly one Task (validated)."""

    kind: Literal["single_t_module"] = "single_t_module"
    """Wire-level discriminator. Lets downstream consumers (the FE
    reconstructed tree, codegen-driven type narrowing) tell single-task
    modules apart from parallel ones without name heuristics."""

    @model_validator(mode="after")
    def _enforce_single_task(self) -> SingleTModule:
        if len(self.children) > 1:
            raise ValueError(
                f"SingleTModule.children must hold at most one Task "
                f"(got {len(self.children)} for node_id={self.node_id!r})"
            )
        return self

    @property
    def task(self) -> Task | None:
        """The lone child Task, or ``None`` before it's been added."""
        return self.children[0] if self.children else None

    def add_task(self, task: Task) -> None:
        """Attach the lone Task. Raises if a child already exists."""
        if self.children:
            raise ValueError(
                f"SingleTModule already has a task (node_id={self.node_id!r}); cannot add a second"
            )
        self.children.append(task)


class ParallelTModule(Module):
    """Fan-out substep. Each child Task runs concurrently with siblings."""

    kind: Literal["parallel_module"] = "parallel_module"
    """Wire-level discriminator (see :class:`SingleTModule.kind`)."""


# Polymorphic union with a tagged discriminator. ``Field(discriminator='kind')``
# lets Pydantic dispatch JSON → correct subclass in O(1) on deserialize, and
# tells ``pydantic2ts`` to emit a discriminated union on the FE so TypeScript
# can narrow with ``mod.kind === "single_t_module"`` instead of poking at
# ``name`` strings.
AnyModule = Annotated[
    SingleTModule | ParallelTModule,
    Field(discriminator="kind"),
]


__all__ = [
    "AnyModule",
    "Module",
    "ParallelTModule",
    "SingleTModule",
]
