"""MdGroup — top-level grouping inside a Run.

In v26 the tree is::

    Run > MdGroup (Loop|Seq) > [LoopIteration >] Module > Task

Every parent stores its descendants in a uniform ``children`` field
(typed per parent). The two MdGroup subclasses encode whether the
group is sequential or iterated:

  - SeqMdGroup.children: list[AnyModule] — modules executed in order.
  - LoopMdGroup.children: list[LoopIteration] — N iterations; each
    iteration's :attr:`LoopIteration.children` holds the per-iteration
    module list.

Identity is :attr:`AIINode.node_id` — pipeline call sites pass a
canonical name (``"gen_paper_repo"`` / ``"invention_loop"``) when
constructing groups so the ``node_id`` carries semantic meaning.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from .loop_iteration import LoopIteration
from .module import AnyModule
from .node import AIINode, NodeStats, NodeStatus


class MdGroup(AIINode):
    """Abstract base for one top-level group inside a Run.

    Inherits from :class:`AIINode`: identity (``node_id``), lifecycle,
    and per-node ``messages`` log. Carries its own :attr:`stats`
    aggregate. The two concrete shapes (Seq / Loop) override
    ``children`` with their typed list (modules vs iterations).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stats: NodeStats = Field(default_factory=NodeStats)

    def _apply_start(self) -> None:
        """group_start — flip PENDING → IN_PROGRESS."""
        self.status = NodeStatus.IN_PROGRESS

    def _apply_end(self, status_override: NodeStatus | None = None) -> None:
        """Group-end transition.

        ``status_override`` (used by :meth:`Run._finalize_orphans`) wins; otherwise
        roll up from children: empty → DONE; any FAILED → FAILED;
        any STOPPED → STOPPED; else DONE.
        """
        if status_override is not None:
            self.status = status_override
            return
        if not self.children:
            self.status = NodeStatus.DONE
            return
        if any(c.status == NodeStatus.FAILED for c in self.children):
            self.status = NodeStatus.FAILED
        elif any(c.status == NodeStatus.STOPPED for c in self.children):
            self.status = NodeStatus.STOPPED
        else:
            self.status = NodeStatus.DONE


class SeqMdGroup(MdGroup):
    """Sequential group — a flat list of Modules executed in order.

    Used for non-iterated phases (e.g. ``gen_paper_repo``, ``gen_hypo``).
    """

    kind: Literal["seq_mdgroup"] = "seq_mdgroup"
    """Wire-level discriminator. Lets downstream consumers (the FE
    reconstructed tree, codegen-driven type narrowing) tell sequential
    groups apart from loop groups without name heuristics."""

    # Forward reference — see ``aii_lib.run.run.Run.children`` for the
    # full mechanism (aii_pipeline widens the union at process boot).
    children: list[AnyModule] = Field(default_factory=list)
    """Modules under this group, executed in order."""

    def _apply_module_added(self, module: AnyModule) -> None:  # type: ignore[valid-type]
        self.children.append(module)


class LoopMdGroup(MdGroup):
    """Loop group — holds N LoopIterations of the same module sequence.

    Used for iterated phases (e.g. ``hypo_loop``, ``invention_loop``).
    """

    kind: Literal["loop_mdgroup"] = "loop_mdgroup"
    """Wire-level discriminator (see :class:`SeqMdGroup.kind`)."""

    children: list[LoopIteration] = Field(default_factory=list)
    """LoopIterations under this group; each iteration owns its own
    module list. The 1-based iteration number is the iteration's
    position in this list."""

    def find_iteration(self, iteration: int) -> LoopIteration | None:
        """Look up by 1-based index. Returns ``None`` if out of range."""
        idx = iteration - 1
        if 0 <= idx < len(self.children):
            return self.children[idx]
        return None

    def iteration_number(self, iteration: LoopIteration) -> int | None:
        """Return the 1-based iteration number for ``iteration``.

        Inverse of :meth:`find_iteration`. Returns ``None`` if the
        iteration is not in this group's children.
        """
        try:
            return self.children.index(iteration) + 1
        except ValueError:
            return None

    def _apply_iteration_started(self, it: LoopIteration) -> None:
        self.children.append(it)

    def _apply_iteration_ended(
        self,
        iteration: int,
        status_override: NodeStatus | None = None,
    ) -> None:
        it = self.find_iteration(iteration)
        if it is not None:
            it._apply_end(status_override=status_override)


# Polymorphic union with a tagged discriminator. ``Field(discriminator='kind')``
# lets Pydantic dispatch JSON → correct subclass in O(1) on deserialize, and
# tells ``pydantic2ts`` to emit a discriminated union on the FE so TypeScript
# can narrow with ``g.kind === "loop_mdgroup"`` instead of poking at
# ``name`` strings.
AnyMdGroup = Annotated[
    SeqMdGroup | LoopMdGroup,
    Field(discriminator="kind"),
]


__all__ = [
    "AnyMdGroup",
    "LoopMdGroup",
    "MdGroup",
    "SeqMdGroup",
]
