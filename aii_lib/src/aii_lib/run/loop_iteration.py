"""LoopIteration — one round inside a LoopMdGroup.

A LoopMdGroup contains 1..N LoopIterations. Each iteration holds the
modules executed during that round. Module status rolls up into
LoopIteration.status; LoopIteration status rolls up into LoopMdGroup.

Generic — pipeline-specific concepts (cards, phases) live one layer up
(in the mapper as private inline dicts). LoopIteration only knows
iteration index + status + child modules.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .module import AnyModule
from .node import AIINode, NodeStats, NodeStatus


class LoopIteration(AIINode):
    """One round inside a LoopMdGroup.

    Inherits from :class:`AIINode`: identity, lifecycle, ``children``,
    and per-node ``messages`` log. Carries its own :attr:`stats`
    aggregate. The iteration's 1-based index is its position in
    :attr:`LoopMdGroup.children` — no separate field.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stats: NodeStats = Field(default_factory=NodeStats)

    # Forward reference — resolved at model_rebuild against module
    # globals so aii_pipeline can widen the union to include typed
    # substep modules. See ``aii_lib.run.run.Run.children`` for the
    # full mechanism.
    children: list[AnyModule] = Field(default_factory=list)
    """Modules executed during this round, in pipeline order."""

    # ── domain mutations ──────────────────────────────────────────────────

    def _apply_start(self) -> None:
        """iteration_start — flip PENDING → IN_PROGRESS."""
        self.status = NodeStatus.IN_PROGRESS

    def _apply_end(self, status_override: NodeStatus | None = None) -> None:
        """Iteration-end transition.

        ``status_override`` (used by :meth:`Run._finalize_orphans`) wins; otherwise
        roll up from children: any FAILED → FAILED; any STOPPED →
        STOPPED; else DONE.
        """
        if status_override is not None:
            self.status = status_override
            return
        if any(m.status == NodeStatus.FAILED for m in self.children):
            self.status = NodeStatus.FAILED
        elif any(m.status == NodeStatus.STOPPED for m in self.children):
            self.status = NodeStatus.STOPPED
        else:
            self.status = NodeStatus.DONE

    def _apply_module_added(self, module: AnyModule) -> None:  # type: ignore[valid-type]
        """A new Module was attached to this iteration."""
        self.children.append(module)


__all__ = ["LoopIteration"]
