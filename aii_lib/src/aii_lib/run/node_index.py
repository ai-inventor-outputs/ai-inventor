"""NodeIndex — unified id index over a Run.

Maintained as :attr:`Run.node_index`. Two dicts:

  * :attr:`nodeid_to_node` — flat ``node_id → AIINode`` lookup over
    **every** node in the run, including messages (``BaseMessage``
    subclasses inherit from :class:`AIINode`, so they live alongside
    tree nodes in the same map). Tree nodes are added by
    :meth:`index_node`; messages are added by :meth:`add_message`.

  * :attr:`id_to_ancestors` — per-message ancestor map keyed by
    message ``node_id``. Populated by :meth:`add_message` whenever a
    typed event is routed to its owning node.

Pure data structure. Knows nothing about dispatch or the to_app
transport — callers feed it the resolved ancestor set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .messages import BaseMessage
    from .node import AIINode
    from .node_id import NodeID


class NodeIndex:
    """Run-scoped indices: nodes by id, messages by ancestor scope."""

    def __init__(self) -> None:
        self.nodeid_to_node: dict[NodeID, AIINode] = {}
        self.id_to_ancestors: dict[NodeID, set[NodeID]] = {}

    # ── node index ────────────────────────────────────────────────────────

    def index_node(self, node: AIINode) -> None:
        """Add ``node`` to :attr:`nodeid_to_node`.

        Idempotent — re-indexing replaces the prior entry.
        """
        nid = node.node_id
        if not nid:
            return
        self.nodeid_to_node[nid] = node

    def rebuild_nodes(self, root: AIINode) -> None:
        """Walk the tree from ``root`` and repopulate :attr:`nodeid_to_node`.

        Called by ``CloneSink.load`` after seed-event hydration so the
        snapshot tree is queryable.
        """
        self.nodeid_to_node = {}
        stack: list[AIINode] = [root]
        while stack:
            n = stack.pop()
            self.index_node(n)
            stack.extend(getattr(n, "children", []))

    # ── message index ─────────────────────────────────────────────────────

    def add_message(self, msg: BaseMessage, *, ancestors: set[NodeID]) -> None:
        """Record ``msg`` in both maps:

          * :attr:`nodeid_to_node[msg.node_id] = msg` — so
            ``GET /messages/{node_id}/full`` can resolve in O(1).
          * :attr:`id_to_ancestors[msg.node_id] = ancestors` — so
            destructive truncate can drop the subtree's messages.

        Idempotent: a re-emit of the same ``node_id`` is a no-op so
        replay paths can call freely without double-counting.
        """
        nid = msg.node_id
        if nid in self.id_to_ancestors:
            return
        self.id_to_ancestors[nid] = ancestors
        self.nodeid_to_node[nid] = msg


__all__ = ["NodeIndex"]
