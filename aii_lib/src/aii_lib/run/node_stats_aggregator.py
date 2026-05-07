"""NodeStats roll-up + node summary formatter.

Three helpers, all fired by the dispatcher:

  - :func:`apply_leaf_summary` — on every ``agent_summary`` /
    ``llm_summary`` event, adds the leaf's standardized fields to the
    owning Task's ``stats`` and to every ancestor (Module → MdGroup →
    [LoopIteration →] Run). Keeps each node's cost/token aggregate
    live during the run.

  - :func:`update_derived_stats_from_message` — on every routed
    message (any type), pushes its timestamp up the parent chain to
    update each ancestor's
    :attr:`NodeStats.first_message_at` /
    :attr:`NodeStats.last_message_at` /
    :attr:`NodeStats.runtime_seconds` (derived from the bounds), and
    bumps :attr:`NodeStats.total_messages` by one at every level.
    Walks only the affected path (owner + ancestors), not the full
    tree.

  - :func:`format_node_summary` — on every ``*_end`` event (task /
    module / iteration / mdgroup / run), formats a one-line summary
    from the node's accumulated ``stats``. The dispatcher writes the
    result into the end event's ``text`` field so the activity feed
    shows a rich summary line at every level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.run import get_current_run

if TYPE_CHECKING:
    from datetime import datetime

    from aii_lib.run.node import AIINode


def apply_leaf_summary(
    *,
    task_id: str,
    total_cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    """Push a leaf summary's standardized fields up the tree.

    The arguments are PER-CALL deltas from the originating
    ``agent_summary`` / ``llm_summary`` bus event (one fire per
    ``ClaudeSDKClient`` session = one ``ResultMessage``). The three
    input-side fields (``input_tokens`` + ``cache_read_tokens`` +
    ``cache_write_tokens``) are summed and ADDED into each ancestor's
    :attr:`NodeStats.cum_all_input_tokens`. ``output_tokens`` adds
    into :attr:`NodeStats.cum_all_output_tokens`. ``total_cost`` adds
    into :attr:`NodeStats.total_cost`. Walks ``parent_id`` from Task →
    Module → [Iteration →] MdGroup → Run.

    The ``all`` prefix on the destination fields signals the input-side
    sum spans every input token type (uncached + cache-read +
    cache-write) — i.e. the full prompt context billed across calls.

    No-op if the run / task isn't found in the index.
    """
    run = get_current_run()
    if run is None:
        return
    all_input = (input_tokens or 0) + (cache_read_tokens or 0) + (cache_write_tokens or 0)
    node = run.find_node(task_id)
    while node is not None:
        node.stats.total_cost += total_cost
        node.stats.cum_all_input_tokens += all_input
        node.stats.cum_all_output_tokens += output_tokens
        if node.parent_id is None:
            break
        node = run.find_node(node.parent_id)


def update_derived_stats_from_message(node: AIINode, ts: datetime) -> None:
    """Update derived stats on a node and every ancestor.

    For each node in the chain:
      - Expand the ``first_message_at`` / ``last_message_at`` bounds
        with ``ts``;
      - Recompute ``runtime_seconds = (last - first).total_seconds()``;
      - Bump ``total_messages`` by one (each ancestor counts every
        message that landed anywhere in its subtree).

    Only the path from the routed-message owner up to the run root is
    touched — never the full tree.
    """
    run = get_current_run()
    if run is None:
        return
    while node is not None:
        s = node.stats
        if s.first_message_at is None or ts < s.first_message_at:
            s.first_message_at = ts
        if s.last_message_at is None or ts > s.last_message_at:
            s.last_message_at = ts
        if s.first_message_at is not None and s.last_message_at is not None:
            s.runtime_seconds = (s.last_message_at - s.first_message_at).total_seconds()
        s.total_messages += 1
        if node.parent_id is None:
            break
        node = run.find_node(node.parent_id)


def _fmt_tokens(n: int) -> str:
    """Format token counts compactly: ``1,234`` / ``12.3K`` / ``1.2M``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def format_node_summary(label: str, node: AIINode) -> str:
    """One-line summary of a node's accumulated stats.

    Format: ``"{label} | Cost: $X.XXXX | In-Out: Y-Z | Runtime: T.Ts | N msgs"``.
    Mirrors the agent-side ``_format_summary_line`` style — pulls
    every field from ``NodeStats`` (cost / tokens / runtime /
    total_messages). Pieces with no data are omitted.
    """
    s = node.stats
    parts = [label]
    if s.total_cost:
        parts.append(f"Cost: ${s.total_cost:.4f}")
    if s.cum_all_input_tokens or s.cum_all_output_tokens:
        parts.append(
            f"In-Out: {_fmt_tokens(s.cum_all_input_tokens)}-{_fmt_tokens(s.cum_all_output_tokens)}"
        )
    if s.runtime_seconds > 0:
        parts.append(f"Runtime: {s.runtime_seconds:.1f}s")
    if s.total_messages:
        parts.append(f"{s.total_messages} msgs")
    return " | ".join(parts)


__all__ = [
    "apply_leaf_summary",
    "format_node_summary",
    "update_derived_stats_from_message",
]
