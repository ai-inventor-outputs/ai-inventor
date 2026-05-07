"""AIINode — the abstract base for any node in the run tree.

A single shared base class for every node: structural ones (Run,
MdGroup, SeqMdGroup, LoopMdGroup, LoopIteration, Module, SingleTModule,
ParallelTModule, Task) AND the bus messages themselves (BaseMessage and
its subclass family). Every node carries:

  - ``node_id: NodeID`` — 12-char alphanumeric identity (auto-generated
    per instance; type info lives on the runtime class).
  - ``parent_id: NodeID | None`` — pointer to the structural parent.
  - ``status: NodeStatus`` — shared lifecycle enum.
  - ``start_at`` / ``end_at: datetime | None`` (UTC-validated).
  - ``children: list`` — override per subclass with typed list.
  - ``messages: list[BaseMessage]`` — per-node event log routed by
    ``parent_id`` (widened to the discriminated ``AnyMessage`` union
    by :func:`aii_lib.run.messages.bind_message_union` at boot).
  - ``output: Any`` — typed result for nodes that produce one (Run /
    MdGroup / Module / Task); widened to a discriminated union over
    every known output class by ``bind_pipeline_typed_unions``.

``stats: NodeStats`` (cost / token / runtime aggregate) lives on the
subclasses that carry measurements (Task / Module / MdGroup /
LoopIteration / Run) — not on this base.

The hierarchy lives in the ``parent_id`` field plus :class:`Run`-level
indexes (``Run._node_index``).

``RunStatus`` from ``aii_lib.run_lifecycle`` is *not* retired — it's a
separate lifecycle FSM with values (STARTING, RUNNING, COMPLETED,
STOPPED, FAILED) used by the server-side orchestrator + DB column +
frontend Literal type. ``Run`` keeps it as ``Run.status`` for the
moment; ``Run.node_id`` is its tree-position identity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    # Forward-only — :class:`BaseMessage` lives in ``messages.py`` which
    # imports :class:`AIINode` from this module. To break the cycle the
    # type-checking import is gated; the runtime annotation
    # ``messages: list[BaseMessage]`` is a string forward reference,
    # resolved by ``AIINode.model_rebuild()`` at the bottom of
    # ``messages.py`` once :class:`BaseMessage` is defined.
    #
    # DO NOT move this to the top — ruff's TC001 / I001 will be tempted,
    # but eager-importing here re-creates the runtime cycle (boot crash
    # with "cannot import name 'AIINode' from partially initialized
    # module").
    from .messages import BaseMessage  # noqa: TC004
from .node_id import NodeID, generate_short_id

_UTC_OFFSET = UTC.utcoffset(None)


# ---------------------------------------------------------------------------
# NodeStatus — single shared status enum for every tree node
# ---------------------------------------------------------------------------


class NodeStatus(StrEnum):
    """Shared lifecycle enum for every node in the run tree.

    Values:
      - PENDING:     declared but not yet started (the construction
        default — a node enters the tree as PENDING and is flipped
        to IN_PROGRESS by its ``*_start`` lifecycle event)
      - IN_PROGRESS: currently executing
      - DONE:        finished successfully
      - FAILED:      finished with error (also the rollup result for
        any mixed-terminal aggregate)
      - STOPPED:     terminated by user / external signal
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# NodeStats — cost / token / runtime aggregate
# ---------------------------------------------------------------------------


class NodeStats(BaseModel):
    """Cost / token / runtime aggregate carried by Task / Module / MdGroup / Run."""

    model_config = ConfigDict(extra="forbid")

    total_cost: float = 0.0
    """USD cost (token_cost + tool_cost)."""

    cum_all_input_tokens: int = 0
    """Sum of EVERY per-call input-side token field across every
    ``agent_summary`` / ``llm_summary`` event landed in this node's
    subtree. Per call the aggregator sums:

        input_tokens + cache_read_tokens + cache_write_tokens

    so this is the "full context tokens billed" number — uncached new
    input PLUS cache-read tokens PLUS cache-write tokens. Cumulative
    across calls (each call's combined sum adds in)."""
    cum_all_output_tokens: int = 0
    """Sum of per-call ``output_tokens`` across every ``agent_summary``
    / ``llm_summary`` event in this node's subtree. ``all`` prefix kept
    for naming symmetry with :attr:`cum_all_input_tokens`; output has
    only one type so the prefix is informational, not a sum."""

    current_all_input_tokens: int = 0
    """Live context-window size for the task's CURRENT LLM call —
    sum of ``input_tokens`` + ``cache_read_input_tokens`` +
    ``cache_creation_input_tokens`` from the most recent
    ``agent_message_delta``. SET (overwrite), not added. Per-task only:
    Module / Group / Run don't carry a meaningful "current call" value.
    Updated mid-stream; useful for live FE displays showing the agent's
    context occupancy."""
    current_all_output_tokens: int = 0
    """Live output-token count for the task's CURRENT LLM call. Grows
    during streaming as each ``message_delta`` reports an updated
    cumulative ``output_tokens``. ``all`` prefix kept for symmetry."""

    runtime_seconds: float = 0.0
    """Derived: ``last_message_at - first_message_at`` in seconds. Updated
    incrementally by the dispatcher every time a message routes to this
    node or any of its descendants."""

    total_messages: int = 0

    first_message_at: datetime | None = None
    """Earliest message timestamp seen anywhere in this node's subtree.
    Used to derive :attr:`runtime_seconds` — read by
    :func:`aii_lib.run.node_stats_aggregator.update_runtime_from_message`."""

    last_message_at: datetime | None = None
    """Latest message timestamp seen anywhere in this node's subtree.
    Together with :attr:`first_message_at`, derives :attr:`runtime_seconds`."""


# ---------------------------------------------------------------------------
# AIINode — abstract base for any node in the run tree
# ---------------------------------------------------------------------------


class AIINode(BaseModel):
    """Abstract base for any node in the run tree.

    Subclasses (Run, MdGroup, SeqMdGroup, LoopMdGroup, LoopIteration,
    Module, SingleTModule, ParallelTModule, Task) inherit ``node_id``,
    ``parent_id``, ``status``, ``start_at`` / ``end_at`` and override
    ``children`` to expose their tree shape. ``stats`` lives on the
    subclasses that carry aggregate measurements (Task / Module /
    MdGroup / Run) — not on this base.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: NodeID = Field(default_factory=generate_short_id)
    """Free-form string identity. Default factory generates a 12-char
    alphanumeric short id; structural emitters override with
    :func:`aii_lib.run.node_id.gen_path_id` so the suffix is a
    deterministic ``uuid5(NS, path).hex[:12]`` — same path → same id
    across runs, parent / fork pairs, and resume re-walks."""

    path: str = ""
    """Structural address from the run root, used to derive
    :attr:`node_id`. Empty for the :class:`Run` root; child paths
    have shape ``"{parent.path}/{name}[{idx}]"`` where ``idx`` is the
    sibling-position-among-same-name (0-based, monotonic per Run
    instance via :attr:`Run._emit_counter`). Set once at construction
    by the emitter that creates the node."""

    name: str = ""
    """Optional human-readable label for this node. Empty by default
    — set by callers when there's a presentable name (canonical step
    token for Modules, display name for Tasks, free-form for Runs /
    groups / iterations)."""

    parent_id: NodeID | None = None
    """Pointer to the structural parent in the run tree. ``None`` for
    :class:`Run` (the tree root). Populated when the dispatcher attaches
    a node under its parent — :meth:`Run._index` keeps
    :attr:`Run.node_index` up-to-date."""

    status: NodeStatus = NodeStatus.PENDING
    """Default PENDING — a node is created in the tree before it
    starts (e.g. by a parent pre-declaring its children, or by boot
    scaffolds in resume / fork). The matching ``*_start`` lifecycle
    event flips it to IN_PROGRESS via the node's ``_apply_start``
    method. Live-only nodes that are constructed and started in the
    same dispatch tick still pass through PENDING for one step."""

    start_at: datetime | None = None
    """Stamped at construction (or first lifecycle event). tz-aware UTC
    datetime — the validator below rejects naive + non-UTC timestamps so
    every node in the tree carries a timestamp the :class:`Timestamp`
    wrapper would accept (this is the lib-side guard parallel to
    ``aii_lib.timestamp``)."""

    end_at: datetime | None = None
    """Set when the node transitions to a terminal status. Same UTC-only
    invariant as :attr:`start_at` — see the validator."""

    @field_validator("start_at", "end_at")
    @classmethod
    def _must_be_utc(cls, v: datetime | None) -> datetime | None:
        """Reject naive + non-UTC datetimes on AIINode timestamp fields.

        Mirrors :class:`aii_lib.timestamp.Timestamp` — every node carries
        a timestamp that round-trips through ``Timestamp(dt=...)`` without
        further coercion. Serializers can therefore call ``.isoformat()``
        and trust the suffix is ``+00:00``.
        """
        if v is None:
            return None
        if v.utcoffset() is None:
            raise ValueError(
                f"AIINode timestamps must be tz-aware; got naive {v!r}",
            )
        if v.utcoffset() != _UTC_OFFSET:
            raise ValueError(
                f"AIINode timestamps must be UTC; got tzinfo={v.tzinfo!r} "
                f"with offset {v.utcoffset()!r}",
            )
        return v

    children: list = Field(default_factory=list)
    """Structural descendants. Subclasses override with a typed list
    (e.g. ``Run.children: list[AnyMdGroup]``); leaf nodes inherit the
    empty default."""

    messages: list[BaseMessage] = Field(default_factory=list)
    """Per-node event log — append-only typed events scoped to this
    node. The dispatcher routes each event to its scope-owning node
    (Run/MdGroup/LoopIteration/Module/Task). The global activity
    timeline is reconstructed by walking the tree and merge-sorting
    every node's ``messages``.

    Lifecycle is captured by ``status`` / ``start_at`` /
    ``end_at`` and is independent of this list — the messages
    here are content events (agent_*, llm_*, status_*, task_*, etc.),
    not pure state transitions."""

    output: Any = None
    """Result returned by this node's ``execute()`` (or equivalent).

    Annotation widened to a discriminated union of all known output
    classes by ``aii_pipeline.run.typed_union.bind_pipeline_typed_unions``
    at process boot. At base-class level it stays ``Any`` so ``aii_lib``
    doesn't import ``aii_pipeline``.

    Populated by:
      * Live execution: callers (pipeline.py, agent task wrappers,
        non-agent module bodies) set ``node.output`` directly when
        their work returns.
      * Replay: dispatch handlers for ``run_output`` /
        ``mdgroup_output`` / ``module_output`` / ``task_output``
        events deserialize the JSON payload into the typed model
        and assign it.

    Round-trip: ``model_dump_json()`` serializes via the discriminator;
    ``model_validate_json()`` reconstructs the typed subclass via
    pydantic's discriminated-union machinery — same mechanism that
    works for ``Run.children`` / ``LoopIteration.children`` already.
    Plain ``None`` for nodes whose ``execute()`` doesn't return data
    (LoopIteration, in-flight nodes pre-completion)."""

    # ── tree shape ────────────────────────────────────────────────────────
    # Subclasses with descendants override ``children`` with a typed
    # ``list[<ChildType>]`` field. Leaf nodes leave the empty default.

    # ── lifecycle predicates ──────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Check if node is currently running."""
        return self.status == NodeStatus.IN_PROGRESS


__all__ = ["AIINode", "NodeStats", "NodeStatus"]
