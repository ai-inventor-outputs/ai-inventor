"""Task — abstract leaf in the run tree.

Concrete subclasses carry execution-engine-specific runtime state:

  - :class:`ClaudeAgentTask` — task executed by a Claude agent;
    holds ``session_id`` for resume.

Future engines (other LLM SDKs, human-in-the-loop tasks, webhook
tasks) would each get their own :class:`Task` subclass.

Identity is :attr:`AIINode.node_id`. Two tasks with the same
``node_id`` are the same task. Cost/tokens/runtime totals live on
:attr:`AIINode.stats`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field

from aii_lib.task_id import TaskID

from .node import AIINode, NodeStats, NodeStatus

if TYPE_CHECKING:
    from datetime import datetime


class Task(AIINode):
    """Abstract base for any concrete task type.

    Leaf node — has no structural children. Inherits from :class:`AIINode`:
    identity (``node_id`` / ``name``), lifecycle, ``messages`` (per-task
    event log). Carries its own :attr:`stats` aggregate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["task"] = "task"
    """Wire-level discriminator. Lets ``model_validate`` pick the right
    Task subclass on seed-driven rehydrate (fork_init reload) — without
    this tag, every ClaudeAgentTask would collapse back to base Task."""

    stats: NodeStats = Field(default_factory=NodeStats)

    @property
    def is_active(self) -> bool:
        """Check if task is currently running."""
        return self.status == NodeStatus.IN_PROGRESS

    # ── domain mutations ──────────────────────────────────────────────────

    def _apply_start(self, *, ts: datetime | None) -> None:
        """task_start — flip PENDING → IN_PROGRESS and stamp start_at."""
        self.status = NodeStatus.IN_PROGRESS
        if self.start_at is None and ts is not None:
            self.start_at = ts

    def _apply_end(
        self,
        *,
        status: NodeStatus,
        ts: datetime | None,
    ) -> None:
        """task_end — set terminal status + end_at."""
        self.status = status
        if ts is not None:
            self.end_at = ts


class ClaudeAgentTask(Task):
    """A Task executed by a Claude SDK agent.

    Adds the SDK ``session_id`` (captured at agent_end) so destructive
    replay can locate the conversation file to attach to. The
    :attr:`parsed` view exposes the structured ``TaskID`` decode of
    the node_id (model / iteration / slot / idx).
    """

    kind: Literal["claude_agent_task"] = "claude_agent_task"
    """Wire-level discriminator (see :class:`Task.kind`)."""

    session_id: str | None = None
    """Claude SDK session_id, set on the matching ``agent_end``. Used
    by destructive replay to locate the conversation file."""

    @property
    def parsed(self) -> TaskID | None:
        """Return the TaskID parse of this node_id, or None for bespoke ids."""
        return TaskID.parse(self.node_id)

    def _apply_agent_end(self, *, session_id: str | None) -> None:
        """agent_end — capture the SDK session_id."""
        if session_id:
            self.session_id = session_id


# Tagged-union discriminator: lets pydantic dispatch dict → typed subclass
# via ``Field(discriminator="kind")`` on Module.children annotations.
from typing import Annotated  # noqa: E402

from pydantic import Field as _Field  # noqa: E402

AnyTask = Annotated[
    Task | ClaudeAgentTask,
    _Field(discriminator="kind"),
]


__all__ = ["AnyTask", "ClaudeAgentTask", "Task"]
