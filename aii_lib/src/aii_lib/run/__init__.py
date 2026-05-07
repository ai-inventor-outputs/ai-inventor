"""aii_lib.run — generic OO machinery for pipeline runs.

In v26 the tree is::

    Run > MdGroup (Loop|Seq) > [LoopIteration >] Module > Task

Every node inherits :class:`AIINode` and carries the same identity
(``node_id``), lifecycle (``status`` / ``start_at`` / ``end_at``),
aggregate (``stats``), structure (``children``), and event log
(``messages``) shape. Subclasses add their type-specific fields.

This package is pipeline-agnostic. It provides the OO scaffold that
any agent pipeline can embed. UI labels (display names, card
grouping) live in the embedding pipeline's mapper layer, not in the
domain.
"""

from __future__ import annotations

from typing import Literal

from .context import (
    current_run,
    get_current_run,
    set_current_run,
)
from .loop_iteration import LoopIteration
from .mdgroup import AnyMdGroup, LoopMdGroup, MdGroup, SeqMdGroup
from .messages import AnyMessage, BaseMessage
from .module import AnyModule, Module, ParallelTModule, SingleTModule
from .node import AIINode, NodeStats, NodeStatus
from .run import (
    Run,
    set_dispatch,
    set_ensure_for_task,
)
from .sink import RunSink
from .task import AnyTask, ClaudeAgentTask, Task

# ---------------------------------------------------------------------------
# Resolve forward refs on every AIINode subclass.
#
# AIINode declares ``messages: list[BaseMessage]`` but BaseMessage is defined
# in ``.messages`` which itself imports AIINode — a cycle broken by gating
# the BaseMessage import on TYPE_CHECKING. After ``.messages`` is loaded
# above, each AIINode subclass needs ``.model_rebuild()`` so its inherited
# ``messages`` field annotation resolves to the concrete BaseMessage
# class. Pydantic does NOT cascade rebuilds from parent → subclass; each
# concrete class is rebuilt explicitly here. Passing the namespace
# explicitly avoids relying on ``sys.modules`` lookups that miss across
# subclass module boundaries.
# ---------------------------------------------------------------------------
_types_ns = {"BaseMessage": BaseMessage, "Literal": Literal}
for _cls in (
    AIINode,
    LoopIteration,
    MdGroup,
    SeqMdGroup,
    LoopMdGroup,
    Module,
    SingleTModule,
    ParallelTModule,
    Run,
    Task,
    ClaudeAgentTask,
):
    _cls.model_rebuild(_types_namespace=_types_ns, force=True)


__all__ = [
    # Core hierarchy
    "AIINode",
    "Task",
    "ClaudeAgentTask",
    "AnyTask",
    "Module",
    "SingleTModule",
    "ParallelTModule",
    "AnyModule",
    "MdGroup",
    "SeqMdGroup",
    "LoopMdGroup",
    "AnyMdGroup",
    "LoopIteration",
    "Run",
    # Messages
    "BaseMessage",
    "AnyMessage",
    # Status + aggregates
    "NodeStatus",
    "NodeStats",
    # Channel contracts
    "RunSink",
    # Context vars
    "current_run",
    "get_current_run",
    "set_current_run",
    # Pluggable hook setters
    "set_dispatch",
    "set_ensure_for_task",
]
