"""Typed message classes for Run — the v26 single-write-path bus.

Run is the canonical message source. Every state mutation goes through
``Run._on(message)`` which:

  1. dispatches the message to a domain primitive that mutates state,
  2. routes it onto its scope-owning node's ``messages`` list,
  3. fans out to registered subscribers (sinks).

Every message inherits from :class:`BaseMessage`. Two structural roots:

  - :class:`BaseMessage` — root. Carries ``type`` (wire discriminator)
    + ``text`` + the inherited :class:`AIINode` shape.
  - :class:`SummarizedMessage` — adds ``summary`` for messages that
    get an LLM-generated short-form summary post-processed in.

We only declare a dedicated subclass when the message has UNIQUE EXTRA
FIELDS beyond what its parent provides — e.g. :class:`ModuleStartMessage`
adds ``module_type`` / ``name`` / ``module_id``;
:class:`StatusPublicPublishedMessage` adds ``run_id`` / ``artifacts``.
Status messages that have no extras (``status_public_info``,
``status_public_warning``, ``status_private_debug``, …) are constructed
as :class:`BaseMessage` / :class:`SummarizedMessage` instances directly
with a string ``type``; the discriminator map in :data:`_MESSAGE_CLASSES`
picks which root they instantiate as.

The discriminator key on the wire is ``type`` (string).
``parse_message(d)`` looks up the class by ``d["type"]`` and instantiates
it. Unknown types fall back to a bare :class:`BaseMessage` so a
forward-compatible reader never crashes.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from aii_lib.timestamp import Timestamp

from .artifact import Artifact
from .node import AIINode, NodeStatus
from .node_id import NodeID


def _now_dt() -> datetime:
    return Timestamp.now().dt


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


class BaseMessage(AIINode):
    """Root of every message on the Run bus — a leaf :class:`AIINode`.

    Inherits the full AIINode shape (``node_id``, ``parent_id``,
    ``name``, ``status``, ``start_at`` / ``end_at``, ``stats``,
    ``children``, ``messages``) plus carries ``type`` (wire
    discriminator) and ``text`` (payload). ``extra="allow"`` keeps any
    facade-level metadata round-tripping (``module``, ``group``,
    ``extras``, ``metadata``, …) without each subclass having to
    re-declare the slot.

    For atomic point-in-time messages, the ``_atomic_defaults``
    validator stamps ``end_at = now()`` and ``status = DONE`` at
    construction. ``start_at`` is left ``None`` — we don't know when
    an atomic message "started", only when it was emitted.

    ``parent_id`` is REQUIRED (narrowing AIINode's optional default).
    Every message must declare its owning node up front so the
    dispatcher / sinks never have to recover the routing decision from
    type-prefix sniffing. Use ``run.node_id`` for run-level events
    (lifecycle, status diagnostics).

    Many message types have NO unique extra fields (status_*,
    agent_start, etc.) — they're constructed as bare BaseMessage /
    SummarizedMessage instances with the right ``type`` string.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    type: str
    """Wire discriminator. Concrete subclasses with unique fields pin
    via ``Literal[...]``; type-only messages just pass a string."""

    text: str = ""
    """Common payload field — the human-readable body of the message."""

    parent_id: NodeID  # type: ignore[assignment]
    """Owner node id — required at construction. Narrows the optional
    inherited from :class:`AIINode` (where ``None`` is legal because
    :class:`Run` is the tree root). For run-level events with no
    clearer scope, pass ``run.node_id``."""

    @model_validator(mode="before")
    @classmethod
    def _atomic_defaults(cls, data: Any) -> Any:
        """Stamp ``end_at`` / ``status = DONE`` and derive a deterministic id.

        Phase 2.3 contract: when a :class:`Run` is in scope and the
        caller didn't supply ``path`` (= structural emitter
        :meth:`Run._emit_path_and_id` already derived it), we compute
        ``path = "{parent.path}/{type}[{idx}]"`` from the active Run's
        per-``(parent_id, type)`` emit counter and stamp
        ``node_id = gen_path_id(type, path)``. Same Run + same emit
        sequence → same ids across runs / forks / replay re-walks, so
        DBOS step caches keyed on ``node_id`` survive replay.

        Outside a Run scope (tests / manual scripts) the AIINode
        :func:`generate_short_id` factory stands and a random id is
        used. Structural emitters that pass an explicit ``path`` keep
        their explicit identity — derivation skips when ``path`` is
        already set.
        """
        if not isinstance(data, dict):
            return data
        data.setdefault("end_at", _now_dt())
        data.setdefault("status", NodeStatus.DONE)

        if not data.get("path") and data.get("type") and data.get("parent_id"):
            from loguru import logger

            from .context import get_current_run
            from .node_id import gen_path_id

            run = get_current_run()
            if run is not None:
                try:
                    path = run._compute_emit_path(
                        parent_id=data["parent_id"],
                        name=data["type"],
                    )
                    data["path"] = path
                    data.setdefault("node_id", gen_path_id(data["type"], path))
                except Exception:
                    # Best-effort: a parent_id that doesn't resolve, a
                    # Run mid-mutation, or any other transient state
                    # must never block message construction. The default
                    # short-id factory then stands and replay loses
                    # determinism for THIS message only.
                    logger.opt(exception=True).debug(
                        f"BaseMessage._atomic_defaults: path derivation "
                        f"failed (type={data.get('type')!r}, "
                        f"parent_id={data.get('parent_id')!r})"
                    )
        return data


class SummarizedMessage(BaseMessage):
    """Carries an LLM-generated short-form summary.

    Used directly (with a string ``type``) for status messages that get
    summarized but have no other unique fields
    (``status_public_info``, ``status_public_error``). Subclassed by
    leaves that ALSO add unique fields
    (:class:`StatusPublicPublishedMessage`,
    :class:`StatusPublicInterimSummaryMessage`,
    :class:`AgentMessage` and its leaves, :class:`RunStartMessage`).
    """

    summary: str = ""


# ---------------------------------------------------------------------------
# Run-object lifecycle markers (each adds at least one unique field)
# ---------------------------------------------------------------------------


class RunStartMessage(SummarizedMessage):
    """Run start marker."""

    type: Literal["run_start"] = "run_start"
    run_id: str


class RunEndMessage(BaseMessage):
    """Run end marker."""

    type: Literal["run_end"] = "run_end"
    run_id: str
    status: str = "completed"


class RunTitleMessage(BaseMessage):
    """Short human-readable name for the run.

    Emitted once by ``TitleGeneratorSink`` after the first
    ``AgentUserPromptMessage`` flows through, when the run's name is
    still empty. ``text`` carries the LLM-generated short title; the
    dispatcher copies it onto ``run.name``.
    """

    type: Literal["run_title"] = "run_title"


class GroupStartMessage(BaseMessage):
    """Group/phase start marker."""

    type: Literal["mdgroup_start"] = "mdgroup_start"
    group_type: Literal["seq", "loop"]
    group_id: str = ""


class GroupEndMessage(BaseMessage):
    """Group/phase end marker."""

    type: Literal["mdgroup_end"] = "mdgroup_end"
    group_id: str
    status: Literal["done", "failed", "stopped"] = "done"


class IterationStartMessage(BaseMessage):
    """Loop iteration start marker."""

    type: Literal["iteration_start"] = "iteration_start"
    group_id: str
    iteration: int
    iteration_id: str = ""


class IterationEndMessage(BaseMessage):
    """Loop iteration end marker."""

    type: Literal["iteration_end"] = "iteration_end"
    group_id: str
    iteration: int
    status: Literal["done", "failed", "stopped"] = "done"


class ModuleStartMessage(BaseMessage):
    """Module start marker."""

    type: Literal["module_start"] = "module_start"
    module_type: Literal["single", "parallel"]
    name: str
    module_id: str = ""
    attach_under_id: NodeID
    """Structural parent in the tree — id of the iteration / seq-group
    the new module attaches under. Distinct from inherited
    :attr:`parent_id` which is the message's *owner* (= the module
    being started, not its parent). Dispatch reads ``attach_under_id``
    to position the new node; the slim sink reads ``parent_id`` to
    show the message under the right node in the FE feed."""


class ModuleEndMessage(BaseMessage):
    """Module end marker."""

    type: Literal["module_end"] = "module_end"
    module_id: str
    status: Literal["done", "failed", "stopped"] = "done"


class ModuleOutputMessage(SummarizedMessage):
    """Output produced by a module.

    The message lives on the owning module's ``messages`` list — i.e.
    its inherited ``parent_id`` IS the module's node_id. The inherited
    ``name`` carries the canonical module name (``"gen_strat"``,
    ``"gen_viz"``, …).

    The ``output`` field is the canonical typed payload — pipeline-side
    boot widens its annotation to a discriminated union over every
    output class (see ``aii_pipeline/run/typed_union.py``). Dispatch
    sets the owning :class:`~aii_lib.run.module.Module`'s ``output``
    attribute from this field.
    """

    type: Literal["module_output"] = "module_output"
    output: Any = None


class RunOutputMessage(SummarizedMessage):
    """Run-level aggregate output.

    Emitted once at the end of a run, carrying the run's overall
    typed result. Inherited ``parent_id`` is :attr:`Run.node_id`.
    Dispatch assigns the payload to ``run.output``.
    """

    type: Literal["run_output"] = "run_output"
    output: Any = None


class MdGroupOutputMessage(SummarizedMessage):
    """MdGroup-level (phase) aggregate output.

    Emitted at the end of a phase group (``hypo_loop``,
    ``invention_loop``, ``gen_paper_repo``, …) carrying the phase's
    aggregate typed result. Inherited ``parent_id`` IS the group's
    ``node_id``. Dispatch assigns the payload to the matching
    :class:`~aii_lib.run.mdgroup.MdGroup`'s ``output`` attribute.
    """

    type: Literal["mdgroup_output"] = "mdgroup_output"
    output: Any = None


class TaskStartMessage(BaseMessage):
    """Task start marker."""

    type: Literal["task_start"] = "task_start"
    task_id: str
    task_name: str | None = None
    module: str | None = None
    group: str | None = None
    attach_under_id: NodeID
    """Structural parent — id of the parent Module the new task
    attaches under. Distinct from inherited :attr:`parent_id` which is
    the message's owner (= the task itself). See
    :class:`ModuleStartMessage.attach_under_id` for the same split."""


class TaskEndMessage(BaseMessage):
    """Task end marker."""

    type: Literal["task_end"] = "task_end"
    task_id: str
    status: Literal["done", "failed", "stopped"] = "done"
    text: str = "OK"


class TaskOutputMessage(SummarizedMessage):
    """Task-level typed output.

    Emitted at task completion carrying the task's structured result
    (e.g. one ``Hypothesis`` from a parallel ``gen_hypo`` slot). Inherited
    ``parent_id`` IS the task's ``node_id``. Dispatch assigns the payload
    to the matching :class:`~aii_lib.run.task.Task`'s ``output`` attribute.
    """

    type: Literal["task_output"] = "task_output"
    output: Any = None


# ---------------------------------------------------------------------------
# Status messages with unique extra fields
#
# Status leaves with NO extra fields (``status_public_info``,
# ``status_public_error``, ``status_public_warning``, etc.) don't get
# their own class — :data:`_MESSAGE_CLASSES` maps their ``type`` strings
# directly to :class:`BaseMessage` or :class:`SummarizedMessage`.
# ---------------------------------------------------------------------------


class StatusPublicPublishedMessage(SummarizedMessage):
    """Run-level deliverables published mid-run."""

    type: Literal["status_public_published"] = "status_public_published"
    run_id: str = ""
    artifacts: list[Artifact] = Field(default_factory=list)


class StatusPublicInterimSummaryMessage(SummarizedMessage):
    """Periodic LLM narrative summary of a long-running task."""

    type: Literal["status_public_interim_summary"] = "status_public_interim_summary"
    task_id: str = ""


# Status leaves with no extra fields beyond their parent — each gets a
# dedicated subclass with a unique ``type`` Literal so the discriminated
# ``AnyMessage`` union can dispatch dict→subclass on seed-driven reload.


class StatusPublicInfoMessage(SummarizedMessage):
    """Operator-visible info message."""

    type: Literal["status_public_info"] = "status_public_info"


class StatusPublicErrorMessage(SummarizedMessage):
    """Operator-visible error message."""

    type: Literal["status_public_error"] = "status_public_error"


class StatusPublicWarningMessage(BaseMessage):
    """Operator-visible warning."""

    type: Literal["status_public_warning"] = "status_public_warning"


class StatusPublicProgressMessage(BaseMessage):
    """Coarse-grained progress beat."""

    type: Literal["status_public_progress"] = "status_public_progress"


class StatusPublicSuccessMessage(BaseMessage):
    """Operator-visible success."""

    type: Literal["status_public_success"] = "status_public_success"


class StatusPrivateInfoMessage(BaseMessage):
    """Diagnostic info for operators (not surfaced to end users)."""

    type: Literal["status_private_info"] = "status_private_info"


class StatusPrivateDebugMessage(BaseMessage):
    """Debug-level diagnostic."""

    type: Literal["status_private_debug"] = "status_private_debug"


# ---------------------------------------------------------------------------
# Agent family — whole family is summarization-eligible.
#
# AgentMessage is the family base (carries ``task_id`` and ``summary``).
# Used directly for ``agent_start`` (no extras beyond ``task_id``) and
# subclassed by ``AgentEndMessage`` / ``AgentRetryMessage`` which add
# unique fields.
# ---------------------------------------------------------------------------


class AgentMessage(SummarizedMessage):
    """Base for agent messages.

    Every agent message inherits ``task_id`` (the structural Task it
    brackets) and ``summary`` (LLM-generated narrative).
    """

    task_id: str = ""


class AgentStartMessage(AgentMessage):
    """Mark the start of an agent run.

    Has no extras beyond the family base — exists as a dedicated subclass
    so the ``AnyMessage`` discriminated union can dispatch ``agent_start``
    dicts to a typed instance instead of collapsing to ``AgentMessage``.
    """

    type: Literal["agent_start"] = "agent_start"


class AgentEndMessage(AgentMessage):
    """Agent end marker."""

    type: Literal["agent_end"] = "agent_end"
    session_id: str | None = None


class AgentRetryMessage(AgentMessage):
    """Agent retry marker."""

    type: Literal["agent_retry"] = "agent_retry"
    attempt: int = 0
    reason: str = ""


class AgentUserPromptMessage(AgentMessage):
    """User-authored prompt sent into a task's running agent.

    ``prompt_source`` distinguishes user-typed (``"human"``) from
    pipeline-generated (``"pipeline"``) — the to_app mapper keys
    rendering and dedup on this field.
    """

    type: Literal["agent_user_prompt"] = "agent_user_prompt"
    prompt_source: Literal["pipeline", "human"] = "pipeline"
    prompt_index: int = 0


class AgentSystemPromptMessage(AgentMessage):
    """System prompt sent at the start of an SDK session."""

    type: Literal["agent_system_prompt"] = "agent_system_prompt"
    prompt_index: int = 0


class AgentConfigMessage(AgentMessage):
    """Config snapshot emitted when an agent SDK session starts.

    Carries the model + permission + cwd + reasoning_effort so the
    activity-feed renderer can show the agent's effective settings.
    """

    type: Literal["agent_config"] = "agent_config"
    model: str | None = None
    cwd: str | None = None
    permission_mode: str | None = None
    reasoning_effort: str | None = None


class AgentResponseMessage(AgentMessage):
    """Agent's response (assistant turn) — one per SDK ``claude_msg``."""

    type: Literal["agent_response"] = "agent_response"


class AgentThinkMessage(AgentMessage):
    """Reasoning block emitted between assistant turns."""

    type: Literal["agent_think"] = "agent_think"
    signature: str | None = None


class AgentToolCallMessage(AgentMessage):
    """Tool invocation ('tool_input') from the agent SDK."""

    type: Literal["agent_tool_call"] = "agent_tool_call"
    tool: str | None = None
    tool_id: str | None = None


class AgentToolResultMessage(AgentMessage):
    """Tool result ('tool_output') from the agent SDK."""

    type: Literal["agent_tool_result"] = "agent_tool_result"
    tool: str | None = None
    tool_id: str | None = None
    is_error: bool = False


class AgentHookMessage(AgentMessage):
    """Hook callback fired by the SDK (e.g. permission, time-remaining)."""

    type: Literal["agent_hook"] = "agent_hook"
    hook_type: str | None = None


class AgentSummaryMessage(AgentMessage):
    """Aggregated summary for one agent run.

    Carries the standardized stats fields shared with
    :class:`LlmSummaryMessage`: ``total_cost``, ``input_tokens``,
    ``output_tokens``, ``cache_read_tokens``, ``cache_write_tokens``,
    ``model``, ``backend``. The dispatcher sums the three input-side
    fields (input + cache_read + cache_write) into
    :attr:`NodeStats.cum_all_input_tokens` per call. Other
    provider-specific extras (token_cost, tool_cost, num_calls,
    runtime, tool dicts, …) ride in the inherited ``extras`` slot.
    """

    type: Literal["agent_summary"] = "agent_summary"
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None
    backend: str = ""


class AgentSchemaErrorMessage(AgentMessage):
    """Tool-schema validation error from the SDK."""

    type: Literal["agent_schema_error"] = "agent_schema_error"


class AgentMessageDeltaMessage(AgentMessage):
    """Mid-stream token-usage update for the agent's CURRENT LLM call.

    Translated from the SDK's :class:`StreamEvent` whose inner
    ``event["type"] == "message_delta"``. Fires repeatedly during one
    LLM call's stream (~every 1-2 s of generation, plus once at end).

    Carries per-call usage values, NOT cumulative across calls:

      * ``input_tokens`` — uncached prompt portion, FIXED for the
        duration of the call (same value on every delta).
      * ``output_tokens`` — cumulative-within-this-call output count,
        GROWS across consecutive deltas as the model streams.
      * ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
        — also fixed per call.

    Dispatcher SETs (overwrites) these onto
    :attr:`NodeStats.current_input_tokens` / ``current_output_tokens``
    on the owning Task only — no walk-up to ancestors. ``cum_*``
    aggregates are still driven by ``agent_summary`` (per-session
    aggregate from ``ResultMessage``), not by this event.
    """

    type: Literal["agent_message_delta"] = "agent_message_delta"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


# ---------------------------------------------------------------------------
# LLM family — direct LLM calls (OpenRouter, OpenAI, Gemini, Anthropic) via
# tool_loop / research_workflow. Parallel to the agent_* family but
# distinct — agents wrap an SDK session with task_start/end ceremony,
# LLMs are point-in-time calls. Same dispatcher contract.
# ---------------------------------------------------------------------------


class LlmMessage(SummarizedMessage):
    """Family base for ``llm_*`` messages — direct LLM calls.

    Carries ``task_id`` so the dispatcher can route to the structural
    Task that initiated the call, plus the inherited ``summary`` slot
    for an LLM-generated narrative line.
    """

    task_id: str = ""


class LlmResponseMessage(LlmMessage):
    """Assistant text response from the LLM."""

    type: Literal["llm_response"] = "llm_response"
    finish_reason: str | None = None


class LlmThinkMessage(LlmMessage):
    """Reasoning content from the LLM (where supported).

    ``signature`` is set by Anthropic for cached-reasoning blocks; other
    providers leave it ``None``.
    """

    type: Literal["llm_think"] = "llm_think"
    signature: str | None = None


class LlmToolCallMessage(LlmMessage):
    """Tool invocation requested by the LLM."""

    type: Literal["llm_tool_call"] = "llm_tool_call"
    tool: str | None = None
    tool_id: str | None = None
    input: dict[str, Any] | list | str | None = None


class LlmToolResultMessage(LlmMessage):
    """Tool result returned to the LLM (in tool_loop)."""

    type: Literal["llm_tool_result"] = "llm_tool_result"
    tool: str | None = None
    tool_id: str | None = None
    output: dict[str, Any] | list | str | None = None
    is_error: bool = False


class LlmConfigMessage(LlmMessage):
    """Snapshot of the LLM call configuration — emitted once per task.

    Carries the model + provider + invocation knobs (reasoning_effort,
    temperature, …) + the structured-output schema name + tool list so
    the dashboard / replay can surface the effective settings without
    digging through later messages. Same role as
    :class:`AgentConfigMessage` for the SDK-agent backend.
    """

    type: Literal["llm_config"] = "llm_config"
    provider: str = ""
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    response_format: str | None = None
    context_window: int | None = None
    tools: list[str] = Field(default_factory=list)


class LlmSystemPromptMessage(LlmMessage):
    """System prompt text sent to the LLM."""

    type: Literal["llm_system_prompt"] = "llm_system_prompt"


class LlmUserPromptMessage(LlmMessage):
    """User prompt sent to the LLM (including retry feedback)."""

    type: Literal["llm_user_prompt"] = "llm_user_prompt"


class LlmSummaryMessage(LlmMessage):
    """Aggregated cost/token summary from a single LLM call (or tool_loop session).

    Mirrors :class:`AgentSummaryMessage` — same standardized typed
    fields (``total_cost``, ``input_tokens``, ``output_tokens``,
    ``cache_read_tokens``, ``cache_write_tokens``, ``model``,
    ``backend``); provider-specific details (token_cost, tool_cost,
    num_calls, runtime_seconds, llm_time_seconds, tool_calls,
    tool_costs, finish_reason) ride in the inherited ``extras`` slot.
    """

    type: Literal["llm_summary"] = "llm_summary"
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None
    backend: str = ""


class LlmRetryMessage(LlmMessage):
    """Schema validation failed, retrying with feedback."""

    type: Literal["llm_retry"] = "llm_retry"
    attempt: int = 0
    max_attempts: int = 0
    reason: str = ""
    schema_name: str = ""


class LlmSchemaErrorMessage(LlmMessage):
    """Schema validation exhausted — all retries failed."""

    type: Literal["llm_schema_error"] = "llm_schema_error"
    schema_name: str = ""
    validation_error: str = ""
    attempts_made: int = 0


# ---------------------------------------------------------------------------
# Discriminator → class
#
# Maps wire ``type`` strings to the class to instantiate. For status
# leaves with no extras, the entries point at :class:`BaseMessage` /
# :class:`SummarizedMessage` rather than per-leaf subclasses. For
# ``agent_start``, we point at :class:`AgentMessage` (the family base).
# ---------------------------------------------------------------------------


_MESSAGE_CLASSES: dict[str, type[BaseMessage]] = {
    # Run-object lifecycle (each has unique fields)
    "run_start": RunStartMessage,
    "run_end": RunEndMessage,
    "run_title": RunTitleMessage,
    "mdgroup_start": GroupStartMessage,
    "mdgroup_end": GroupEndMessage,
    "iteration_start": IterationStartMessage,
    "iteration_end": IterationEndMessage,
    "module_start": ModuleStartMessage,
    "module_end": ModuleEndMessage,
    "module_output": ModuleOutputMessage,
    "task_start": TaskStartMessage,
    "task_end": TaskEndMessage,
    "task_output": TaskOutputMessage,
    "run_output": RunOutputMessage,
    "mdgroup_output": MdGroupOutputMessage,
    # Status — every type string maps 1:1 to a dedicated subclass with
    # a unique ``type: Literal[...]`` so the ``AnyMessage`` discriminated
    # union can dispatch dict→subclass on seed reload.
    "status_public_published": StatusPublicPublishedMessage,
    "status_public_interim_summary": StatusPublicInterimSummaryMessage,
    "status_public_info": StatusPublicInfoMessage,
    "status_public_error": StatusPublicErrorMessage,
    "status_public_warning": StatusPublicWarningMessage,
    "status_public_progress": StatusPublicProgressMessage,
    "status_public_success": StatusPublicSuccessMessage,
    "status_private_info": StatusPrivateInfoMessage,
    "status_private_debug": StatusPrivateDebugMessage,
    # Agent — every type string has its own subclass.
    "agent_start": AgentStartMessage,
    "agent_end": AgentEndMessage,
    "agent_retry": AgentRetryMessage,
    "agent_user_prompt": AgentUserPromptMessage,
    "agent_system_prompt": AgentSystemPromptMessage,
    "agent_config": AgentConfigMessage,
    "agent_response": AgentResponseMessage,
    "agent_think": AgentThinkMessage,
    "agent_tool_call": AgentToolCallMessage,
    "agent_tool_result": AgentToolResultMessage,
    "agent_hook": AgentHookMessage,
    "agent_summary": AgentSummaryMessage,
    "agent_schema_error": AgentSchemaErrorMessage,
    "agent_message_delta": AgentMessageDeltaMessage,
    # LLM — direct LLM calls (OpenRouter / OpenAI / Gemini / Anthropic):
    "llm_response": LlmResponseMessage,
    "llm_think": LlmThinkMessage,
    "llm_tool_call": LlmToolCallMessage,
    "llm_tool_result": LlmToolResultMessage,
    "llm_config": LlmConfigMessage,
    "llm_system_prompt": LlmSystemPromptMessage,
    "llm_user_prompt": LlmUserPromptMessage,
    "llm_summary": LlmSummaryMessage,
    "llm_retry": LlmRetryMessage,
    "llm_schema_error": LlmSchemaErrorMessage,
}


def parse_message(d: dict | Any) -> BaseMessage:
    """Re-hydrate a message dict into the matching typed class.

    Looks up the class via ``d["type"]`` in :data:`_MESSAGE_CLASSES`
    and validates. Unknown ``type`` strings fall through to a bare
    :class:`BaseMessage` (``extra="allow"`` preserves any unknown
    fields).
    """
    if not isinstance(d, dict):
        raise TypeError(f"parse_message: expected dict, got {type(d).__name__}")
    cls = _MESSAGE_CLASSES.get(d.get("type", ""), BaseMessage)
    return cls.model_validate(d)


# ---------------------------------------------------------------------------
# ``AnyMessage`` — discriminated union over every known message subclass.
#
# Each member of :data:`_MESSAGE_CLASSES` has a unique
# ``type: Literal[...]``, so pydantic dispatches dict→subclass on
# ``model_validate`` via the ``type`` discriminator. Unknown ``type``
# strings fall through to a callable :class:`Discriminator` that routes
# them to ``BaseMessage`` so a future build emitting a new event type
# can still be deserialized on a not-yet-upgraded reader.
#
# This is the standard pydantic-v2 tagged-union pattern — same shape
# as ``AnyModule`` / ``AnyMdGroup`` / ``AnyTask``. No runtime
# ``__annotations__`` patching, no BeforeValidator, no force-rebuild
# choreography: just a concrete annotation that ``model_validate``
# walks naturally.
# ---------------------------------------------------------------------------
AIINode.model_rebuild()  # resolve forward-ref ``list[BaseMessage]``

from typing import Annotated as _Annotated  # noqa: E402

from pydantic import Discriminator as _Discriminator  # noqa: E402
from pydantic import Tag as _Tag  # noqa: E402


def _message_discriminator(v: Any) -> str:
    """Pick the union tag for a message dict / instance.

    Known ``type`` strings → that string (so the matching subclass
    fires). Unknown / missing → ``"_unknown"`` → :class:`BaseMessage`.
    """
    if isinstance(v, dict):
        t = v.get("type")
    else:
        t = getattr(v, "type", None)
    return t if isinstance(t, str) and t in _MESSAGE_CLASSES else "_unknown"


_message_union_members: list = [
    _Annotated[cls, _Tag(type_str)]  # ty: ignore[invalid-type-form]
    for type_str, cls in _MESSAGE_CLASSES.items()
]
_message_union_members.append(_Annotated[BaseMessage, _Tag("_unknown")])

# Build the union via ``Union[args]`` from typing — pep604 ``|`` chain
# breaks on ``Annotated`` operands, and a linter strips ``Union[tuple(...)]``
# as "redundant". ``Union.__getitem__`` accepts a tuple directly.
import typing as _typing  # noqa: E402

_member_union = _typing.Union[tuple(_message_union_members)]  # type: ignore[valid-type]  # noqa: UP007

AnyMessage = _Annotated[
    _member_union,  # ty: ignore[invalid-type-form]
    _Discriminator(_message_discriminator),
]
"""Discriminated union over every known message subclass. Element type
of :attr:`AIINode.messages` after :func:`bind_message_union` runs, so
``model_validate`` reloads typed instances on seed-driven rehydrate."""


_MESSAGE_UNION_BOUND = False


def bind_message_union() -> None:
    """Wire AnyMessage type annotation on AIINode.messages.

    Idempotent. Must run AFTER every AIINode subclass is imported (Run /
    MdGroup / Module / Task / pipeline phase + substep subclasses) so the
    walk reaches them all. ``aii_pipeline.run.typed_union`` calls this.

    Mirrors the ``children`` / ``output`` widening pattern used by
    ``bind_pipeline_typed_unions``: assign ``model_fields[...].annotation``
    to the new (concrete) discriminated union, then ``model_rebuild(force=
    True)`` to regenerate the compiled validator. ``force=True`` is safe —
    the new annotation is concrete (no forward refs), so re-resolution from
    class-body wouldn't change it.
    """
    global _MESSAGE_UNION_BOUND
    if _MESSAGE_UNION_BOUND:
        return

    msg_ann = list[AnyMessage]

    seen: set[type] = set()
    stack: list[type] = [AIINode]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if "messages" in cls.model_fields:
            cls.model_fields["messages"].annotation = msg_ann
        stack.extend(cls.__subclasses__())

    # Bottom-up — parents' compiled schemas embed children's, so leaves
    # rebuild first.
    for cls in sorted(seen, key=lambda c: -len(c.__mro__)):
        cls.model_rebuild(force=True)

    _MESSAGE_UNION_BOUND = True


__all__ = [
    # Roots
    "BaseMessage",
    "SummarizedMessage",
    # Run-object lifecycle
    "RunStartMessage",
    "RunEndMessage",
    "RunTitleMessage",
    "GroupStartMessage",
    "GroupEndMessage",
    "IterationStartMessage",
    "IterationEndMessage",
    "ModuleStartMessage",
    "ModuleEndMessage",
    "ModuleOutputMessage",
    "TaskStartMessage",
    "TaskEndMessage",
    "TaskOutputMessage",
    "RunOutputMessage",
    "MdGroupOutputMessage",
    # Status with extras
    "StatusPublicPublishedMessage",
    "StatusPublicInterimSummaryMessage",
    # Agent family
    "AgentMessage",
    "AgentEndMessage",
    "AgentRetryMessage",
    "AgentUserPromptMessage",
    "AgentSystemPromptMessage",
    "AgentConfigMessage",
    "AgentResponseMessage",
    "AgentThinkMessage",
    "AgentToolCallMessage",
    "AgentToolResultMessage",
    "AgentHookMessage",
    "AgentSummaryMessage",
    "AgentSchemaErrorMessage",
    "AgentMessageDeltaMessage",
    # Llm family (raw LLM-level messages — peer to the Agent family)
    "LlmMessage",
    "LlmResponseMessage",
    "LlmThinkMessage",
    "LlmToolCallMessage",
    "LlmToolResultMessage",
    "LlmConfigMessage",
    "LlmSystemPromptMessage",
    "LlmUserPromptMessage",
    "LlmSummaryMessage",
    "LlmRetryMessage",
    "LlmSchemaErrorMessage",
    # Discriminated union + boot binder
    "AnyMessage",
    "bind_message_union",
    # Helpers
    "parse_message",
    # Discriminator
    "_MESSAGE_CLASSES",
]
