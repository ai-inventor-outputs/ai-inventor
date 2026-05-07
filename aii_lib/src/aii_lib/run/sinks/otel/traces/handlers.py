"""Trace handlers — translate Run-bus events into nested OTel spans.

Span hierarchy mirrors the Run-object tree:

    Run span (zero-duration root, exported at run_start)
    └── MdGroup span (mdgroup_start → mdgroup_end)
        └── Iteration span (iteration_start → iteration_end)
            └── Module span (module_start → module_end)
                └── Task span (task_start → task_end)
                    └── per-message child spans (see below)
    [+ ``run_complete`` marker emitted at run_end]

The root run span ends *immediately* on ``run_start``. Tempo derives
the trace name from the root span, and SimpleSpanProcessor only
exports spans on ``.end()`` — a long-lived root would leave the trace
unnamed in trace search until run_end. Ending early gives Tempo the
trace name within seconds; descendants still parent against the (now
ended) span via trace_id, since OTel allows context references after
``.end()``. The Run object's recorded runtime is the source of truth
for actual duration; the root span's duration is intentionally zero.

A ``run_complete`` marker is emitted at run_end carrying the terminal
status (attribute mutation on an already-ended span is a no-op).

Every span carries its full ancestor chain as attributes
(``aii.run_id``, ``aii.group_id``, ``aii.group``, ``aii.iteration``,
``aii.module_id``, ``aii.module``, ``aii.task_id``, ``aii.task_name``)
so partial trace data from a crashed run remains queryable by ancestor
identity even when the parent span never closes cleanly. Lineage is
gathered from ``_scope_attrs``, a parallel dict keyed identically to
``_spans``; each scope contributes only its own keys (no overlaps).

Span naming convention: every lifecycle / tool span name is
``{name}_{id}`` — display name (or level keyword for run/iter) joined
with the structural id. ``_spanname`` collapses to ``{id}`` alone when
the name is empty or duplicates the id, so we never emit ``foo_foo``.

Per-message child spans inside a task:

  * ``agent_tool_call`` / ``llm_tool_call`` — stashed in
    ``_pending_tool_calls`` keyed by ``(task_id, tool_id)``.
  * ``agent_tool_result`` / ``llm_tool_result`` — looks up the matching
    pending call and emits ONE span ``{tool}_{tool_id}`` covering the
    call → result interval. Unmatched results emit a zero-duration span.
  * ``agent_summary`` / ``llm_summary`` — decorate the active task span
    with ``gen_ai.usage.*`` + ``aii.cost_usd`` attributes; no new span.
  * Anything else (status_*, agent_response, agent_think, agent_config,
    agent_user_prompt, …) — emit a zero-duration child span on the
    deepest open scope. They appear as visible markers on the trace
    waterfall while keeping correlation through the parent task.

Orphan tool calls (call without a result by task_end) are emitted as
zero-duration spans with status ERROR ``no result received`` so they
don't disappear from the trace.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, StatusCode

if TYPE_CHECKING:
    from opentelemetry.metrics import Histogram
    from opentelemetry.trace import Span, Tracer


# Message types that are paired (call → result). Pairing key = ``(task_id, tool_id)``.
_TOOL_CALL_TYPES = {"agent_tool_call", "llm_tool_call"}
_TOOL_RESULT_TYPES = {"agent_tool_result", "llm_tool_result"}

# Summary message types — decorate active task span, no new span.
_SUMMARY_TYPES = {"agent_summary", "llm_summary"}


def _spanname(name: str | None, ident: str | None) -> str:
    """Canonical span name = ``{name}_{ident}``.

    Falls back to ``ident`` alone when ``name`` is empty or already equals
    ``ident`` — avoids ugly ``foo_foo`` when the upstream message lacks a
    distinct display name.
    """
    n = (name or "").strip()
    i = (ident or "").strip() or "unknown"
    if not n or n == i:
        return i
    return f"{n}_{i}"


class TraceHandlers:
    """Stateful span manager — owns the open-span dict + tool-call pairing.

    One instance per :class:`OTelRunSink`. Methods are called from
    ``OTelRunSink.flush`` per typed event. Thread-safe (single ``_lock``
    guards both the span dict and the pairing dict).
    """

    def __init__(
        self,
        tracer: Tracer,
        *,
        task_duration_histogram: Histogram | None = None,
    ) -> None:
        self._tracer = tracer
        self._task_duration_histogram = task_duration_histogram
        self._lock = threading.Lock()
        # Open lifecycle spans, keyed by scope token (e.g. "run", "group:hypo_loop",
        # "iter:hypo_loop:1", "module:gen_hypo", "task:gen_hypo_it1__haiku").
        # The ``run`` entry's span is zero-duration (ended at run_start) but
        # kept here so child handlers can still use it as parent context.
        self._spans: dict[str, Span] = {}
        # Identity attrs for each currently-open scope, parallel to
        # ``_spans``. ``_lineage()`` merges these into every newly-created
        # span so each carries its full ancestor chain — keeps spans
        # queryable by ancestor identity even on partial/crashed runs.
        self._scope_attrs: dict[str, dict[str, Any]] = {}
        # Pending tool calls awaiting their results, keyed by (task_id, tool_id).
        # Value: (call_msg_dict, call_span_attrs)
        self._pending_tool_calls: dict[tuple[str, str], dict] = {}
        # Task start timestamps (ns), keyed by task_id — used by ``on_task_end``
        # to compute the duration recorded into the latency histogram.
        self._task_start_ns: dict[str, int] = {}

    # ── Lifecycle handlers ────────────────────────────────────────────

    def on_run_start(self, msg: dict) -> None:
        """Create and end the root span; emit it immediately for trace naming."""
        # Idempotent — the sink may eagerly open the run span at
        # construction (because aii_server emits run_start upstream of the
        # pipeline subprocess, so it never reaches us). If it's already
        # open, this is a no-op.
        with self._lock:
            if "run" in self._spans:
                return
        run_id = msg.get("run_id") or "run"
        # ``context=Context()`` forces the run to be a true root span,
        # ignoring any ambient OTel context that an outer caller (e.g.
        # FastAPI auto-instrumentation) may have set.
        span = self._tracer.start_span(
            _spanname("run", run_id),
            context=Context(),
            attributes={"aii.type": "run", "aii.run_id": run_id},
        )
        # End the root immediately. Tempo names the trace from the root
        # span, so until it's exported the trace shows up nameless — even
        # though child spans are streaming in. With SimpleSpanProcessor the
        # export happens within ms of .end(). Descendants still link to it
        # via trace_id; OTel accepts ended spans as parent contexts.
        span.end()
        with self._lock:
            if "run" in self._spans:
                return  # raced — discard ours (orphan zero-duration span)
            self._spans["run"] = span
            self._scope_attrs["run"] = {"aii.run_id": run_id}

    def on_run_end(self, msg: dict) -> None:
        """Emit a run_complete marker with terminal status."""
        run_id = msg.get("run_id") or "run"
        status_text = msg.get("status", "") or msg.get("text", "") or ""
        with self._lock:
            span = self._spans.pop("run", None)
            self._scope_attrs.pop("run", None)
        if span is None:
            return
        # Root span was ended at run_start; attribute mutation is a no-op
        # on ended spans. Emit a sibling marker carrying the terminal
        # status so it's still queryable in Tempo.
        parent_ctx = trace.set_span_in_context(span)
        completion = self._tracer.start_span(
            "run_complete",
            context=parent_ctx,
            attributes={
                "aii.type": "run_complete",
                "aii.run_id": run_id,
                "aii.status": status_text,
            },
        )
        self._set_terminal_status(completion, status_text)
        completion.end()

    def on_group_start(self, msg: dict) -> None:
        """Open a group span keyed by group_id."""
        # Key by ``group_id`` (stable across start/end). ``text`` carries
        # the display name on _start but is rewritten by the dispatcher
        # to a "Group X | Cost: ... | ..." summary on _end — so any
        # text-based key would mismatch and orphan the span.
        group_id = msg.get("group_id") or ""
        name = msg.get("text", "") or ""
        parent_ctx = self._parent_context_of("run")
        span = self._tracer.start_span(
            _spanname(name, group_id),
            context=parent_ctx,
            attributes={
                **self._lineage(),
                "aii.type": "group",
                "aii.group": name or group_id,
                "aii.group_id": group_id,
            },
        )
        with self._lock:
            self._spans[f"group:{group_id}"] = span
            self._scope_attrs[f"group:{group_id}"] = {
                "aii.group_id": group_id,
                "aii.group": name or group_id,
            }

    def on_group_end(self, msg: dict) -> None:
        """Close and end a group span."""
        group_id = msg.get("group_id") or ""
        with self._lock:
            span = self._spans.pop(f"group:{group_id}", None)
            self._scope_attrs.pop(f"group:{group_id}", None)
        if span:
            span.set_status(StatusCode.OK)
            span.end()

    def on_iteration_start(self, msg: dict) -> None:
        """Open an iteration span."""
        group_id = msg.get("group_id") or ""
        iteration = msg.get("iteration", 0)
        # Parent group span is keyed by ``group_id`` (post-fix).
        parent_ctx = self._parent_context_of("group", group_id) or self._parent_context_of("group")
        span = self._tracer.start_span(
            f"iter_{iteration}",
            context=parent_ctx,
            attributes={
                **self._lineage(),
                "aii.type": "iteration",
                "aii.iteration": iteration,
            },
        )
        with self._lock:
            self._spans[f"iter:{group_id}:{iteration}"] = span
            self._scope_attrs[f"iter:{group_id}:{iteration}"] = {
                "aii.iteration": iteration,
            }

    def on_iteration_end(self, msg: dict) -> None:
        """Close and end an iteration span."""
        group_id = msg.get("group_id") or ""
        iteration = msg.get("iteration", 0)
        with self._lock:
            span = self._spans.pop(f"iter:{group_id}:{iteration}", None)
            self._scope_attrs.pop(f"iter:{group_id}:{iteration}", None)
        if span:
            span.set_status(StatusCode.OK)
            span.end()

    def on_module_start(self, msg: dict) -> None:
        """Open a module span."""
        # Key by ``module_id`` (stable across start/end). ``text`` carries
        # the display name on _start but is rewritten to a summary on _end.
        module_id = msg.get("module") or ""
        name = msg.get("text", "") or ""
        # Prefer the deepest open iteration as parent; fall back to group.
        parent_ctx = self._parent_context_of("iter") or self._parent_context_of("group")
        span = self._tracer.start_span(
            _spanname(name, module_id),
            context=parent_ctx,
            attributes={
                **self._lineage(),
                "aii.type": "module",
                "aii.module": name or module_id,
                "aii.module_id": module_id,
            },
        )
        with self._lock:
            self._spans[f"module:{module_id}"] = span
            self._scope_attrs[f"module:{module_id}"] = {
                "aii.module_id": module_id,
                "aii.module": name or module_id,
            }

    def on_module_end(self, msg: dict) -> None:
        """Close and end a module span."""
        module_id = msg.get("module") or ""
        with self._lock:
            span = self._spans.pop(f"module:{module_id}", None)
            self._scope_attrs.pop(f"module:{module_id}", None)
        if span:
            span.set_status(StatusCode.OK)
            span.end()

    def on_task_start(self, msg: dict) -> None:
        """Open a task span and record the start timestamp."""
        task_id = msg.get("task_id") or msg.get("run_id") or msg.get("tool_id") or "task"
        task_name = msg.get("task_name") or ""
        module = msg.get("module")
        parent_ctx = self._parent_context_of("module", module) or self._parent_context_of("module")
        span = self._tracer.start_span(
            _spanname(task_name, task_id),
            context=parent_ctx,
            attributes={
                **self._lineage(),
                "aii.type": "task",
                "aii.task_id": task_id,
                "aii.task_name": task_name or task_id,
            },
        )
        start_ns = _to_unix_nanos(msg.get("start_at")) or _to_unix_nanos(msg.get("end_at"))
        with self._lock:
            self._spans[f"task:{task_id}"] = span
            self._scope_attrs[f"task:{task_id}"] = {
                "aii.task_id": task_id,
                "aii.task_name": task_name or task_id,
            }
            if start_ns is not None:
                self._task_start_ns[task_id] = start_ns

    def on_task_end(self, msg: dict) -> None:
        """Close task span and record task duration in histogram."""
        task_id = msg.get("task_id") or msg.get("run_id") or msg.get("tool_id") or "task"
        status_text = msg.get("text", "") or ""
        # Sweep any orphan tool calls for this task — emit zero-duration
        # spans with ERROR status so they appear in the waterfall.
        self._flush_orphan_tool_calls(task_id)
        with self._lock:
            span = self._spans.pop(f"task:{task_id}", None)
            self._scope_attrs.pop(f"task:{task_id}", None)
            start_ns = self._task_start_ns.pop(task_id, None)
        if span:
            self._set_terminal_status(span, status_text)
            span.end()
        # Record the latency histogram independently of span lifecycle —
        # metrics survive even when the span itself was force-closed by
        # ``close_open_spans``. Source of truth for duration is
        # ``end_at - start_at`` rather than the span's own clocks
        # (those default to wall clock at handler call time).
        end_ns = _to_unix_nanos(msg.get("end_at"))
        if (
            self._task_duration_histogram is not None
            and start_ns is not None
            and end_ns is not None
            and end_ns >= start_ns
        ):
            duration_s = (end_ns - start_ns) / 1e9
            # Low-cardinality labels only — ``module`` and ``task_name``
            # are bounded by the pipeline definition, so series count
            # stays manageable. ``task_id`` is intentionally excluded.
            self._task_duration_histogram.record(
                duration_s,
                attributes={
                    "aii.module": msg.get("module") or "",
                    "aii.task_name": msg.get("task_name") or "",
                },
            )

    # ── Per-message handlers ──────────────────────────────────────────

    def on_summary(self, msg: dict) -> None:
        """Decorate the active task span with cost / token / model attrs."""
        task_id = msg.get("task_id") or msg.get("run_id")
        if not task_id:
            return
        with self._lock:
            span = self._spans.get(f"task:{task_id}")
        if not span:
            return
        meta = msg.get("extras") or msg
        cost = meta.get("total_cost", 0) or msg.get("total_cost", 0) or 0
        if cost:
            span.set_attribute("aii.cost_usd", float(cost))
        # OTel GenAI semconv attrs — Tempo/derived RED metrics + the
        # service-graph view recognise these names. Only set when the
        # source field is non-empty so we don't pollute spans that
        # don't carry that info.
        for src_key, otel_key in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
            ("model", "gen_ai.request.model"),
            ("response_model", "gen_ai.response.model"),
            ("system", "gen_ai.system"),
            ("operation", "gen_ai.operation.name"),
            ("response_id", "gen_ai.response.id"),
            ("temperature", "gen_ai.request.temperature"),
            ("max_tokens", "gen_ai.request.max_tokens"),
        ):
            v = meta.get(src_key, msg.get(src_key))
            if v not in (None, 0, ""):
                span.set_attribute(otel_key, v)
        # ``finish_reasons`` is spec'd as a list — accept either a string
        # or list source and normalise. Skip when missing/empty.
        fr = meta.get("finish_reason") or msg.get("finish_reason")
        if fr:
            span.set_attribute(
                "gen_ai.response.finish_reasons",
                [fr] if isinstance(fr, str) else list(fr),
            )
        rt = meta.get("runtime_seconds", 0)
        if rt:
            span.set_attribute("aii.runtime_seconds", float(rt))

    def on_tool_call(self, msg: dict) -> None:
        """Stash the call until its matching result arrives."""
        task_id = msg.get("task_id") or msg.get("run_id") or ""
        tool_id = msg.get("tool_id") or ""
        if not task_id or not tool_id:
            # Can't pair without both keys — fall back to a zero-duration
            # span so the call still appears in the trace.
            self._emit_zero_duration_span(msg)
            return
        with self._lock:
            self._pending_tool_calls[(task_id, tool_id)] = msg

    def on_tool_result(self, msg: dict) -> None:
        """Pair with the pending call → emit one span covering call→result."""
        task_id = msg.get("task_id") or msg.get("run_id") or ""
        tool_id = msg.get("tool_id") or ""
        with self._lock:
            call_msg = self._pending_tool_calls.pop((task_id, tool_id), None)
        if not call_msg:
            # Unmatched result — zero-duration span at result.end_at.
            self._emit_zero_duration_span(msg)
            return
        # Build a span covering call.end_at → result.end_at. Both are
        # ISO-8601 strings on the wire; convert to nanoseconds.
        call_ts = _to_unix_nanos(call_msg.get("end_at"))
        result_ts = _to_unix_nanos(msg.get("end_at"))
        # Defensive: if either timestamp is missing, fall back to "now".
        if call_ts is None or result_ts is None or result_ts < call_ts:
            self._emit_zero_duration_span(msg)
            return
        parent_ctx = self._parent_context_of("task", task_id)
        tool_name = msg.get("tool") or call_msg.get("tool") or ""
        span = self._tracer.start_span(
            _spanname(tool_name, tool_id),
            context=parent_ctx,
            kind=SpanKind.CLIENT,
            start_time=call_ts,
            attributes={
                **self._lineage(),
                "aii.type": "tool_call",
                "aii.tool": tool_name or "tool",
                "aii.tool_id": tool_id,
                "aii.task_id": task_id,
            },
        )
        if msg.get("is_error"):
            err_text = (msg.get("text") or "tool returned is_error=true")[:1000]
            span.set_status(StatusCode.ERROR, err_text[:200])
            # Synthetic exception event — we don't have a Python exception
            # to ``record_exception`` here, but the spec encourages
            # surfacing failure detail via the same semantic attrs so
            # Tempo's "errors" panel surfaces them.
            span.add_event(
                "exception",
                attributes={
                    "exception.type": "ToolError",
                    "exception.message": err_text,
                },
            )
        else:
            span.set_status(StatusCode.OK)
        span.end(end_time=result_ts)

    def on_other(self, msg: dict) -> None:
        """Default fallthrough — emit a zero-duration child span on the deepest open scope."""
        self._emit_zero_duration_span(msg)

    # ── Shutdown ──────────────────────────────────────────────────────

    def close_open_spans(self) -> None:
        """Force-end any spans still open at sink shutdown.

        Calling ``.end()`` on the already-ended root run span is a no-op
        in the OTel SDK, so this is safe even though run is zero-duration
        by design.
        """
        with self._lock:
            for span in self._spans.values():
                try:
                    span.end()
                except Exception:
                    pass
            self._spans.clear()
            self._scope_attrs.clear()
            self._pending_tool_calls.clear()

    # ── Helpers ───────────────────────────────────────────────────────

    def _lineage(self) -> dict[str, Any]:
        """Snapshot of identity attrs from every currently-open scope.

        Merged into every newly-created span so the span carries its
        full ancestor chain (run_id, group_id, group, iteration,
        module_id, module, task_id, task_name as applicable). Each
        scope contributes only its own keys, so merging is conflict-free.

        Crash-safe: if a parent span never ends, descendants still carry
        the parent's identity as attributes — queryable via TraceQL like
        ``{ aii.run_id="X" && aii.module_id="Y" }`` even on partial traces.
        """
        out: dict[str, Any] = {}
        with self._lock:
            for attrs in self._scope_attrs.values():
                out.update(attrs)
        return out

    def _parent_context_of(self, prefix: str, name: str | None = None) -> object | None:
        """Return an OTel context whose active span.

        Deepest open span matching ``prefix:`` (optionally exact-name
        ``prefix:<name>``).
        """
        with self._lock:
            parent: Span | None = None
            if name is not None:
                parent = self._spans.get(f"{prefix}:{name}")
            if parent is None:
                # Walk back through insertion order for the deepest match.
                for key in reversed(list(self._spans)):
                    head = key.split(":", 1)[0]
                    if head == prefix:
                        parent = self._spans[key]
                        break
        if parent is None:
            return None
        return trace.set_span_in_context(parent)

    def _deepest_open_scope(self) -> Span | None:
        """Return the deepest-open lifecycle span for fallback parent.

        Deepest-currently-open lifecycle span (task > module > iter > group >
        run), used as parent for fallback zero-duration spans. The dict
        insertion order doubles as nesting order because nested scopes
        always open after their parents.
        """
        with self._lock:
            if not self._spans:
                return None
            return next(reversed(self._spans.values()))

    def _emit_zero_duration_span(self, msg: dict) -> None:
        """Emit a zero-duration fallback span as a trace marker.

        Fallback: emit a span with start == end == msg.end_at, parented
        on the deepest open scope. Visually a marker on the trace timeline.
        """
        parent = self._deepest_open_scope()
        if parent is None:
            return
        ts = _to_unix_nanos(msg.get("end_at")) or _to_unix_nanos(msg.get("start_at"))
        msg_type = msg.get("type", "event")
        attrs: dict[str, Any] = {
            **self._lineage(),
            "aii.type": "marker",
            "aii.event_type": msg_type,
        }
        for k in ("module", "task_id", "run_id", "tool", "tool_id", "source", "name"):
            v = msg.get(k)
            if isinstance(v, (str, int, float, bool)) and v not in ("", None):
                attrs[f"aii.{k}"] = v
        text = msg.get("text") or ""
        if text:
            attrs["aii.text"] = text[:1000]
        ctx = trace.set_span_in_context(parent)
        kwargs: dict[str, Any] = {"context": ctx, "attributes": attrs}
        if ts is not None:
            kwargs["start_time"] = ts
        span = self._tracer.start_span(msg_type, **kwargs)
        if msg.get("is_error"):
            err_msg = text[:1000] if text else "is_error"
            span.set_status(StatusCode.ERROR, err_msg[:200])
            span.add_event(
                "exception",
                attributes={
                    "exception.type": "MarkerError",
                    "exception.message": err_msg,
                },
            )
        else:
            span.set_status(StatusCode.OK)
        if ts is not None:
            span.end(end_time=ts)
        else:
            span.end()

    def _flush_orphan_tool_calls(self, task_id: str) -> None:
        """Emit zero-duration ERROR spans for any unmatched calls on ``task_id``."""
        with self._lock:
            orphans = [(k, v) for k, v in self._pending_tool_calls.items() if k[0] == task_id]
            for k in [k for k, _ in orphans]:
                self._pending_tool_calls.pop(k, None)
        for (_, tool_id), call_msg in orphans:
            ts = _to_unix_nanos(call_msg.get("end_at"))
            parent_ctx = self._parent_context_of("task", task_id)
            tool_name = call_msg.get("tool") or ""
            span = self._tracer.start_span(
                _spanname(tool_name, tool_id),
                context=parent_ctx,
                kind=SpanKind.CLIENT,
                start_time=ts,
                attributes={
                    **self._lineage(),
                    "aii.type": "tool_call",
                    "aii.tool": tool_name or "tool",
                    "aii.tool_id": tool_id,
                    "aii.task_id": task_id,
                    "aii.orphan": True,
                },
            )
            span.set_status(StatusCode.ERROR, "no result received before task_end")
            span.add_event(
                "exception",
                attributes={
                    "exception.type": "OrphanToolCall",
                    "exception.message": "no result received before task_end",
                },
            )
            if ts is not None:
                span.end(end_time=ts)
            else:
                span.end()

    @staticmethod
    def _set_terminal_status(span: Span, status_text: str) -> None:
        s = (status_text or "").lower()
        if any(t in s for t in ("error", "fail", "interrupt", "stop", "paused")):
            span.set_status(StatusCode.ERROR, status_text)
            # Synthetic exception event so Tempo's "errors" panel surfaces
            # the terminal status text. ``record_exception`` requires a
            # real Exception object; we only have a status string.
            span.add_event(
                "exception",
                attributes={
                    "exception.type": "TerminalError",
                    "exception.message": (status_text or "(no message)")[:1000],
                },
            )
        else:
            span.set_status(StatusCode.OK)


def _to_unix_nanos(ts: object) -> int | None:
    """Coerce an event timestamp (ISO string, datetime, or unix ns int) to ns."""
    if ts is None:
        return None
    if isinstance(ts, int):
        # Heuristic: > 1e15 → already ns; otherwise seconds.
        return ts if ts > 10**15 else int(ts * 1e9)
    if isinstance(ts, float):
        return int(ts * 1e9)
    if isinstance(ts, str):
        from datetime import datetime

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1e9)
        except ValueError:
            return None
    # datetime
    try:
        return int(ts.timestamp() * 1e9)
    except Exception:
        return None


# Dispatch table — maps event.type → bound method on a TraceHandlers instance.
# Used by the sink facade. Methods not in this table fall through to ``on_other``.
HANDLER_BY_TYPE = {
    "run_start": TraceHandlers.on_run_start,
    "run_end": TraceHandlers.on_run_end,
    "mdgroup_start": TraceHandlers.on_group_start,
    "mdgroup_end": TraceHandlers.on_group_end,
    "iteration_start": TraceHandlers.on_iteration_start,
    "iteration_end": TraceHandlers.on_iteration_end,
    "module_start": TraceHandlers.on_module_start,
    "module_end": TraceHandlers.on_module_end,
    "task_start": TraceHandlers.on_task_start,
    "task_end": TraceHandlers.on_task_end,
    "agent_summary": TraceHandlers.on_summary,
    "llm_summary": TraceHandlers.on_summary,
    "agent_tool_call": TraceHandlers.on_tool_call,
    "llm_tool_call": TraceHandlers.on_tool_call,
    "agent_tool_result": TraceHandlers.on_tool_result,
    "llm_tool_result": TraceHandlers.on_tool_result,
}


__all__ = ["HANDLER_BY_TYPE", "TraceHandlers"]
