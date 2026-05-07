"""OpenTelemetry sink for the Run bus.

Two pipelines, one facade. The facade :class:`OTelRunSink` (in
``sink.py``) wires up traces and metrics independently and dispatches
each ``BaseMessage`` to both:

  * :mod:`.traces` — :class:`TraceHandlers`, :func:`build_tracer_provider`,
    :class:`JSONLSpanExporter`. Span hierarchy mirrors the Run-object
    tree: run → mdgroup → iteration → module → task, with per-message
    child spans (paired tool_call/result, zero-duration markers for
    everything else).
  * :mod:`.metrics` — :func:`build_meter_provider`, :class:`JSONLMetricExporter`.
    Registers one :class:`ObservableGauge` per (node, stats field)
    named ``aii.{name}_{node_id}.{metric_suffix}``. A daemon thread
    re-walks the tree every 2 s to register gauges for late-appearing
    nodes (tasks that materialize on iteration_start). Pure
    periodic-snapshot — no per-event work in the sink.

Optional: requires the ``opentelemetry`` family on the import path.
The ``__init__`` swallows the import so environments without OTel
(skill servers, ability venv) still import the package.
"""

from __future__ import annotations

try:
    from .sink import OTelRunSink
except ImportError:
    OTelRunSink = None  # type: ignore[assignment,misc]


__all__ = ["OTelRunSink"]
