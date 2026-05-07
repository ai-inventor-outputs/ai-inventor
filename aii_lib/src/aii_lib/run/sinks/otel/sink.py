"""OTelRunSink — facade that wires the trace + metric pipelines.

Keeps the single :class:`RunSink` contract exposed to ``Run`` (one
``flush(event)`` entry point + ``close()``) while delegating the actual
work to two independent pipelines that live in sibling subpackages:

  * :mod:`.traces` — :class:`TraceHandlers` + :func:`build_tracer_provider`.
    Mirrors the Run-object tree: run → mdgroup → iteration → module →
    task spans, plus per-message child spans (paired tool_call/result,
    zero-duration markers for everything else). Type-dispatched on
    every flush().
  * :mod:`.metrics` — :func:`build_meter_provider` registers one
    :class:`ObservableGauge` per (node, stats field) named
    ``aii.{name}_{node_id}.{metric_suffix}``. Initial registration
    walks the live Run at build time; a daemon thread re-walks every
    2 s to catch tasks that materialize mid-run. Each gauge's callback
    closes over ONE node + ONE field — emits a single Observation per
    export tick. No per-flush metric work.

``flush()`` only touches the trace pipeline; the metric pipeline is
fully autonomous via the periodic reader's callback mechanism.
"""

from __future__ import annotations

import os
import platform
import socket
from typing import TYPE_CHECKING

from opentelemetry.sdk.resources import Resource

from aii_lib.run import emit
from aii_lib.run.sink import RunSink

from .log_correlation import enable_trace_correlation
from .metrics import build_meter_provider
from .traces import (
    HANDLER_BY_TYPE as _TRACE_HANDLERS,
)
from .traces import (
    TraceHandlers,
    build_tracer_provider,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aii_lib.run.messages import BaseMessage


class OTelRunSink(RunSink):
    """Run-bus subscriber: emits OTel traces (per-event) + metrics (periodic).

    Construction wires both pipelines with the supplied config
    (paths + cadences + OTLP destination + bearer auth header).
    """

    def __init__(
        self,
        *,
        run: object,
        traces_path: Path | str | None = None,
        metrics_path: Path | str | None = None,
        metrics_interval_ms: int = 30_000,
        trace_export_interval_ms: int = 0,
        service_name: str = "aii_pipeline",
        service_version: str | None = None,
        environment: str | None = None,
        sample_rate: float = 1.0,
        otlp_endpoint: str | None = None,
        otlp_insecure: bool = True,
        otlp_headers: dict[str, str] | None = None,
        resource_attrs: dict[str, str] | None = None,
    ) -> None:
        # ``run`` is captured by the metric ObservableGauge callbacks so
        # they can walk the live tree from the metric reader's
        # background thread (where ContextVar-based ``get_current_run``
        # doesn't propagate).
        # Bake run identity + environment into the Resource so EVERY span
        # and metric exported from this process carries it — independent
        # of any per-event handler attaching it. Survives partial/crashed
        # runs because Resource attrs ride along on every OTLP export,
        # even for spans whose parent never ended cleanly.
        attrs: dict[str, str] = {
            "service.name": service_name,
            "aii.run_id": run.node_id or "",
            # OTel semconv resource attrs — service.version /
            # deployment.environment let Grafana filter by deploy and
            # correlate regressions to releases. host/process attrs let
            # you slice by pod/container in multi-instance deployments.
            "host.name": socket.gethostname(),
            "host.arch": platform.machine(),
            "os.type": platform.system().lower(),
            "process.pid": str(os.getpid()),
            "process.runtime.name": platform.python_implementation().lower(),
            "process.runtime.version": platform.python_version(),
        }
        if service_version or os.environ.get("AII_SERVICE_VERSION"):
            attrs["service.version"] = service_version or os.environ["AII_SERVICE_VERSION"]
        if environment or os.environ.get("AII_ENVIRONMENT"):
            attrs["deployment.environment"] = environment or os.environ["AII_ENVIRONMENT"]
        if resource_attrs:
            attrs.update(resource_attrs)
        resource = Resource.create(attrs)

        # Inject the active span's trace_id/span_id into every loguru
        # log record so log lines correlate to traces in Grafana.
        enable_trace_correlation()

        # ── Trace pipeline ────────────────────────────────────────────
        self._tracer_provider = build_tracer_provider(
            resource=resource,
            sample_rate=sample_rate,
            traces_path=traces_path,
            otlp_endpoint=otlp_endpoint,
            otlp_insecure=otlp_insecure,
            otlp_headers=otlp_headers,
            trace_export_interval_ms=trace_export_interval_ms,
        )

        # ── Metric pipeline (autonomous; ObservableGauges fire on
        #    each periodic export tick by walking the live Run).
        # ``_discoverer_stop`` halts the discoverer daemon thread so
        # close() can shut down the meter provider without the daemon
        # racing it. ``task_duration_histogram`` is the sync histogram
        # the trace handlers record into on every task_end.
        (
            self._meter_provider,
            self._discoverer_stop,
            task_duration_histogram,
        ) = build_meter_provider(
            run=run,
            resource=resource,
            service_name=service_name,
            metrics_path=metrics_path,
            metrics_interval_ms=metrics_interval_ms,
            otlp_endpoint=otlp_endpoint,
            otlp_insecure=otlp_insecure,
            otlp_headers=otlp_headers,
        )

        self._trace_handlers = TraceHandlers(
            self._tracer_provider.get_tracer(service_name),
            task_duration_histogram=task_duration_histogram,
        )

        # Eagerly open the run span. ``aii_server`` emits ``run_start``
        # upstream of the pipeline subprocess (so the SSE/UI sees it
        # within ms instead of waiting for pipeline import), which means
        # the event never reaches sinks running in the subprocess. Without
        # this, every top-level ``mdgroup_start`` would fall back to an
        # empty ambient context and get its own fresh trace_id — so a
        # single run would be split into N independent traces (one per
        # mdgroup). Opening here pins all subsequent spans to one trace.
        self._trace_handlers.on_run_start({"run_id": run.node_id or "run"})
        emit.status_private_info(
            f"OTelRunSink started (service={service_name}, "
            f"sample_rate={sample_rate}, "
            f"otlp_endpoint={otlp_endpoint or 'none'}, "
            f"metrics_interval_ms={metrics_interval_ms})",
        )

    # ── RunSink contract ──────────────────────────────────────────────

    def flush(self, event: BaseMessage) -> None:
        """Dispatch a message event to the trace pipeline."""
        msg = event.model_dump(mode="json")
        msg_type = msg.get("type", "")

        # Trace side: dispatch by type, fall through to on_other for unknowns.
        # Metric side: nothing to do here — ObservableGauge callbacks fire
        # on the periodic reader's own schedule.
        trace_handler = _TRACE_HANDLERS.get(msg_type)
        if trace_handler is not None:
            trace_handler(self._trace_handlers, msg)
        else:
            self._trace_handlers.on_other(msg)

    def close(self) -> None:
        """Shut down trace and metric providers."""
        # End any spans still open at shutdown so they get exported.
        self._trace_handlers.close_open_spans()
        # Stop the metric discoverer daemon BEFORE the meter provider's
        # shutdown — otherwise the daemon's next scan can race and call
        # ``meter.create_observable_gauge`` on a dead provider.
        self._discoverer_stop.set()
        for label, provider in (
            ("tracer", self._tracer_provider),
            ("meter", self._meter_provider),
        ):
            try:
                provider.shutdown()
            except Exception as e:
                from loguru import logger

                logger.warning(f"OTelRunSink: {label} provider shutdown failed: {e}")


__all__ = ["OTelRunSink"]
