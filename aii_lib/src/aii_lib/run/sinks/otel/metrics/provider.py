"""MeterProvider construction for the OTel sink.

Per-node instruments: every Run / Group / Iteration / Module in the live
tree gets its own set of OTel observable instruments named
``aii.{display_name}_{node_id}.{field}``. Each instrument has a closure
callback that reads exactly one stat field on one specific node, so the
metric *name* alone identifies the node — no attribute-pivot needed in
Grafana.

Registration is a hybrid, gated on ``stats.total_messages > 0``:

  * **Initial pass**: at provider build time, walk the tree and register
    instruments for every active node.
  * **Discoverer thread**: a daemon thread re-walks every 2 s and
    registers instruments for nodes that have since received their first
    event (covers pre-scaffolded nodes that have just gone active).

Tasks are visited by the walker but skipped here — task-level series
multiply cardinality without adding signal (every task value rolls up
into its parent module). Per-task latency is still observable via the
``aii.task.duration`` histogram (recorded by the trace pipeline on every
``task_end``), and per-task cost shows up via trace span attributes.
Pre-scaffolded but inert nodes from ``pipeline.yaml`` that haven't been
entered yet are also NOT registered — we don't want N empty instruments
in the SDK.
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from .handlers import _walk_nodes
from .jsonl_exporter import JSONLMetricExporter

if TYPE_CHECKING:
    from pathlib import Path

    from opentelemetry.sdk.resources import Resource

    from aii_lib.run.run import Run


# (field_on_NodeStats, description, unit) — the field name IS the metric
# suffix verbatim (no rename), so Grafana series mirror the in-memory
# NodeStats schema 1:1. All five are monotonic cumulative values, hence
# ObservableCounter — PromQL ``rate()`` / ``increase()`` work and Mimir
# handles process restarts via counter-reset detection.
_FIELDS = [
    ("runtime_seconds", "wall-clock duration so far", "s"),
    ("total_messages", "total bus events so far", ""),
    ("total_cost", "cumulative cost so far", "usd"),
    ("cum_all_input_tokens", "cumulative all-input tokens", ""),
    ("cum_all_output_tokens", "cumulative output tokens", ""),
]

_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize(s: str) -> str:
    """Force ``s`` into a Prometheus-safe metric name fragment."""
    cleaned = _NAME_SAFE.sub("_", s) if s else ""
    return cleaned or "unnamed"


def _make_field_callback(field_name: str, registry: list, registry_lock: threading.Lock) -> object:
    """Closure that reads ONE field across ALL registered nodes per export tick.

    Returns one ``Observation`` per registered node, with node identity
    (``node_id`` / ``node_name``) and ancestor chain in the attribute
    dict. Replaces the prior per-node-per-field callback design, which
    multiplied unique metric NAMES by ~node_count (≈180 metric names
    per run before this refactor — each fresh per-run-uid node_id
    minted brand-new series in the OTel backend, unbounded over time).

    Now: 5 metric names total (one per field in :data:`_FIELDS`), every
    node's value rides on labels. Mimir / Prometheus handles label
    cardinality natively and supports per-series GC.

    The ``total_messages == 0`` skip stays as a per-node defensive
    filter so tests that reset stats mid-run don't emit ghost zeros.
    """

    def _cb(options: object) -> list:
        observations: list = []
        # Snapshot the registry under lock so a concurrent
        # ``_register_for_node`` mutation can't tear iteration.
        with registry_lock:
            entries = list(registry)
        for node, attrs in entries:
            stats = getattr(node, "stats", None)
            if stats is None or getattr(stats, "total_messages", 0) == 0:
                continue
            v = getattr(stats, field_name, 0) or 0
            observations.append(Observation(value=v, attributes=attrs))
        return observations

    return _cb


def build_meter_provider(
    *,
    run: Run,
    resource: Resource,
    service_name: str,
    metrics_path: Path | str | None,
    metrics_interval_ms: int,
    otlp_endpoint: str | None,
    otlp_insecure: bool,
    otlp_headers: dict[str, str] | None,
) -> tuple:
    """Returns ``(provider, stop_event, task_duration_histogram)``.

    Set ``stop_event`` to halt the discoverer daemon before shutting down
    ``provider``. ``task_duration_histogram`` is the sync histogram the
    trace pipeline records into on every ``task_end``.
    """
    readers = []
    if metrics_path:
        readers.append(
            PeriodicExportingMetricReader(
                JSONLMetricExporter(metrics_path),
                export_interval_millis=metrics_interval_ms,
            )
        )
    if otlp_endpoint:
        try:
            if otlp_endpoint.startswith("https://") and not otlp_insecure:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                otlp_exporter = OTLPMetricExporter(
                    endpoint=f"{otlp_endpoint}/v1/metrics",
                    headers=otlp_headers or {},
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )

                otlp_exporter = OTLPMetricExporter(
                    endpoint=otlp_endpoint,
                    insecure=otlp_insecure,
                    headers=otlp_headers or {},
                )
            readers.append(
                PeriodicExportingMetricReader(
                    otlp_exporter,
                    export_interval_millis=metrics_interval_ms,
                )
            )
        except ImportError:
            from loguru import logger

            logger.warning("OTLP exporter not installed — skipping remote metric export")

    provider = MeterProvider(resource=resource, metric_readers=readers)
    meter = provider.get_meter(service_name)

    # ── Latency histogram ─────────────────────────────────────────────
    # Sync instrument recorded by the trace pipeline on every task_end —
    # gives p95/p99 latency distributions per (module, task_name) without
    # per-instance series explosion. Buckets are seconds-scale because
    # tasks routinely run minutes.
    task_duration_histogram = meter.create_histogram(
        name="aii.task.duration",
        description="Task wall-clock duration",
        unit="s",
    )

    # ── Per-field counter registration (single instrument per field) ──
    # Five instruments total (one per :data:`_FIELDS` entry). Each
    # instrument's callback iterates a per-field registry of nodes and
    # emits one Observation per registered node, with the node's identity
    # in attributes. ``registered`` is keyed by node_id so re-walks don't
    # double-add. Locks: ``registered_lock`` for the membership set,
    # ``field_registries_lock`` for the per-field entry lists (callback
    # snapshots under it). Both are short hot-paths.
    field_registries: dict[str, list[tuple]] = {f[0]: [] for f in _FIELDS}
    field_registries_lock = threading.Lock()
    for field_name, desc, unit in _FIELDS:
        meter.create_observable_counter(
            name=f"aii.{field_name}",
            description=desc,
            unit=unit,
            callbacks=[
                _make_field_callback(
                    field_name, field_registries[field_name], field_registries_lock
                )
            ],
        )

    registered: set[str] = set()
    registered_lock = threading.Lock()

    def _register_for_node(node: object, display_name: str, ancestor_attrs: dict) -> None:
        # Defer registration until the node has actually fired an event.
        # Pre-scaffolded modules (created from pipeline.yaml at build
        # time) shouldn't appear as empty observations in the SDK — the
        # next 2 s discoverer scan picks them up the moment they receive
        # their first bus event.
        stats = getattr(node, "stats", None)
        if stats is None or getattr(stats, "total_messages", 0) == 0:
            return
        node_id = getattr(node, "node_id", "") or ""
        if not node_id:
            return
        with registered_lock:
            if node_id in registered:
                return
            registered.add(node_id)
        # Snapshot ancestor_attrs at registration — later mutations to the
        # source dict can't tamper with what gets emitted. Add the leaf
        # node's own identity as labels so the FE / Grafana can filter on
        # ``node_id`` instead of discovering a separate metric name per node.
        attrs = {
            **(ancestor_attrs or {}),
            "node_id": node_id,
            "node_name": _sanitize(display_name) if display_name else "",
        }
        with field_registries_lock:
            for field_name, _desc, _unit in _FIELDS:
                field_registries[field_name].append((node, attrs))

    def _scan_tree() -> None:
        try:
            for level, node, display_name, ancestor_attrs in _walk_nodes(run):
                # Task-level series multiply cardinality without adding
                # signal (every task value rolls up into its parent
                # module, and per-task latency comes via the
                # ``aii.task.duration`` histogram instead).
                if level == "task":
                    continue
                _register_for_node(node, display_name, ancestor_attrs)
        except Exception as e:
            from loguru import logger

            logger.debug(f"otel discoverer scan failed: {e}")

    # Initial pass — captures everything pre-scaffolded by pipeline.yaml.
    _scan_tree()

    # Daemon thread for groups/modules that go active mid-run (their
    # first bus event arrives after build time). 2 s cadence is well
    # below the 30 s metric export tick so newly-active nodes get an
    # instrument before their first emission. ``stop_event`` is set by
    # ``OTelRunSink.close()`` so the loop exits within 2 s instead of
    # racing the meter provider's shutdown (which would make the next
    # ``meter.create_observable_counter`` call raise).
    stop_event = threading.Event()

    def _discovery_loop() -> None:
        while not stop_event.wait(2.0):
            _scan_tree()

    threading.Thread(
        target=_discovery_loop,
        daemon=True,
        name="otel-metric-discoverer",
    ).start()

    return provider, stop_event, task_duration_histogram


__all__ = ["build_meter_provider"]
