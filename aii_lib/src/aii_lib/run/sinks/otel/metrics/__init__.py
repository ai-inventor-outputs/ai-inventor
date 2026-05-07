"""Metric pipeline for the OTel sink — provider + JSONL exporter.

Per-node :class:`ObservableGauge` instruments named
``aii.{name}_{node_id}.{metric_suffix}`` (one set per Group / Module /
Task). Initial registration walks the live Run at provider-build time;
a daemon thread (``otel-metric-discoverer``) re-walks every 2 s to
catch nodes that materialize mid-run. No per-event handler — the
metric pipeline is fully periodic-snapshot driven.
"""

from __future__ import annotations

from .jsonl_exporter import JSONLMetricExporter
from .provider import build_meter_provider

__all__ = [
    "JSONLMetricExporter",
    "build_meter_provider",
]
