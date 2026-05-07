"""TracerProvider construction for the OTel sink.

Wires the JSONL exporter (always on) and an optional OTLP exporter
(Grafana Cloud) onto a shared :class:`TracerProvider` with the
configured sampler. Caller picks ``Simple`` vs ``Batch`` span processing
via ``trace_export_interval_ms`` (0 → Simple, >0 → Batch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from .jsonl_exporter import JSONLSpanExporter

if TYPE_CHECKING:
    from pathlib import Path

    from opentelemetry.sdk.resources import Resource


def _make_processor(
    exporter: object, trace_export_interval_ms: int
) -> SimpleSpanProcessor | BatchSpanProcessor:
    """0 → SimpleSpanProcessor; >0 → BatchSpanProcessor with that delay."""
    if trace_export_interval_ms > 0:
        return BatchSpanProcessor(
            exporter,
            schedule_delay_millis=trace_export_interval_ms,
        )
    return SimpleSpanProcessor(exporter)


def build_tracer_provider(
    *,
    resource: Resource,
    sample_rate: float,
    traces_path: Path | str | None,
    otlp_endpoint: str | None,
    otlp_insecure: bool,
    otlp_headers: dict[str, str] | None,
    trace_export_interval_ms: int,
) -> TracerProvider:
    """Build a :class:`TracerProvider` with local + (optional) OTLP exporters."""
    sampler = TraceIdRatioBased(sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    if traces_path:
        provider.add_span_processor(
            _make_processor(JSONLSpanExporter(traces_path), trace_export_interval_ms)
        )

    if otlp_endpoint:
        try:
            if otlp_endpoint.startswith("https://") and not otlp_insecure:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp_exporter = OTLPSpanExporter(
                    endpoint=f"{otlp_endpoint}/v1/traces",
                    headers=otlp_headers or {},
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    insecure=otlp_insecure,
                    headers=otlp_headers or {},
                )
            provider.add_span_processor(_make_processor(otlp_exporter, trace_export_interval_ms))
        except ImportError:
            from loguru import logger

            logger.warning("OTLP exporter not installed — skipping remote trace export")

    return provider


__all__ = ["build_tracer_provider"]
