"""Inject OTel trace IDs into loguru log records for Grafana correlation.

Inject the active OTel ``trace_id`` / ``span_id`` into every loguru log
record so log lines correlate to traces in Grafana.

Called once from :class:`OTelRunSink.__init__`. Uses
``loguru.logger.configure(patcher=...)`` so the patch is global and idempotent
— calling repeatedly just re-installs the same patcher. The active span is
read via ``opentelemetry.trace.get_current_span()`` at log time, which works
across threads, asyncio tasks, and ContextVar copies.

Format strings can reference the injected fields with ``{extra[trace_id]}``
and ``{extra[span_id]}``; logs emitted outside any span get empty strings
rather than ``KeyError`` to keep format strings simple.
"""

from __future__ import annotations


def enable_trace_correlation() -> None:
    """Configure loguru to emit ``extra.trace_id`` / ``extra.span_id``.

    Idempotent — safe to call multiple times.
    """
    from loguru import logger
    from opentelemetry import trace as otel_trace

    def _patcher(record: dict) -> None:
        span = otel_trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            record["extra"]["trace_id"] = format(ctx.trace_id, "032x")
            record["extra"]["span_id"] = format(ctx.span_id, "016x")
        else:
            record["extra"]["trace_id"] = ""
            record["extra"]["span_id"] = ""

    logger.configure(patcher=_patcher)


__all__ = ["enable_trace_correlation"]
