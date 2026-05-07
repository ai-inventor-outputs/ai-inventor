"""``JSONLSpanExporter`` — write OTel spans as JSONL for offline analysis.

One JSONL line per :class:`ReadableSpan`. Used by the local file sink so
the run dir keeps a self-contained copy of every span without standing
up a collector.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan


class JSONLSpanExporter(SpanExporter):
    """Write spans as JSONL for offline analysis."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Write spans to JSONL."""
        if self._file is None:
            return SpanExportResult.SUCCESS
        with self._lock:
            for span in spans:
                attrs = dict(span.attributes) if span.attributes else {}
                events = [
                    {
                        "name": e.name,
                        "timestamp": e.timestamp,
                        "attributes": dict(e.attributes) if e.attributes else {},
                    }
                    for e in (span.events or [])
                ]
                record = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_span_id": (
                        format(span.parent.span_id, "016x") if span.parent else None
                    ),
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "status": span.status.status_code.name,
                    "attributes": attrs,
                    "events": events,
                }
                self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Close the JSONL file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


__all__ = ["JSONLSpanExporter"]
