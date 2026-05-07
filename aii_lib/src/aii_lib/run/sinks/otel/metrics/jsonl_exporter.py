"""``JSONLMetricExporter`` — write OTel metric snapshots as JSONL.

One JSONL line per data point per metric per scope per resource.
Histograms include their bucket counts + bounds so percentiles can be
recovered offline.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics.export import MetricsData


class JSONLMetricExporter(MetricExporter):
    """Write metrics as JSONL for offline analysis."""

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    # Signatures below mirror ``MetricExporter`` upstream — ``timeout_millis``
    # is a ``float`` with the SDK's default, not ``int | None``.
    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs
    ) -> MetricExportResult:
        """Write metrics to JSONL."""
        if self._file is None:
            return MetricExportResult.SUCCESS
        with self._lock:
            for resource_metrics in metrics_data.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        for dp in metric.data.data_points:
                            record = {
                                "name": metric.name,
                                "description": metric.description,
                                "unit": metric.unit,
                                "value": (
                                    v
                                    if (v := getattr(dp, "value", None)) is not None
                                    else getattr(dp, "sum", None)
                                ),
                                "attributes": dict(dp.attributes) if dp.attributes else {},
                                "timestamp": dp.time_unix_nano,
                            }
                            if hasattr(dp, "bucket_counts"):
                                record["bucket_counts"] = list(dp.bucket_counts)
                                record["explicit_bounds"] = list(dp.explicit_bounds)
                                record["count"] = dp.count
                                record["sum"] = dp.sum
                            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Force-flush pending metrics."""
        with self._lock:
            if self._file:
                self._file.flush()
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
        """Close the JSONL file."""
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


__all__ = ["JSONLMetricExporter"]
