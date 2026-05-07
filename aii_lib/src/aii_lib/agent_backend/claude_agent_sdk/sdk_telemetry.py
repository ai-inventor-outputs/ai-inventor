"""Claude Agent SDK telemetry — wire the CLI's built-in OTel emission.

The Claude SDK runs the ``claude`` CLI as a subprocess and the CLI ships
OTel spans/metrics/logs over OTLP when ``CLAUDE_CODE_ENABLE_TELEMETRY=1``
plus the standard ``OTEL_EXPORTER_OTLP_*`` vars are present in its env
(verified in ``claude_agent_sdk._internal.transport.subprocess_cli`` —
the SDK merges ``os.environ`` into ``process_env`` before spawning).

This module is the *transport* config — it runs once at process boot and
sets the OTLP endpoint, headers, protocol, exporter selection, etc. on
``os.environ`` so every SDK subprocess inherits them. The per-call
on/off switch (``CLAUDE_CODE_ENABLE_TELEMETRY``) lives on the
:class:`AgentOptions.telemetry` flag and is set by the SDK options
builder, not here.

Config source:
``aii_config/pipeline/harness/agent_backend.yaml::claude_agent_sdk.sdk_telemetry``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

# Search order matches the rest of the codebase (autologin/accounts,
# credentials, monitor): repo-local first, Docker layout second.
_CONFIG_CANDIDATES: tuple[Path, ...] = (
    Path(__file__).resolve().parents[5]
    / "aii_config"
    / "pipeline"
    / "harness"
    / "agent_backend.yaml",
    Path("/ai-inventor/aii_config/pipeline/harness/agent_backend.yaml"),
)


def _find_config() -> Path | None:
    for candidate in _CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _load_section(config_path: Path) -> dict[str, Any] | None:
    try:
        from aii_lib.utils.config_overrides import load_config_with_overrides

        data = load_config_with_overrides(config_path)
    except Exception as e:
        logger.warning(f"sdk_telemetry: cannot read {config_path}: {e}")
        return None
    section = data.get("claude_agent_sdk", {}).get("sdk_telemetry")
    return section if isinstance(section, dict) else None


def configure_sdk_telemetry(config_path: Path | None = None) -> bool:
    """Apply OTLP transport env vars from ``sdk_telemetry`` config.

    Reads ``aii_config/pipeline/harness/agent_backend.yaml`` (or ``config_path``
    if given), pulls the ``claude_agent_sdk.sdk_telemetry`` section,
    resolves the auth header from ``auth_env``, and sets:

      - ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``_PROTOCOL`` / ``_HEADERS``
      - ``OTEL_METRICS_EXPORTER`` / ``OTEL_LOGS_EXPORTER`` (always ``otlp``)
      - ``OTEL_TRACES_EXPORTER`` + ``CLAUDE_CODE_ENHANCED_TELEMETRY_BETA``
        only when ``enable_traces_beta`` is on (traces are SDK-beta).
      - ``OTEL_SERVICE_NAME`` — defaults to ``aii-claude-sdk``.
      - ``OTEL_METRIC_EXPORT_INTERVAL`` / ``OTEL_LOGS_EXPORT_INTERVAL``.

    Does NOT set ``CLAUDE_CODE_ENABLE_TELEMETRY`` — that's the per-call
    switch and lives on ``AgentOptions.telemetry``.

    Returns True if env was applied, False otherwise (config missing,
    section absent, or ``enabled: false``). Idempotent — re-running with
    the same config is a no-op since values match. Existing env entries
    are NOT overwritten so callers can pin their own (e.g. tests, dev
    overrides via shell ``OTEL_EXPORTER_OTLP_ENDPOINT=...``).
    """
    path = config_path or _find_config()
    if path is None:
        logger.debug("sdk_telemetry: harness/agent_backend.yaml not found, skipping")
        return False

    section = _load_section(path)
    if section is None or not section.get("enabled", False):
        logger.debug(f"sdk_telemetry: disabled or missing in {path}")
        return False

    # ---- Resolve auth header from the env-var indirection -------------------
    auth_env = section.get("auth_env") or "GRAFANA_OTLP_AUTH"
    auth_value = os.environ.get(auth_env, "").strip()
    headers: dict[str, str] = {}
    if auth_value:
        # ``OTEL_EXPORTER_OTLP_HEADERS`` is a comma-separated string of
        # ``key=value`` pairs. Per the OTel spec the value section can
        # contain raw spaces (the spec's own example is
        # ``Authorization=Bearer 12345``). The Claude CLI's Node OTel SDK
        # does NOT url-decode values, so url-encoding here would result in
        # the literal ``Basic%20...`` reaching Grafana, which it rejects.
        # Pass the env var through verbatim — same shape as the AII
        # pipeline OTel sink.
        headers["Authorization"] = auth_value
    else:
        logger.warning(
            f"sdk_telemetry: ${auth_env} is not set — telemetry transport "
            "configured but exports will fail at the gateway",
        )

    # ---- Build the env mapping ---------------------------------------------
    endpoint = section.get("otlp_endpoint")
    protocol = section.get("otlp_protocol", "http/protobuf")
    service_name = section.get("service_name", "aii-claude-sdk")
    metric_interval = section.get("metric_export_interval_ms", 60000)
    logs_interval = section.get("logs_export_interval_ms", 5000)
    enable_traces = bool(section.get("enable_traces_beta", False))
    log_user_prompts = bool(section.get("log_user_prompts", False))

    env_updates: dict[str, str] = {
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_SERVICE_NAME": service_name,
        "OTEL_METRIC_EXPORT_INTERVAL": str(metric_interval),
        "OTEL_LOGS_EXPORT_INTERVAL": str(logs_interval),
        # Grafana Cloud Mimir's OTLP ingester rejects delta temporality with
        # ``400 Bad Request``. The Claude CLI defaults to ``delta`` (we
        # observed it auto-set ``OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE
        # =delta`` on subprocess spawn). Force cumulative so counters land.
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "cumulative",
    }
    if endpoint:
        env_updates["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        # The Claude CLI's Node OTel SDK does NOT auto-append ``/v1/<signal>``
        # to the base endpoint for HTTP/protobuf — Grafana Cloud's OTLP
        # gateway returns ``400 Bad Request`` when the unsigned base URL is
        # POSTed to. Set signal-specific endpoints with full paths so each
        # signal lands at the right ingress (matches what
        # ``aii_lib.run.sinks.otel.metrics.provider`` does for the AII
        # pipeline OTel sink — it explicitly builds ``f"{base}/v1/metrics"``).
        base = endpoint.rstrip("/")
        env_updates["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] = f"{base}/v1/metrics"
        env_updates["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = f"{base}/v1/logs"
        if enable_traces:
            env_updates["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = f"{base}/v1/traces"
    if protocol:
        env_updates["OTEL_EXPORTER_OTLP_PROTOCOL"] = protocol
    if headers:
        env_updates["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(f"{k}={v}" for k, v in headers.items())
    if enable_traces:
        env_updates["OTEL_TRACES_EXPORTER"] = "otlp"
        env_updates["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] = "1"
    if log_user_prompts:
        # Adds prompt text to ``claude_code.user_prompt`` events and the
        # ``claude_code.interaction`` span. SDK default is redacted; we
        # flip it on so Loki/Tempo carry the full conversation content.
        env_updates["OTEL_LOG_USER_PROMPTS"] = "1"

    # ---- Apply, preserving caller-pinned overrides --------------------------
    applied: list[str] = []
    for k, v in env_updates.items():
        if k not in os.environ:
            os.environ[k] = v
            applied.append(k)

    logger.info(
        f"sdk_telemetry: configured {len(applied)} env vars from {path.name} "
        f"(endpoint={endpoint or 'unset'}, traces_beta={enable_traces})",
    )
    return True


__all__ = ["configure_sdk_telemetry"]
