"""Loader for ``aii_config/dbos_run.yaml`` — runtime knobs for the DBOS pipeline.

Mirrors :mod:`aii_lib.dbos_app` (which loads ``dbos.yaml`` for the
connection string); this loads ``dbos_run.yaml`` for behavioural
settings (title timeout, interim-summary cadence, etc.). Both files
support a ``.private.yaml`` sibling overlay via
``aii_lib.utils.config_overrides.load_config_with_overrides``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aii_lib.utils.config_overrides import load_config_with_overrides
from pydantic import BaseModel, ConfigDict, Field

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DBOS_RUN_CONFIG_PATH = _REPO_ROOT / "aii_config" / "dbos_run.yaml"


class TitleStepConfig(BaseModel):
    """Run-title generator step config (gates the post-prompt LLM title call)."""

    enabled: bool = True
    timeout_s: float = 10.0
    model_config = ConfigDict(extra="allow")


class InterimSummaryStepConfig(BaseModel):
    """Periodic interim-summary loop config (cadence + LLM tuning)."""

    enabled: bool = True
    interval_s: int = 120
    initial_delay_s: float = 10.0
    min_new_messages: int = 2
    timeout_s: float = 20.0
    max_chars_per_msg: int = 5000
    reasoning_effort: str = "medium"
    model_config = ConfigDict(extra="allow")


class DbosRunConfig(BaseModel):
    """Top-level shape of ``aii_config/dbos_run.yaml::dbos_run``."""

    title: TitleStepConfig = Field(default_factory=TitleStepConfig)
    interim_summary: InterimSummaryStepConfig = Field(
        default_factory=InterimSummaryStepConfig,
    )
    model_config = ConfigDict(extra="allow")


def load_dbos_run_config() -> DbosRunConfig:
    """Read ``aii_config/dbos_run.yaml`` (+ private overlay) and validate."""
    raw: dict[str, Any] = load_config_with_overrides(_DBOS_RUN_CONFIG_PATH)
    return DbosRunConfig.model_validate(raw.get("dbos_run", {}))
