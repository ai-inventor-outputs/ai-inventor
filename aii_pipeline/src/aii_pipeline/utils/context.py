"""Pipeline context — ambient PipelineConfig accessible from anywhere.

Set once at pipeline start, read from prompts/steps/executors without threading.
"""

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_config import PipelineConfig

_config_var: contextvars.ContextVar["PipelineConfig | None"] = contextvars.ContextVar(
    "aii_pipeline_config", default=None
)


def set_pipeline_config(config: "PipelineConfig") -> None:
    """Set the ambient PipelineConfig context variable."""
    _config_var.set(config)


def get_pipeline_config() -> "PipelineConfig | None":
    """Retrieve the ambient PipelineConfig context variable."""
    return _config_var.get()
