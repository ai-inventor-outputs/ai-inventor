"""Schema for invention loop step output."""

from __future__ import annotations

from typing import Literal

from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,  # noqa: TC002 — pydantic v2 needs the runtime import to resolve ``list[BaseArtifact]`` annotation at validation time; ``from __future__ import annotations`` stringifies the annotation but pydantic still loads the symbol from the module's globals when ``model_validate`` runs. Moving this under ``TYPE_CHECKING`` would break that.
)
from aii_pipeline.steps.base import BaseStepOut
from pydantic import Field


class InventionLoopOut(BaseStepOut):
    """Output of the invention_loop module."""

    kind: Literal["invention_loop_out"] = "invention_loop_out"
    pools_dir: str = Field(default="", description="Path to pools directory")
    artifacts: list[BaseArtifact] = Field(
        default_factory=list, description="All artifacts produced"
    )
    hypothesis: dict = Field(
        default_factory=dict, description="Input hypothesis (possibly revised)"
    )
