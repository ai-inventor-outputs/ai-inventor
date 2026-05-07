"""Schema for hypo_loop phase output."""

from __future__ import annotations

from typing import Literal

from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.out_schema import GenHypoOut
from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.out_schema import (
    ReviewHypoOut,
)
from pydantic import BaseModel, Field


class HypoLoopOut(BaseModel):
    """Aggregate output of the hypo_loop phase.

    Wraps the per-substep results that downstream phases (currently
    ``invention_loop``) consume. The ``hypothesis`` field surfaces the
    canonical "best" hypothesis at the end of the loop — sourced from
    the latest ``review_hypo`` if available, falling back to the first
    ``gen_hypo`` hypothesis. Carrying it on the aggregate lets readers
    avoid re-walking the per-iteration tree.
    """

    kind: Literal["hypo_loop_out"] = "hypo_loop_out"
    gen_hypo: GenHypoOut | None = Field(
        default=None,
        description="First-iteration gen_hypo output (used as a fallback hypothesis source).",
    )
    review_hypo: ReviewHypoOut | None = Field(
        default=None,
        description="Latest review_hypo output across iterations (carries the final hypothesis).",
    )
    iterations_completed: int = Field(
        default=0,
        description="Number of iterations the loop ran to completion.",
    )
