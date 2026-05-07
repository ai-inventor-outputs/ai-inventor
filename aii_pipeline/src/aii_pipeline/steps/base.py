"""Pipeline step abstractions — StepContext, ModuleCtx, BaseStepOut.

``StepContext`` is the top-level ctx pushed onto :data:`aii_lib.run.context._ctx`
by :func:`run_pipeline` before iterating phases. Each phase's
``get_context()`` reads it via :func:`current_ctx` and builds its own
narrower phase ctx (``HypoLoopCtx`` / ``LoopCtx`` / etc.).

``ModuleCtx`` is the shared base for per-substep ctx dataclasses
(``GenVizCtx``, ``GenArtDemoCtx``, ``GenFullPaperCtx``, ``GenPaperCtx``,
``DeployGhCtx`` …) — adds ``config`` / ``output_dir`` slots that every
substep helper expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from aii_pipeline.utils import PipelineConfig


@dataclass
class ModuleCtx:
    """Common context for any pipeline module's internal state.

    Shared base for the various per-module Ctx dataclasses (LoopCtx,
    GenPaperCtx, ...) that thread state through their substep helpers.
    Each module subclasses this and adds its own fields — pools, ledgers,
    accumulated results, callbacks. Use this base when a helper only needs
    config/output_dir so signatures don't have to know which concrete Ctx
    is in play.
    """

    config: PipelineConfig
    output_dir: Path


@dataclass
class StepContext:
    """Top-level context pushed onto the run context by run_pipeline.

    Each phase's ``get_context()`` reads this via :func:`current_ctx`
    and builds its own narrower typed phase ctx
    (``HypoLoopCtx`` / ``LoopCtx`` / ``GenPaperRepoPhaseCtx`` / ...).
    Phase results are NOT stored here — they live on the run tree as
    :attr:`MdGroup.output` (set by ``mdgroup_output`` events) and each
    phase's ``get_context`` reads upstream results via
    :meth:`Run.find_group_by_name`.

    Pre-Stage-7 this dataclass also carried ``start_iter`` /
    ``start_substep`` for skip-ahead resume; v27 replay-execute
    (REPLAY_EXECUTE_PLAN.md) made those fields unnecessary by
    handling resume uniformly via ``Run._playback_mode`` + idempotent
    dispatch + ``Agent.run`` synthesis. The fields were deleted in
    Stage 10 cleanup.
    """

    config: PipelineConfig
    run_dir: Path


class BaseStepOut(BaseModel):
    """Base class for all pipeline step outputs.

    Every step must produce an output_dir and metadata dict.
    """

    output_dir: str = Field(default="", description="Output directory path")
    metadata: dict = Field(default_factory=dict, description="Module metadata")
