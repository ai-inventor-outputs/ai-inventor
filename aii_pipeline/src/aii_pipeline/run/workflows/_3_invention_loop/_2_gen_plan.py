"""``gen_plan`` as a DBOS child workflow.

Same delegation pattern as :func:`gen_strat_workflow` — minimal LoopCtx
+ stand-in ``invention_loop_group`` + call into the legacy
:class:`GenPlanModule.execute`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dbos import DBOS
from pydantic import BaseModel, Field, TypeAdapter

from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
    Strategy,
)
from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
    BaseArtifact,
)
from aii_pipeline.run.workflows._3_invention_loop._loop_group_stub import _LoopGroupStub
from aii_pipeline.steps._3_invention_loop._2_gen_plan import GenPlanModule
from aii_pipeline.steps._3_invention_loop.invention_loop import LoopCtx
from aii_pipeline.utils import PipelineConfig


class GenPlanWorkflowInput(BaseModel):
    """JSON-safe input to ``gen_plan_workflow``."""

    iteration: int
    """1-indexed iteration number."""

    parent_id: str
    """Node id of the owning iteration (deterministic via path-derived id)."""

    hypothesis: dict[str, Any]
    """The current hypothesis as a dict."""

    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    """Existing artifacts as serialised ``BaseArtifact`` dicts."""

    strategies: list[dict[str, Any]] = Field(default_factory=list)
    """Strategies produced by ``gen_strat`` for this iteration, as
    serialised ``Strategy`` dicts. Workflow body re-validates."""

    user_uploads_path: str
    """Absolute path to the run's ``user_uploads/`` directory."""

    output_dir: str
    """Absolute path to the iteration's output directory."""

    config_dirs: list[str] = Field(default_factory=list)
    """Optional layered ``aii_config/pipeline/`` dirs (mirrors cli
    ``--config-dir`` flags). Empty for default canonical-only config."""


@DBOS.workflow()
async def gen_plan_workflow(wf_input: GenPlanWorkflowInput) -> list[dict[str, Any]]:
    """Generate detailed plans from a strategy set for one iteration.

    Args:
        input: :class:`GenPlanWorkflowInput` — iteration / parent / hypothesis
            / artifact pool / strategies (output of gen_strat) / paths.

    Returns:
        Serialised :class:`BasePlan` dicts. Callers re-validate via the
        plan discriminator union.

    Determinism: same as :func:`gen_strat_workflow` — workflow body is
    deterministic (config / ctx reconstruction), expensive LLM work
    happens inside ``Agent.run`` steps wrapped in Phase 1.1.
    """
    config = PipelineConfig.from_yaml(
        *[Path(d) for d in wf_input.config_dirs],
    )

    # Reconstruct the typed lists from JSON dicts. ``Strategy`` is a
    # plain Pydantic model with no discriminated union; ``BaseArtifact``
    # uses ``kind``-based dispatch widened by
    # :func:`bind_pipeline_typed_unions` at boot.
    strategies = [Strategy.model_validate(d) for d in wf_input.strategies]
    artifact_adapter = TypeAdapter(BaseArtifact)
    artifacts = [artifact_adapter.validate_python(d) for d in wf_input.artifacts]

    ctx = LoopCtx(
        config=config,
        output_dir=Path(wf_input.output_dir),
        hypothesis=wf_input.hypothesis,
        invention_loop_group=_LoopGroupStub(artifacts=artifacts),  # type: ignore[arg-type]
        user_uploads_path=wf_input.user_uploads_path,
        max_iterations=config.invention_loop.max_iterations,
        run_dir=Path(wf_input.output_dir).parent.parent,
    )

    module = GenPlanModule()
    plans = await module.execute(
        ctx=ctx,
        iteration=wf_input.iteration,
        output_dir=Path(wf_input.output_dir),
        strategies=strategies,
        parent_id=wf_input.parent_id,
    )

    return [p.model_dump() for p in (plans or [])]
