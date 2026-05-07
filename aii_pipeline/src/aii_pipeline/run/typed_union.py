"""Pipeline-side rebinding of ``children`` and ``output`` unions.

aii_lib's ``Run.children`` / ``LoopIteration.children`` /
``SeqMdGroup.children`` annotations default to base discriminated
unions (``SeqMdGroup | LoopMdGroup`` / ``SingleTModule | ParallelTModule``).
``model_validate`` of a ``fork_init`` seed dict whose nested groups
carry e.g. ``kind="hypo_loop_group"`` would raise a discriminator
error — the value isn't in the base union. We need the union to widen
to include every phase + substep subclass before any seed deserialization.

The same applies to :attr:`AIINode.output`. The base annotation is
``Any`` (so aii_lib doesn't import aii_pipeline). Replay of
``run_output`` / ``mdgroup_output`` / ``module_output`` / ``task_output``
events deserializes JSON payloads back into typed output models
(``Hypothesis``, ``BaseArtifact``, ``GenPaperRepoOut`` …) via a
``kind`` discriminator. We widen ``output`` on every ``AIINode``
subclass to a discriminated union covering every known output class.

Approach:

  1. Import every phase MdGroup + substep Module typed subclass and
     every output class (``Hypothesis``, ``BaseArtifact``, etc.).
  2. Build wider discriminated unions for ``children`` and ``output``.
  3. **Mutate** the ``annotation`` on each FieldInfo —
     ``Run.model_fields["children"]``, every AIINode subclass's
     ``model_fields["output"]``, etc. ``model_rebuild(force=True)``
     alone doesn't re-evaluate already-resolved annotations; the
     forward reference was bound to the base annotation at class
     build time. Direct field mutation is what actually swaps the
     active type.
  4. Call ``model_rebuild(force=True)`` on every affected class to
     regenerate validators against the new annotations.

Idempotent: a guard flag short-circuits subsequent calls.

Call :func:`bind_pipeline_typed_unions` once at process boot, BEFORE
any ``from_fork`` / ``from_resume`` runs and before any seed
deserialization. cli.py wires it alongside config-load.
"""

from __future__ import annotations

from typing import Annotated

from aii_lib.run.loop_iteration import LoopIteration
from aii_lib.run.mdgroup import LoopMdGroup, MdGroup, SeqMdGroup
from aii_lib.run.messages import BaseMessage
from aii_lib.run.module import Module, ParallelTModule, SingleTModule
from aii_lib.run.node import AIINode
from aii_lib.run.run import Run
from aii_lib.run.task import ClaudeAgentTask, Task
from pydantic import Field

_BOUND = False


def bind_pipeline_typed_unions() -> None:
    """Widen children and output annotations for pipeline subclasses.

    Widen ``children`` annotations on Run / LoopIteration / SeqMdGroup
    to include pipeline phase + substep subclasses, AND widen the
    ``output`` annotation on every ``AIINode`` subclass to a
    discriminated union over every known output class. Then rebuild
    validators. Idempotent.
    """
    global _BOUND
    if _BOUND:
        return

    # Widen ``AIINode.messages`` from base ``list[BaseMessage]`` to the
    # discriminated union over every known message subclass — wired
    # FIRST so the union schemas (pipeline_mdgroup / pipeline_module)
    # later embed sub-class schemas that already carry typed messages.
    from aii_lib.run.messages import bind_message_union

    bind_message_union()

    # Phase MdGroup subclasses.
    # Output classes — members of the ``pipeline_output`` discriminated
    # union. Every class carries a ``kind: Literal[...]`` tag; pydantic
    # uses it to dispatch JSON → correct subclass on replay.
    from aii_pipeline.prompts.steps._1_seed_hypo.out_schema import SeedHypoOut
    from aii_pipeline.prompts.steps._2_hypo_loop._1_gen_hypo.out_schema import (
        GenHypoOut,
        Hypothesis,
    )
    from aii_pipeline.prompts.steps._2_hypo_loop._2_review_hypo.out_schema import (
        ReviewHypoOut,
    )
    from aii_pipeline.prompts.steps._2_hypo_loop.out_schema import (
        HypoLoopOut,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
        GenStratOut,
        Strategies,
        Strategy,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._2_gen_plan.out_schema import (
        BasePlan,
        DatasetPlan,
        EvaluationPlan,
        ExperimentPlan,
        GenPlanOut,
        ProofPlan,
        ResearchPlan,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.dataset.out_schema import (
        DatasetArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.evaluation.out_schema import (
        EvaluationArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.experiment.out_schema import (
        ExperimentArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.proof.out_schema import (
        ProofArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.research.out_schema import (
        ResearchArtifact,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
        PaperText,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._5_review_paper.out_schema import (
        ReviewerFeedback,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._6_upd_hypo.out_schema import (
        RevisedHypothesis,
    )
    from aii_pipeline.prompts.steps._3_invention_loop.out_schema import (
        InventionLoopOut,
    )
    from aii_pipeline.prompts.steps._4_gen_paper_repo._2_gen_viz.out_schema import (
        Figure,
        VizFigureOutput,
    )
    from aii_pipeline.prompts.steps._4_gen_paper_repo._3_gen_art_demo.schema_code import (
        BaseDemo,
        CodeDemo,
        GenArtDemoOut,
        LeanDemo,
        MarkdownDemo,
    )
    from aii_pipeline.prompts.steps._4_gen_paper_repo._4_gen_full_paper.out_schema import (
        GenPaperRepoOut,
    )
    from aii_pipeline.prompts.steps._4_gen_paper_repo.out_schema import (
        DeployGhOut,
        GenRepoOut,
    )
    from aii_pipeline.steps._1_seed_hypo.seed_hypo import SeedHypoGroup

    # Substep Module subclasses.
    from aii_pipeline.steps._2_hypo_loop._1_gen_hypo import GenHypoModule
    from aii_pipeline.steps._2_hypo_loop._2_review_hypo import ReviewHypoModule
    from aii_pipeline.steps._2_hypo_loop.hypo_loop import HypoLoopGroup
    from aii_pipeline.steps._3_invention_loop._1_gen_strat import GenStratModule
    from aii_pipeline.steps._3_invention_loop._2_gen_plan import GenPlanModule
    from aii_pipeline.steps._3_invention_loop._3_gen_art import GenArtModule
    from aii_pipeline.steps._3_invention_loop._4_gen_paper_text import (
        GenPaperTextModule,
    )
    from aii_pipeline.steps._3_invention_loop._5_review_paper import ReviewPaperModule
    from aii_pipeline.steps._3_invention_loop._6_upd_hypo import UpdHypoModule
    from aii_pipeline.steps._3_invention_loop.invention_loop import InventionLoopGroup
    from aii_pipeline.steps._4_gen_paper_repo._1_gen_repo import GenRepoModule
    from aii_pipeline.steps._4_gen_paper_repo._2_gen_viz import GenVizModule
    from aii_pipeline.steps._4_gen_paper_repo._3_gen_art_demo import GenArtDemoModule
    from aii_pipeline.steps._4_gen_paper_repo._4_gen_full_paper import (
        GenFullPaperModule,
    )
    from aii_pipeline.steps._4_gen_paper_repo._5_deploy_gh import DeployGhModule
    from aii_pipeline.steps._4_gen_paper_repo.gen_paper_repo import GenPaperRepoGroup

    pipeline_mdgroup = Annotated[
        SeqMdGroup
        | LoopMdGroup
        | SeedHypoGroup
        | HypoLoopGroup
        | InventionLoopGroup
        | GenPaperRepoGroup,
        Field(discriminator="kind"),
    ]
    pipeline_module = Annotated[
        SingleTModule
        | ParallelTModule
        | GenHypoModule
        | ReviewHypoModule
        | GenStratModule
        | GenPlanModule
        | GenArtModule
        | GenPaperTextModule
        | ReviewPaperModule
        | UpdHypoModule
        | GenRepoModule
        | GenVizModule
        | GenArtDemoModule
        | GenFullPaperModule
        | DeployGhModule,
        Field(discriminator="kind"),
    ]
    pipeline_output = Annotated[
        SeedHypoOut
        | Hypothesis
        | GenHypoOut
        | ReviewHypoOut
        | HypoLoopOut
        | Strategy
        | Strategies
        | GenStratOut
        | BasePlan
        | ProofPlan
        | ResearchPlan
        | DatasetPlan
        | ExperimentPlan
        | EvaluationPlan
        | GenPlanOut
        | BaseArtifact
        | ResearchArtifact
        | EvaluationArtifact
        | ExperimentArtifact
        | DatasetArtifact
        | ProofArtifact
        | PaperText
        | ReviewerFeedback
        | RevisedHypothesis
        | InventionLoopOut
        | Figure
        | VizFigureOutput
        | BaseDemo
        | CodeDemo
        | LeanDemo
        | MarkdownDemo
        | GenArtDemoOut
        | GenPaperRepoOut
        | DeployGhOut
        | GenRepoOut,
        Field(discriminator="kind"),
    ]
    pipeline_output_optional = pipeline_output | None

    # Direct field mutation: pydantic's ``model_rebuild(force=True)``
    # alone doesn't re-evaluate annotations that were already resolved
    # to a concrete union at class build time. Replacing the field's
    # ``annotation`` directly is the documented mechanism for runtime
    # type widening.
    #
    # Also: each subclass gets its OWN ``FieldInfo`` copy at class
    # build time (not shared with the parent). So mutating
    # ``SeqMdGroup.model_fields["children"]`` doesn't propagate to
    # ``SeedHypoGroup.model_fields["children"]`` — we have to update
    # each phase subclass separately, then rebuild each.
    Run.model_fields["children"].annotation = list[pipeline_mdgroup]
    LoopIteration.model_fields["children"].annotation = list[pipeline_module]
    SeqMdGroup.model_fields["children"].annotation = list[pipeline_module]
    # Phase subclass copies of the children field
    SeedHypoGroup.model_fields["children"].annotation = list[pipeline_module]
    GenPaperRepoGroup.model_fields["children"].annotation = list[pipeline_module]
    # LoopMdGroup subclasses keep ``children: list[LoopIteration]`` —
    # iterations are typed at the LoopIteration level, not on the
    # group, so we don't touch their ``children`` annotation.

    # Output annotation widening — every AIINode subclass (run-tree
    # nodes + messages) carries its own ``output`` FieldInfo copy
    # whose annotation defaults to ``Any``. Walk the transitive
    # subclass tree and replace each one's annotation with the
    # ``pipeline_output | None`` discriminated union so JSON replay
    # of ``*_output`` events deserializes back into typed models.
    aii_subclasses: set[type] = {AIINode}
    _stack = [AIINode]
    while _stack:
        _cls = _stack.pop()
        for _sub in _cls.__subclasses__():
            if _sub not in aii_subclasses:
                aii_subclasses.add(_sub)
                _stack.append(_sub)
    for _cls in aii_subclasses:
        if "output" in _cls.model_fields:
            _cls.model_fields["output"].annotation = pipeline_output_optional

    # Rebuild order matters: pydantic compiles each model's schema
    # against its CURRENT nested schemas at rebuild time, then caches.
    # We must rebuild leaves first so parents pick up the widened
    # leaf schemas. Otherwise Run's cached schema embeds the OLD
    # (base-only) version of nested mappings.
    #
    # Composition layers (leaves → roots):
    #   L0: BaseMessage subclasses (leaves alongside the run tree;
    #       referenced from every AIINode via ``messages``).
    #   L1: Task / ClaudeAgentTask  — leaves of the run tree.
    #   L2: Module + SingleT/ParallelT + 13 substep modules
    #       — children = list[Task]; rebuild after L1.
    #   L3: LoopIteration — children = list[Module]; rebuild after L2.
    #   L4: MdGroup + SeqMdGroup/LoopMdGroup + 4 phase subclasses
    #       — children = list[Module] (Seq) or list[LoopIteration]
    #       (Loop); rebuild after L3.
    #   L5: Run — children = list[MdGroup]; rebuild after L4.
    #   L6: AIINode — base; subclasses don't embed its compiled
    #       schema, so order vs. subclasses doesn't matter; rebuilt
    #       last for tidiness.
    for _cls in aii_subclasses:
        if issubclass(_cls, BaseMessage):
            _cls.model_rebuild(force=True)
    BaseMessage.model_rebuild(force=True)

    Task.model_rebuild(force=True)
    ClaudeAgentTask.model_rebuild(force=True)

    Module.model_rebuild(force=True)
    SingleTModule.model_rebuild(force=True)
    ParallelTModule.model_rebuild(force=True)
    GenHypoModule.model_rebuild(force=True)
    ReviewHypoModule.model_rebuild(force=True)
    GenStratModule.model_rebuild(force=True)
    GenPlanModule.model_rebuild(force=True)
    GenArtModule.model_rebuild(force=True)
    GenPaperTextModule.model_rebuild(force=True)
    ReviewPaperModule.model_rebuild(force=True)
    UpdHypoModule.model_rebuild(force=True)
    GenRepoModule.model_rebuild(force=True)
    GenVizModule.model_rebuild(force=True)
    GenArtDemoModule.model_rebuild(force=True)
    GenFullPaperModule.model_rebuild(force=True)
    DeployGhModule.model_rebuild(force=True)

    LoopIteration.model_rebuild(force=True)

    MdGroup.model_rebuild(force=True)
    SeqMdGroup.model_rebuild(force=True)
    LoopMdGroup.model_rebuild(force=True)
    SeedHypoGroup.model_rebuild(force=True)
    GenPaperRepoGroup.model_rebuild(force=True)
    HypoLoopGroup.model_rebuild(force=True)
    InventionLoopGroup.model_rebuild(force=True)

    Run.model_rebuild(force=True)
    AIINode.model_rebuild(force=True)

    _BOUND = True


__all__ = ["bind_pipeline_typed_unions"]
