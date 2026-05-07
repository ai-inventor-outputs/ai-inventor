"""Pipeline steps — phase-orchestration logic.

Phase execution lives on the typed :class:`MdGroup` subclasses defined
in :mod:`aii_pipeline.run.scaffold` (``SeedHypoGroup`` /
``HypoLoopGroup`` / ``InventionLoopGroup`` / ``GenPaperRepoGroup``);
each carries an ``execute()`` method that the pipeline runner calls in
order. The ``run_*_module`` free functions inside each ``_N_phase/``
folder hold the per-phase orchestration body and are called from the
group's ``execute()``.

:data:`PIPELINE_SEQUENCE` is the canonical ordering for cli-supplied
``--first-step`` / ``--last-step`` validation; the pipeline runner
itself walks ``run.children`` to drive execution.
"""

PIPELINE_SEQUENCE: list[str] = [
    "seed_hypo",
    "hypo_loop",
    "invention_loop",
    "gen_paper_repo",
]
