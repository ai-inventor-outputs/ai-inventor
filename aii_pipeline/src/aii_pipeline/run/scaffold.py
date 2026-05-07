"""Phase-specific MdGroup subclasses + scaffolder for the AII pipeline.

This module owns three related concerns for the live ``Run`` aggregate:

  1. **Phase MdGroup subclasses** — :class:`SeedHypoGroup` /
     :class:`HypoLoopGroup` / :class:`InventionLoopGroup` /
     :class:`GenPaperRepoGroup` extend the generic ``SeqMdGroup`` /
     ``LoopMdGroup`` from :mod:`aii_lib.run.mdgroup` with typed
     tree-walk accessors (``get_strategies`` / ``get_plans`` /
     ``get_artifacts`` / ``get_paper_texts`` / ``get_hypotheses`` /
     ``get_hypo_reviews`` / ``get_figures`` / ``get_demos``). The
     accessors reconstitute outputs by walking
     ``LoopIteration → Module.output`` on demand — the typed
     :class:`GenStratOut` / :class:`GenPlanOut` / etc. populated by
     the per-substep ``module_output`` event. No separate storage,
     no projection into ledger shapes.

  2. **Per-phase execution contract** — every phase class defines an
     async ``execute()`` (run the phase body). The pipeline runner
     iterates ``run.children`` and calls ``await phase.execute()``
     directly; there is no orchestrator wrapper class. ``execute()``
     is **required** — Python raises ``AttributeError`` at the
     runner site if a phase is missing it.

     ``get_context()`` is **optional** — define it only when the
     phase pushes a narrower typed ctx for its substeps via
     :func:`ctx_scope`. Phases without a narrower ctx (or that just
     thread ``StepContext`` through unchanged) skip ``get_context()``
     and read :func:`current_ctx` directly inside ``execute()``.
     ``SeedHypoGroup`` is the canonical example.

     The same contract applies to typed substep ``Module`` subclasses
     instantiated by the scaffolder (see ``SUBSTEP_CLASS_BY_NAME``):
     ``execute()`` mandatory, ``get_context()`` optional.

  3. **Pre-population** — :func:`scaffold_pipeline` pre-creates the
     full expected shape (loop groups + their iterations, seq groups +
     their modules), all PENDING with auto-generated node_ids, so the
     FE renders the static structure before any step fires. The
     runtime ``Run.start_*`` methods reuse the scaffold's nodes by
     name lookup.

The classes live here (not in ``aii_lib/run/``) because they're
pipeline-specific — other pipelines could embed ``aii_lib`` with
totally different domain content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from aii_lib.run.loop_iteration import LoopIteration
from aii_lib.run.module import ParallelTModule, SingleTModule

# Phase MdGroup subclasses + their ctxs — moved out of scaffold per
# REFACTOR_PLAN §7. Re-imported here so ``aii_pipeline/run/__init__.py``'s
# existing ``from .scaffold import (SeedHypoGroup, …)`` keeps working
# unchanged, and so :data:`_GROUP_CLASS_BY_NAME` can reference them at
# module level.
from aii_pipeline.steps._1_seed_hypo.seed_hypo import (
    SeedHypoCtx,  # noqa: F401  (re-exported)
    SeedHypoGroup,
)
from aii_pipeline.steps._2_hypo_loop.hypo_loop import (
    HypoLoopCtx,
    HypoLoopGroup,
)
from aii_pipeline.steps._3_invention_loop.invention_loop import (
    InventionLoopCtx,
    InventionLoopGroup,
)
from aii_pipeline.steps._4_gen_paper_repo.gen_paper_repo import (
    GenPaperRepoGroup,
    GenPaperRepoPhaseCtx,
)

if TYPE_CHECKING:
    from aii_lib.run.mdgroup import LoopMdGroup, SeqMdGroup
    from aii_lib.run.run import Run

    from aii_pipeline.utils import PipelineConfig


# ===========================================================================
# id → class lookup — used by dispatch._apply_mdgroup_start
# ===========================================================================
#
# Private to scaffold: scaffold's helpers below use this to instantiate
# typed phases from the canonical ``name`` (``"hypo_loop"`` →
# ``HypoLoopGroup``). No runtime registry crosses the lib/pipeline
# boundary anymore — fork-replay finds typed phases via find-by-name
# on the pre-scaffolded run rather than reaching into a global lookup.

_GROUP_CLASS_BY_NAME: dict[str, type[SeqMdGroup] | type[LoopMdGroup]] = {
    "seed_hypo": SeedHypoGroup,
    "hypo_loop": HypoLoopGroup,
    "invention_loop": InventionLoopGroup,
    "gen_paper_repo": GenPaperRepoGroup,
}


# ===========================================================================
# Pre-population scaffolder
# ===========================================================================
#
# The pipeline emits ``mdgroup_start`` / ``module_start`` events at
# runtime to construct the live ``Run`` aggregate. Anything not yet
# started is absent from ``run.children`` (and therefore from the FE's
# snapshot), so a fresh run only renders one card at a time as steps
# fire.
#
# This scaffolder fills in the static structure (loop groups + their
# iterations, seq groups + their modules) at boot, all with status
# ``PENDING`` and auto-generated node_ids. Canonical phase tokens
# (``"hypo_loop"``, ``"hypo"``) live in the ``name`` field, never in
# ``node_id``. The runtime ``Run.start_*`` methods look the scaffold's
# nodes up by name (or by iteration index) and reuse their auto-gen
# ids, so callers and the indexed tree stay consistent.
#
# Modules whose count is dynamic (parallel slots: number of strategies
# → plans → artifacts) are NOT pre-scaffolded — those tasks are created
# at the moment the parent module's actual events fire.

_Shape = Literal["single", "parallel"]


# Per-iteration module shape inside ``hypo_loop``. Names match the
# canonical step tokens emitted by ``steps/_2_hypo_loop/_*.py``.
# ``hypo`` is parallel because it fans out (seeded + unseeded) × models
# concurrent hypothesis-generation calls.
_HYPO_LOOP_MODULES: list[tuple[str, _Shape]] = [
    ("gen_hypo", "parallel"),
    ("review_hypo", "single"),
]

# Per-iteration module shape inside ``invention_loop``. Mirrors the
# pipeline emit sites in ``steps/_3_invention_loop/_*.py``. ``strat`` is
# parallel because it fans out models × calls_per_llm.
_INVENTION_LOOP_MODULES: list[tuple[str, _Shape]] = [
    ("gen_strat", "parallel"),
    ("gen_plan", "parallel"),
    ("gen_art", "parallel"),
    ("gen_paper_text", "single"),
    ("review_paper", "single"),
    ("upd_hypo", "single"),
]

# ``gen_paper_repo`` (seq) — one module per stage.
_GEN_PAPER_REPO_MODULES: list[tuple[str, _Shape]] = [
    ("gen_repo", "single"),
    ("gen_viz", "parallel"),
    ("gen_art_demo", "parallel"),
    ("gen_full_paper", "single"),
    ("deploy_gh", "single"),
]


def _build_substep_class_registry() -> dict[str, type[SingleTModule] | type[ParallelTModule]]:
    """Lazily resolve ``substep_name → typed Module subclass``.

    Each phase's substep files (``steps/_2_hypo_loop/_1_gen_hypo.py``,
    etc.) define a small typed ``Module`` subclass per substep whose
    ``execute()`` delegates to the existing ``run_*_module`` free
    function. Registered here so :func:`_make_module` constructs the
    typed instance at scaffold time — once the live ``Run`` is built,
    each substep node is the right pipeline-aware class with its own
    ``execute(...)``. Imports happen lazily inside the call so this
    module's import doesn't cycle through the substep tree.
    """
    from aii_pipeline.steps._2_hypo_loop._1_gen_hypo import GenHypoModule
    from aii_pipeline.steps._2_hypo_loop._2_review_hypo import ReviewHypoModule
    from aii_pipeline.steps._3_invention_loop._1_gen_strat import GenStratModule
    from aii_pipeline.steps._3_invention_loop._2_gen_plan import GenPlanModule
    from aii_pipeline.steps._3_invention_loop._3_gen_art import GenArtModule
    from aii_pipeline.steps._3_invention_loop._4_gen_paper_text import GenPaperTextModule
    from aii_pipeline.steps._3_invention_loop._5_review_paper import ReviewPaperModule
    from aii_pipeline.steps._3_invention_loop._6_upd_hypo import UpdHypoModule
    from aii_pipeline.steps._4_gen_paper_repo._1_gen_repo import GenRepoModule
    from aii_pipeline.steps._4_gen_paper_repo._2_gen_viz import GenVizModule
    from aii_pipeline.steps._4_gen_paper_repo._3_gen_art_demo import GenArtDemoModule
    from aii_pipeline.steps._4_gen_paper_repo._4_gen_full_paper import GenFullPaperModule
    from aii_pipeline.steps._4_gen_paper_repo._5_deploy_gh import DeployGhModule

    return {
        "gen_hypo": GenHypoModule,
        "review_hypo": ReviewHypoModule,
        "gen_strat": GenStratModule,
        "gen_plan": GenPlanModule,
        "gen_art": GenArtModule,
        "gen_paper_text": GenPaperTextModule,
        "review_paper": ReviewPaperModule,
        "upd_hypo": UpdHypoModule,
        "gen_repo": GenRepoModule,
        "gen_viz": GenVizModule,
        "gen_art_demo": GenArtDemoModule,
        "gen_full_paper": GenFullPaperModule,
        "deploy_gh": DeployGhModule,
    }


_SUBSTEP_CLASS_REGISTRY: dict[str, type[SingleTModule] | type[ParallelTModule]] | None = None


def _get_substep_class(name: str) -> type[SingleTModule] | type[ParallelTModule] | None:
    """Return the typed substep class registered for ``name``, or None when unknown.

    Used by ``aii_pipeline.run.__init__``'s ``set_module_class_resolver``
    hook so dispatch can construct typed substep modules during
    clone-log replay (no scaffold step needed).
    """
    global _SUBSTEP_CLASS_REGISTRY
    if _SUBSTEP_CLASS_REGISTRY is None:
        _SUBSTEP_CLASS_REGISTRY = _build_substep_class_registry()
    return _SUBSTEP_CLASS_REGISTRY.get(name)


def _path_id_for_attach(parent: Any, name: str) -> tuple[str, str]:
    """Compute ``(path, node_id)`` for a new ``name`` child of ``parent``.

    Used by :meth:`Run.attach`. Scaffold runs before any live emit, so
    counting same-named siblings on ``parent`` gives the same index a
    runtime emitter's ``Run._next_emit_index`` would produce in canonical
    pipeline order — both schemes agree on idx=0 for unique-name siblings
    (every scaffold-attached node), keeping ids consistent across
    scaffold, live emit, and replay re-walks.
    """
    from aii_lib.run.node_id import gen_path_id

    parent_path = getattr(parent, "path", "") or ""
    sibling_count = sum(
        1 for c in getattr(parent, "children", []) if getattr(c, "name", None) == name
    )
    path = f"{parent_path}/{name}[{sibling_count}]"
    return path, gen_path_id(name, path)


def _make_module(name: str, shape: _Shape, parent: Any) -> SingleTModule | ParallelTModule:
    """Construct a substep node — typed subclass when registered, else fallback.

    The registry is built lazily on first call so ``scaffold.py``'s
    own import doesn't have to traverse the entire substep tree.
    Registry shape is enforced — the typed subclass must inherit from
    the expected ``shape`` (``parallel`` → :class:`ParallelTModule`,
    ``single`` → :class:`SingleTModule``).
    """
    global _SUBSTEP_CLASS_REGISTRY
    if _SUBSTEP_CLASS_REGISTRY is None:
        _SUBSTEP_CLASS_REGISTRY = _build_substep_class_registry()
    cls = _SUBSTEP_CLASS_REGISTRY.get(name)
    if cls is None:
        cls = ParallelTModule if shape == "parallel" else SingleTModule
    path, mid = _path_id_for_attach(parent, name)
    return cls(node_id=mid, name=name, parent_id=parent.node_id, path=path)


def _ensure_loop_group(
    run: Run,
    *,
    name: str,
    iterations: int,
    modules_per_iter: list[tuple[str, _Shape]],
) -> None:
    """Pre-create a LoopMdGroup with N PENDING iterations + per-iter modules.

    Idempotent — runs both for fresh runs (creates everything) and for
    resumes / forks (tops up missing structure left by partial clone-log
    replay). Specifically: clone-log replay only re-creates nodes the
    parent run actually emitted events for, so a fork from mid-iter
    leaves that iteration with only its already-started modules. Without
    the topup pass below, the next module-access in the loop body
    raises ``KeyError`` because ``substeps[<name>]`` misses the
    preempted siblings.

    Existing nodes are preserved verbatim (status, ids, children); only
    missing iterations and missing per-iteration modules get attached as
    fresh PENDING stubs.
    """
    group = run.find_group_by_name(name)
    if group is None:
        cls = _GROUP_CLASS_BY_NAME[name]
        path, gid = _path_id_for_attach(run, name)
        group = cls(node_id=gid, name=name, parent_id=run.node_id, path=path)
        run.attach(group)

    # If the replayed loop is already ``done`` (e.g. parent early-stopped
    # at iter2 with config max_iter=3), don't pad trailing iters. They
    # would never run — the loop body has already exited — but the FE
    # would still render them as ``pending`` rounds, making a completed
    # loop look like it has unfinished work.
    if getattr(group, "status", None) == "done":
        return

    # Per :class:`LoopIteration` design, the 1-based iteration number IS
    # the iteration's position in ``group.children``. Clone-log replay
    # processes ``iteration_start`` events in order and appends — so
    # children[N-1] is iter{N}. Top up any missing trailing iterations
    # without touching what's already there (whether replay-attached or
    # scaffold-attached on a prior call).
    existing_count = len(group.children)
    for i in range(iterations):
        iteration_number = i + 1
        iter_name = f"iter{iteration_number}"
        if iteration_number <= existing_count:
            existing_iter = group.children[iteration_number - 1]
        else:
            iter_path, iter_id = _path_id_for_attach(group, iter_name)
            existing_iter = LoopIteration(
                node_id=iter_id,
                name=iter_name,
                parent_id=group.node_id,
                path=iter_path,
            )
            run.attach(existing_iter, parent=group)

        existing_module_names = {m.name for m in existing_iter.children}
        for module_name, shape in modules_per_iter:
            if module_name in existing_module_names:
                continue
            module = _make_module(module_name, shape, existing_iter)
            run.attach(module, parent=existing_iter)


def _ensure_seq_group(
    run: Run,
    *,
    name: str,
    modules: list[tuple[str, _Shape]],
) -> None:
    """Pre-create a SeqMdGroup with N PENDING modules.

    Idempotent — same topup semantics as :func:`_ensure_loop_group`.
    """
    group = run.find_group_by_name(name)
    if group is None:
        cls = _GROUP_CLASS_BY_NAME[name]
        path, gid = _path_id_for_attach(run, name)
        group = cls(node_id=gid, name=name, parent_id=run.node_id, path=path)
        run.attach(group)

    existing_module_names = {m.name for m in group.children}
    for module_name, shape in modules:
        if module_name in existing_module_names:
            continue
        module = _make_module(module_name, shape, group)
        run.attach(module, parent=group)


def _ensure_empty_seq_group(run: Run, *, name: str) -> None:
    """Pre-create a SeqMdGroup with no submodules.

    Used for phases whose substep tree depends on dynamic config
    branches that aren't worth modelling at scaffold time. The phase's
    own body emits the submodule events at runtime.
    """
    if run.find_group_by_name(name) is not None:
        return
    cls = _GROUP_CLASS_BY_NAME[name]
    path, gid = _path_id_for_attach(run, name)
    group = cls(node_id=gid, name=name, parent_id=run.node_id, path=path)
    run.attach(group)


def scaffold_pipeline(run: Run, config: PipelineConfig) -> None:
    """Pre-populate ``run.children`` with every expected group + module.

    Call this once at pipeline init *after* ``Run(node_id=...)`` and *before*
    the first step runs. All nodes start PENDING with auto-gen ids;
    subsequent ``mdgroup_start`` / ``module_start`` events flip statuses
    in place via name-based lookup.

    Phases scaffolded here in canonical pipeline order so
    ``run.children`` IS the pipeline sequence — :func:`run_pipeline`
    iterates it directly to drive execution.

    Notes:
      - ``seed_hypo`` is scaffolded as an empty SeqMdGroup. Its
        sub-modules (invention_kg / sample_seeds / …) depend on
        dynamic config branches the phase's own body emits at runtime.
      - Parallel modules' tasks are **not** pre-populated: slot counts
        (e.g. how many plans / artifacts) are derived at runtime from
        the strategies generated upstream.
    """
    hypo_iters = int(config.gen_hypo_loop.max_iterations)
    inv_iters = int(config.invention_loop.max_iterations)

    _ensure_empty_seq_group(run, name="seed_hypo")
    _ensure_loop_group(
        run,
        name="hypo_loop",
        iterations=hypo_iters,
        modules_per_iter=_HYPO_LOOP_MODULES,
    )
    _ensure_loop_group(
        run,
        name="invention_loop",
        iterations=inv_iters,
        modules_per_iter=_INVENTION_LOOP_MODULES,
    )
    _ensure_seq_group(
        run,
        name="gen_paper_repo",
        modules=_GEN_PAPER_REPO_MODULES,
    )


__all__ = [
    # Phase MdGroup subclasses
    "SeedHypoGroup",
    "HypoLoopGroup",
    "InventionLoopGroup",
    "GenPaperRepoGroup",
    # Phase ctxs (SeedHypoGroup intentionally has no ctx — see its docstring)
    "HypoLoopCtx",
    "InventionLoopCtx",
    "GenPaperRepoPhaseCtx",
    # Scaffolder
    "scaffold_pipeline",
]
