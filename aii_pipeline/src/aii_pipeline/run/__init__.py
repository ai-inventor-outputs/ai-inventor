"""aii_pipeline.run — AII pipeline composition layer.

In v26 the tree is::

    Run > MdGroup (Loop|Seq) > [LoopIteration >] Module > tasks/slots

UI labels (display name, card grouping, rounds membership) live in the
mapper as private inline dicts — they are presentation-layer concerns,
not domain concerns.

Every node carries an auto-generated 12-char ``node_id``. Identity is
opaque — node_ids are never constructible from outside, only returned
by the ``start_*`` methods. Pipeline call-sites capture them and flow
them through subsequent calls:

    gid = run.start_seq_group(name="gen_paper_repo")
    mid = run.start_single_module(name="gen_repo", parent_id=gid)
    run.end_module(parent_id=gid, module_id=mid)
    run.end_group(id=gid)

For loops:

    loop = run.start_loop_group(name="invention_loop")
    iter_id = run.start_iteration(group_id=loop, iteration=1)
    mid = run.start_single_module(name="gen_strat", parent_id=iter_id)
    ...
    run.end_iteration(group_id=loop, iteration=1)
"""

from __future__ import annotations

# Generic OO machinery — re-exported from aii_lib.run
from aii_lib.run import (
    AnyMdGroup,
    AnyModule,
    ClaudeAgentTask,
    LoopIteration,
    LoopMdGroup,
    MdGroup,
    Module,
    NodeStatus,
    ParallelTModule,
    Run,
    SeqMdGroup,
    SingleTModule,
    Task,
    current_run,
    get_current_run,
    set_current_run,
)

# ---------------------------------------------------------------------------
# Test helper — make_module(name, *, id=...)
# ---------------------------------------------------------------------------


_PARALLEL_NAMES: frozenset[str] = frozenset(
    {
        "gen_hypo",
        "gen_plan",
        "gen_art",
        "gen_viz",
        "gen_art_demo",
    }
)


def make_module(name: str, *, id: str | None = None) -> Module:  # noqa: A002 — `id` matches Module.node_id field; keyword-only test helper
    """Construct an empty Module for ``name`` — TEST HELPER.

    Production code uses ``Run.start_single_module`` / ``start_parallel_module``
    with explicit ``parent_id`` / ``module_id``. This helper is purely for
    test fixtures that just want a Module without the full event ceremony.

    ``name`` is the canonical step token (e.g. ``"gen_plan"``, ``"gen_hypo"``).
    Parallel by membership in ``_PARALLEL_NAMES``.
    """
    cls = ParallelTModule if name in _PARALLEL_NAMES else SingleTModule
    module_id = id or name
    return cls(node_id=module_id, name=name)


# ── Wire AII dispatcher + structure-ensure onto Run ─────
# Both leaf-safe imports — they don't load scaffold.py and therefore
# don't trigger the phase-Group import chain. Phase Group classes
# (``SeedHypoGroup`` / ``HypoLoopGroup`` / …) are NOT re-exported here
# — callers import them from ``aii_pipeline.run.scaffold`` directly.
# Re-exporting them at package level would eager-load scaffold, which
# imports each phase module, which imports utilities back inside
# ``aii_pipeline.run`` — triggering this ``__init__`` while it's
# still loading and producing a partially-initialised module
# circular-import error.
from aii_lib.run import set_ensure_for_task
from aii_lib.run.dispatch import _ensure_task_for_id, dispatch_event
from aii_lib.run.run import (
    set_dispatch,
    set_group_class_resolver,
    set_module_class_resolver,
)

from .path import RunPath


def _resolve_group_class(name: str, group_type: str) -> type[MdGroup] | None:
    """Map a phase ``name`` to its typed group class.

    Lazy-imports the scaffold module only when first called so this
    package's import-time graph stays leaf-safe (see the docstring
    above). Returns ``None`` for unknown names so dispatch falls back
    to the generic base class.
    """
    from .scaffold import _GROUP_CLASS_BY_NAME

    return _GROUP_CLASS_BY_NAME.get(name)


def _resolve_module_class(name: str, module_type: str) -> type[Module] | None:
    """Map a substep ``name`` to its typed module class.

    Lazy-resolves the substep registry built by
    :func:`scaffold._build_substep_class_registry`. Returns ``None``
    for unknown names so dispatch falls back to the generic
    SingleTModule / ParallelTModule base.
    """
    from .scaffold import _get_substep_class

    return _get_substep_class(name)


set_dispatch(dispatch_event)
set_ensure_for_task(_ensure_task_for_id)
set_group_class_resolver(_resolve_group_class)
set_module_class_resolver(_resolve_module_class)


__all__ = [
    # Core hierarchy
    "Task",
    "ClaudeAgentTask",
    "Module",
    "SingleTModule",
    "ParallelTModule",
    "AnyModule",
    "MdGroup",
    "SeqMdGroup",
    "LoopMdGroup",
    "AnyMdGroup",
    "LoopIteration",
    "Run",
    # Helpers
    "make_module",
    # Status enum
    "NodeStatus",
    # AII-specific helpers
    "RunPath",
    # Context vars
    "current_run",
    "get_current_run",
    "set_current_run",
]
