"""Shared helpers for phase MdGroup ``get_*`` accessors.

Lifted out of :mod:`aii_pipeline.run.scaffold` so the per-phase
classes can live in their own ``steps/_X_phase/<phase>.py`` files
(per REFACTOR_PLAN §7) without scaffold.py becoming a circular
import via ``from aii_pipeline.steps._*_phase.<phase> import …``.

Read-only walks over the live run tree — no state, no side effects.
"""

from __future__ import annotations

from typing import Any


def _module_output_for(modules: list, module_name: str) -> Any | None:
    """Return ``Module.output`` for the named module in ``modules``.

    Walks ``modules`` (a list of Module nodes from one iteration's or
    one seq-group's children), matches by ``Module.name``, and returns
    the typed Pydantic instance left on ``module.output`` by the
    ``module_output`` event. ``None`` when the module hasn't run, the
    ``output`` slot is unset, or no module with the name is present.

    The canonical single-output reader. Multi-task modules whose
    aggregate lives on ``module.output`` (gen_strat / gen_plan /
    gen_art_demo / deploy_gh / …) and single-output modules
    (gen_hypo / review_hypo / gen_paper_text / review_paper /
    upd_hypo / gen_full_paper / gen_repo) all use this helper. For
    1:1 parallel modules (gen_art, gen_viz) where each task carries
    its own ``task.output``, accessors walk ``module.children``
    directly.
    """
    for m in modules:
        if getattr(m, "name", None) != module_name:
            continue
        out = getattr(m, "output", None)
        if out is not None:
            return out
    return None


__all__ = ["_module_output_for"]
