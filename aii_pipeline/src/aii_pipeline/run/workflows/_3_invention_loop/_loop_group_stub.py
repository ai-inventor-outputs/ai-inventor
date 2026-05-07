"""Stand-in for :class:`InventionLoopGroup` that satisfies module workflows' read contract.

Module workflows (gen_strat, gen_plan, gen_art, gen_paper_text,
review_paper, upd_hypo) read from
``ctx.invention_loop_group.{get_artifacts,get_plans,get_paper_texts}``.
The full :class:`InventionLoopGroup` is heavy (back-references to
Run + accumulated state); this stub satisfies just the read-only
contract on a flat input list, which the workflow body re-validates
from its JSON-safe input.

Each accessor optionally filters by ``iteration`` via the entry's
``iteration`` attribute (set on every :class:`BaseArtifact` /
:class:`BasePlan` / :class:`PaperText`).

Internal to :mod:`aii_pipeline.run.workflows` — not exported.
Phase 6 deletes when the underlying class hierarchy is retired.
"""

from __future__ import annotations

from typing import Any


class _LoopGroupStub:
    """Read-only :class:`InventionLoopGroup` shape used by module workflows."""

    def __init__(
        self,
        artifacts: list[Any] | None = None,
        plans: list[Any] | None = None,
        paper_texts: list[Any] | None = None,
    ) -> None:
        self._artifacts = artifacts or []
        self._plans = plans or []
        self._paper_texts = paper_texts or []

    @staticmethod
    def _filter_by_iter(items: list[Any], iteration: int | None) -> list[Any]:
        if iteration is None:
            return list(items)
        return [x for x in items if getattr(x, "iteration", None) == iteration]

    def get_artifacts(self, iteration: int | None = None) -> list[Any]:
        return self._filter_by_iter(self._artifacts, iteration)

    def get_plans(self, iteration: int | None = None) -> list[Any]:
        return self._filter_by_iter(self._plans, iteration)

    def get_paper_texts(self, iteration: int | None = None) -> list[Any]:
        return self._filter_by_iter(self._paper_texts, iteration)
