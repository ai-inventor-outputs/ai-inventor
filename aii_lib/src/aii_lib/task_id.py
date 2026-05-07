"""Canonical task_id parser/builder.

Pipeline task_ids are stringly-typed with multiple overlapping shapes,
historically parsed in three places with three different regexes:

  - inject_user_prompt._TASK_ID_PERMISSIVE_RE
  - _2_gen_plan._id_re   (strips ``_idx<N>``)
  - aii_frontend/lib/task-id.ts (TS mirror)

Every change to the format (most recently the per-run ``_<run_suffix>``)
risked silent breakage in any of those locations. This module is the
single source of truth on the Python side.

Format components (all but ``step`` optional):

  [t{seq}_]{step}[_it{iter}][_{slot}][__{model}][_{run_suffix}][_idx{idx}]

Slot vs iter ordering varies per substep (current observed shapes):

  ── single-task ──────────────────────────────────────────────────────────
    hypo_it1__haiku_44490324
    rev_hypo_it2__haiku_44490324
    strat_it1__opus_44490324
    paper_text_it1__opus_44490324
    rev_paper_it1__opus_44490324
    upd_hypo_it1__opus_44490324

  ── parallel: iter-then-slot ─────────────────────────────────────────────
    plan_it1_research_id1__haiku_44490324
    art_it1_research_id1__opus_44490324

  ── parallel: slot-then-iter ─────────────────────────────────────────────
    demo_research_id1_it1__haiku_44490324

  ── parallel: no iteration ───────────────────────────────────────────────
    img_viz_fig1__opus_44490324
    full_paper_main__opus_44490324

  ── strategy index suffix ────────────────────────────────────────────────
    strat_it1__opus_44490324_idx0
    plan_it1_research_id1__haiku_44490324_idx2

  ── transient sequence prefix (telemetry-only, not stored on disk) ───────
    t1_hypo_it1__haiku_44490324
    t4_plan_it2_research_id1__sonnet_44490324
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Regex components (composed at module load — validated by parse())
# ---------------------------------------------------------------------------

# t<N>_ sequence prefix (transient — telemetry sequencing, not stored)
_SEQ_RE = re.compile(r"^t(\d+)_")

# _idx<N> trailing index (strategy strats_per_call > 1, plan strats_per_call > 1)
_IDX_RE = re.compile(r"_idx(\d+)$")

# 2-digit run_suffix (last group before _idx<N>, after the model)
_RUN_SUFFIX_RE = re.compile(r"_(\d{2})$")

# Iteration token (anchored on word boundary so "research_id1" doesn't match)
_IT_RE = re.compile(r"(?:^|_)it(\d+)(?:_|$)")

# Slot pattern: <type>_id<N> (research_id1, experiment_id2). The
# (?:^|_) anchor avoids capturing a leading underscore — without it,
# matching against "it1_research_id1" yields "_research_id1".
# Demo/viz also have slot ids like "fig1" (no _id<N>) — those are caught by
# the fallback token-split branch in parse().
_SLOT_ID_RE = re.compile(r"(?:^|_)([a-z]+(?:_[a-z]+)*_id\d+)")

# Step prefixes — sorted longest-first so e.g. "rev_hypo" matches before "hypo".
# The list is the source of truth for known steps; unknown prefixes return None
# from parse() so callers can short-circuit on bespoke task ids (KG_*).
STEP_PREFIXES: list[str] = [
    "rev_hypo",  # must come before "hypo"
    "rev_paper",
    "upd_hypo",
    "paper_text",
    "img_viz",
    "full_paper",
    "deploy_gh",
    "hypo",
    "strat",
    "plan",
    "art",
    "demo",
    "viz",
    "repo",
]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskID:
    """Parsed pipeline task_id.

    Immutable; construct new instances via
    ``replace(tid, model="opus")`` or the ``with_*`` methods below.
    """

    step: str  # canonical step token, e.g. "plan", "hypo"
    iteration: int | None = None  # ``it<N>`` value
    slot: str | None = None  # parallel slot id, e.g. "research_id1"
    model: str | None = None  # model_short, e.g. "haiku"
    run_suffix: str | None = None  # 2-digit per-run uniqueness tail
    idx: int | None = None  # strats_per_call/plans_per index
    seq: int | None = None  # transient t<N>_ telemetry sequence

    # ── parsing ────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, task_id: str) -> TaskID | None:
        """Return a TaskID for ``task_id``, or None if it doesn't match any registered step shape.

        KG_* and other bespoke ids return None.
        """
        if not task_id:
            return None

        rest = task_id

        # Pop the optional sequence prefix.
        seq: int | None = None
        seq_m = _SEQ_RE.match(rest)
        if seq_m:
            seq = int(seq_m.group(1))
            rest = rest[seq_m.end() :]

        # Identify the step (longest-first).
        step: str | None = None
        for s in STEP_PREFIXES:
            if rest == s or rest.startswith(s + "_"):
                step = s
                rest = rest[len(s) :]
                if rest.startswith("_"):
                    rest = rest[1:]
                break
        if step is None:
            return None

        # Pop trailing _idx<N>.
        idx: int | None = None
        idx_m = _IDX_RE.search(rest)
        if idx_m:
            idx = int(idx_m.group(1))
            rest = rest[: idx_m.start()]

        # Split at "__" to separate body from model_section.
        # Body holds (it<N>, slot, …); model_section holds (model, run_suffix).
        body = rest
        model_section: str | None = None
        if "__" in rest:
            body, model_section = rest.split("__", 1)

        # Iteration from body
        iteration: int | None = None
        it_m = _IT_RE.search(body)
        if it_m:
            iteration = int(it_m.group(1))

        # Slot from body
        slot: str | None = None
        slot_m = _SLOT_ID_RE.search(body)
        if slot_m:
            slot = slot_m.group(1)
        else:
            # Fallback: anything in body that isn't the it<N> token is a slot.
            # Used for img_viz_fig1, full_paper_main, etc. where the slot has
            # no _id<N> shape.
            tokens = [t for t in body.split("_") if t and not re.fullmatch(r"it\d+", t)]
            if tokens:
                slot = "_".join(tokens) or None

        # Model + run_suffix from model_section.
        model: str | None = None
        run_suffix: str | None = None
        if model_section is not None:
            ms = model_section
            rs_m = _RUN_SUFFIX_RE.search(ms)
            if rs_m:
                run_suffix = rs_m.group(1)
                ms = ms[: rs_m.start()]
            model = ms or None

        return cls(
            step=step,
            iteration=iteration,
            slot=slot,
            model=model,
            run_suffix=run_suffix,
            idx=idx,
            seq=seq,
        )

    # ── building ───────────────────────────────────────────────────────────

    def to_str(self, *, include_seq: bool = False, include_idx: bool = True) -> str:
        """Render back into a task_id string.

        ``include_seq=False`` by default because the sequence prefix is a
        telemetry-only transient — on-disk task_ids don't carry it.
        ``include_idx=True`` because plan/strategy callers depend on it.
        """
        parts: list[str] = []
        if include_seq and self.seq is not None:
            parts.append(f"t{self.seq}_")
        parts.append(self.step)
        body_parts: list[str] = []
        # Slot and iteration ordering — match observed shapes per step.
        if (self.step == "demo" and self.slot) or (
            self.step in ("img_viz", "viz", "full_paper", "deploy_gh", "repo") and self.slot
        ):
            body_parts.append(self.slot)
            if self.iteration is not None:
                body_parts.append(f"it{self.iteration}")
        else:
            if self.iteration is not None:
                body_parts.append(f"it{self.iteration}")
            if self.slot:
                body_parts.append(self.slot)
        if body_parts:
            parts.append("_" + "_".join(body_parts))
        # Model + run_suffix
        if self.model or self.run_suffix:
            parts.append("__")
            if self.model:
                parts.append(self.model)
            if self.run_suffix:
                if self.model:
                    parts.append(f"_{self.run_suffix}")
                else:
                    parts.append(self.run_suffix)
        if include_idx and self.idx is not None:
            parts.append(f"_idx{self.idx}")
        return "".join(parts)

    def __str__(self) -> str:
        return self.to_str()

    # ── derived forms ──────────────────────────────────────────────────────

    @property
    def base(self) -> str:
        """Step + iteration + slot. Strips model + run_suffix + idx + seq.

        Used as a stable "what task is this" key — e.g.
        ``hypo_it1`` or ``plan_it1_research_id1`` — for deduplication
        across telemetry events for the same logical task.
        """
        return replace(self, model=None, run_suffix=None, idx=None, seq=None).to_str()

    def bare(self) -> str:
        """Strips ``_idx<N>``.

        Used by gen_plan to recover the parent task_id
        from a ``{task_id}_idx{N}`` plan id.
        """
        return replace(self, idx=None).to_str()


__all__ = [
    "STEP_PREFIXES",
    "TaskID",
]
