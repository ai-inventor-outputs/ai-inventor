"""RunPath — chained accessors for the on-disk run directory layout.

Replaces the dozen-plus hand-concatenated path expressions like
``Path(run.run_path) / "3_invention_loop" / f"iter_{N}" / "gen_plan" /
"gen_plan_output.json"`` with ``RunPath(base).invention_loop.iter(N).gen_plan.output_json``.

Layout owned by this module (single source of truth):

    runs/<run_id>/
    ├── sinks/                      ← per-sink artifacts (channel folder = sink name)
    │   ├── clone/clone_log.jsonl              canonical event stream (CloneSink)
    │   ├── clone/clone_log_sequenced.jsonl    per-task-grouped variant (SequencedCloneSink)
    │   ├── health/.heartbeat            heartbeat liveness file (HealthSink)
    │   ├── otel/traces.jsonl            OTel spans (JSONLSpanExporter)
    │   ├── otel/metrics.jsonl           OTel metric snapshots (JSONLMetricExporter)
    │   ├── title/.title                 LLM-generated run title (TitleGeneratorSink)
    │   └── to_app/                      AppSink localhost FastAPI surface
    │       ├── .port                       bound port for discovery
    │       └── messages.jsonl              slim filtered+stripped messages
    ├── sources/                    ← per-source artifacts
    │   ├── send_message/                SendMessageSource (POST /send_message)
    │   │   └── .port                       bound port for discovery
    │   └── stop/                        StopSource (POST /stop)
    │       └── .port                       bound port for discovery
    ├── user_uploads/
    ├── 1_seed_hypo/
    ├── 2_hypo_loop/
    │   └── iter_<N>/
    │       ├── gen_hypo/
    │       └── review_hypo/
    ├── 3_invention_loop/
    │   └── iter_<N>/
    │       ├── gen_strat/
    │       │   ├── gen_strat_output.json
    │       │   └── <task_id>/
    │       ├── gen_plan/
    │       ├── gen_art/
    │       ├── gen_paper_text/
    │       ├── review_paper/
    │       └── upd_hypo/
    └── 4_gen_paper_repo/
        ├── gen_repo/
        ├── gen_viz/
        ├── gen_demos/
        ├── gen_full_paper/
        └── deploy_gh/

Renaming ``3_invention_loop → invention_loop`` (or any other layout
change) becomes a one-line edit here instead of dozens of call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPath:
    """Root of one run's on-disk layout."""

    base: Path

    @classmethod
    def for_run(cls, base: Path | str) -> RunPath:
        """Construct a RunPath from a base directory."""
        return cls(Path(base))

    # ── files at root ──────────────────────────────────────────────────────
    @property
    def messages_file(self) -> Path:
        """Canonical event stream — ``sinks/clone/clone_log.jsonl`` written by CloneSink."""
        return self.base / "sinks" / "clone" / "clone_log.jsonl"

    @property
    def messages_sequenced_file(self) -> Path:
        """Per-task-grouped variant — ``sinks/clone/clone_log_sequenced.jsonl``."""
        return self.base / "sinks" / "clone" / "clone_log_sequenced.jsonl"

    @property
    def user_uploads(self) -> Path:
        """Directory for user-uploaded files."""
        return self.base / "user_uploads"

    # ── phases ─────────────────────────────────────────────────────────────
    @property
    def seed_hypo(self) -> SeedHypoPath:
        """Access seed_hypo phase subdirectory."""
        return SeedHypoPath(self.base / "1_seed_hypo")

    @property
    def hypo_loop(self) -> HypoLoopPath:
        """Access hypo_loop phase subdirectory."""
        return HypoLoopPath(self.base / "2_hypo_loop")

    @property
    def invention_loop(self) -> InventionLoopPath:
        """Access invention_loop phase subdirectory."""
        return InventionLoopPath(self.base / "3_invention_loop")

    @property
    def gen_paper_repo(self) -> GenPaperRepoPath:
        """Access gen_paper_repo phase subdirectory."""
        return GenPaperRepoPath(self.base / "4_gen_paper_repo")

    # ── traversal helpers ──────────────────────────────────────────────────
    @property
    def run_id(self) -> str:
        """Resolve run_id from the base directory name."""
        return self.base.name


# ---------------------------------------------------------------------------
# Phase: seed_hypo (one-shot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedHypoPath:
    """Path container for seed_hypo phase."""

    base: Path

    @property
    def kg_dir(self) -> Path:
        """Knowledge graph directory."""
        return self.base / "kg"


# ---------------------------------------------------------------------------
# Phase: hypo_loop (iterated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypoLoopPath:
    """Path container for hypo_loop phase."""

    base: Path

    def iter(self, n: int) -> HypoLoopIterPath:
        """Access an iteration's subdirectory."""
        return HypoLoopIterPath(self.base / f"iter_{n}")


@dataclass(frozen=True)
class HypoLoopIterPath:
    """Path container for a hypo_loop iteration."""

    base: Path

    @property
    def gen_hypo(self) -> SubstepPath:
        """Access gen_hypo substep."""
        return SubstepPath(self.base / "gen_hypo", "gen_hypo")

    @property
    def review_hypo(self) -> SubstepPath:
        """Access review_hypo substep."""
        return SubstepPath(self.base / "review_hypo", "review_hypo")


# ---------------------------------------------------------------------------
# Phase: invention_loop (iterated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventionLoopPath:
    """Path container for invention_loop phase."""

    base: Path

    @property
    def iter(self, n: int) -> InventionLoopIterPath:
        """Access an iteration's subdirectory."""
        return InventionLoopIterPath(self.base / f"iter_{n}")


@dataclass(frozen=True)
class InventionLoopIterPath:
    """Path container for an invention_loop iteration."""

    base: Path

    @property
    def gen_strat(self) -> SubstepPath:
        """Access gen_strat substep."""
        return SubstepPath(self.base / "gen_strat", "gen_strat")

    @property
    def gen_plan(self) -> SubstepPath:
        """Access gen_plan substep."""
        return SubstepPath(self.base / "gen_plan", "gen_plan")

    @property
    def gen_art(self) -> SubstepPath:
        """Access gen_art substep."""
        return SubstepPath(self.base / "gen_art", "gen_art")

    @property
    def gen_paper_text(self) -> SubstepPath:
        """Access gen_paper_text substep."""
        return SubstepPath(self.base / "gen_paper_text", "gen_paper_text")

    @property
    def review_paper(self) -> SubstepPath:
        """Access review_paper substep."""
        return SubstepPath(self.base / "review_paper", "review_paper")

    @property
    def upd_hypo(self) -> SubstepPath:
        """Access upd_hypo substep."""
        return SubstepPath(self.base / "upd_hypo", "upd_hypo")


# ---------------------------------------------------------------------------
# Phase: gen_paper_repo (flat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenPaperRepoPath:
    """Path container for gen_paper_repo phase."""

    base: Path

    @property
    def gen_repo(self) -> SubstepPath:
        """Access gen_repo substep."""
        return SubstepPath(self.base / "gen_repo", "gen_repo")

    @property
    def gen_viz(self) -> SubstepPath:
        """Access gen_viz substep."""
        return SubstepPath(self.base / "gen_viz", "gen_viz")

    @property
    def gen_demos(self) -> SubstepPath:
        """Access gen_demos substep."""
        return SubstepPath(self.base / "gen_demos", "gen_demos")

    @property
    def gen_full_paper(self) -> SubstepPath:
        """Access gen_full_paper substep."""
        return SubstepPath(self.base / "gen_full_paper", "gen_full_paper")

    @property
    def deploy_gh(self) -> SubstepPath:
        """Access deploy_gh substep."""
        return SubstepPath(self.base / "deploy_gh", "deploy_gh")


# ---------------------------------------------------------------------------
# Substep — leaf node with the substep name baked in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstepPath:
    """Path container for a substep within a phase."""

    base: Path
    name: str  # canonical substep name (gen_strat, gen_plan, etc.)

    @property
    def output_json(self) -> Path:
        """The substep's primary output file: ``<name>_output.json``."""
        return self.base / f"{self.name}_output.json"

    def task_dir(self, task_id: str) -> Path:
        """Per-task working directory inside the substep dir."""
        return self.base / task_id


# ---------------------------------------------------------------------------
# Per-task workspace sweep — destructive truncate utility
# ---------------------------------------------------------------------------


_TASK_DIR_RE = re.compile(
    r"(?:^|/)(?P<phase>[1-4]_\w+)/iter_(?P<iter>\d+)/(?P<step>[a-z_]+)/(?P<task_id>[^/]+)$"
)
"""Matches per-task workspace dirs:
   {phase}/iter_{N}/{step}/{task_id}/  — e.g. 3_invention_loop/iter_2/gen_strat/strat_it2__opus_27100000
"""


def sweep_task_workspaces(
    run_dir: Path,
    *,
    cutoff_ts: str,
    task_started_at: dict[str, str],
) -> list[Path]:
    """Drop per-task workspace dirs whose owning task started after the cutoff.

    Walks ``run_dir`` looking for directories that match the per-task
    workspace shape (``{phase}/iter_{N}/{step}/{task_id}``). For each,
    looks up ``task_id`` in ``task_started_at`` (a map from task_id →
    isoformat ts, built by the caller from the truncated JSONL). If the
    task_start timestamp is **strictly after** ``cutoff_ts``, the
    workspace dir is removed.

    Never trusts mtime — viewers (preview, refresh) touch files. The
    cutoff is purely event-driven, derived from telemetry.

    Returns the list of paths that were removed.
    """
    import shutil
    from pathlib import Path as _P

    removed: list[_P] = []
    if not run_dir.is_dir():
        return removed

    for entry in run_dir.rglob("*"):
        if not entry.is_dir():
            continue
        rel = str(entry.relative_to(run_dir))
        m = _TASK_DIR_RE.search(rel)
        if not m:
            continue
        task_id = m.group("task_id")
        start_ts = task_started_at.get(task_id)
        # If we don't know when the task started, leave it alone — better
        # to keep stale data than delete something we can't classify.
        if start_ts is None:
            continue
        # Strict greater: tasks at-or-before the cutoff are inside the
        # kept window; only ones strictly after are post-cutoff.
        if start_ts > cutoff_ts:
            try:
                shutil.rmtree(entry)
                removed.append(entry)
            except OSError:
                pass
    return removed


__all__ = [
    "GenPaperRepoPath",
    "HypoLoopIterPath",
    "HypoLoopPath",
    "InventionLoopIterPath",
    "InventionLoopPath",
    "RunPath",
    "SeedHypoPath",
    "SubstepPath",
    "sweep_task_workspaces",
]
