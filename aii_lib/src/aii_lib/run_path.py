"""RunPath — chained accessors for the on-disk run directory layout.

Replaces the dozen-plus hand-concatenated path expressions like
``Path(run.run_path) / "3_invention_loop" / f"iter_{N}" / "gen_plan" /
"gen_plan_output.json"`` with ``RunPath(base).invention_loop.iter(N).gen_plan.output_json``.

Layout owned by this module (single source of truth):

    runs/<run_id>/
    ├── sinks/                      ← per-sink artifacts (channel folder = sink name)
    │   ├── clone/clone_log.jsonl              canonical event stream (CloneSink)
    │   ├── clone/clone_log_sequenced.jsonl    per-task-grouped variant
    │   ├── health/.heartbeat            heartbeat liveness (HealthSink)
    │   ├── otel/{traces,metrics}.jsonl  OTel exports
    │   ├── title/.title                 LLM-generated run title
    │   └── to_app/{.port,messages.jsonl} AppSink port + slim messages
    ├── sources/                    ← per-source artifacts
    │   ├── send_message/.port      SendMessageSource port file
    │   └── stop/.port              StopSource port file
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
        """Construct from a base path."""
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
        """User upload directory."""
        return self.base / "user_uploads"

    # ── phases ─────────────────────────────────────────────────────────────
    @property
    def seed_hypo(self) -> SeedHypoPath:
        """Seed hypothesis phase accessor."""
        return SeedHypoPath(self.base / "1_seed_hypo")

    @property
    def hypo_loop(self) -> HypoLoopPath:
        """Hypothesis loop phase accessor."""
        return HypoLoopPath(self.base / "2_hypo_loop")

    @property
    def invention_loop(self) -> InventionLoopPath:
        """Invention loop phase accessor."""
        return InventionLoopPath(self.base / "3_invention_loop")

    @property
    def gen_paper_repo(self) -> GenPaperRepoPath:
        """Paper repo generation phase accessor."""
        return GenPaperRepoPath(self.base / "4_gen_paper_repo")

    # ── traversal helpers ──────────────────────────────────────────────────
    @property
    def run_id(self) -> str:
        """Extract run ID from base path."""
        return self.base.name


# ---------------------------------------------------------------------------
# Phase: seed_hypo (one-shot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedHypoPath:
    """Seed hypothesis phase path accessor."""

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
    """Hypothesis loop phase path accessor."""

    base: Path

    def iter(self, n: int) -> HypoLoopIterPath:
        """Access iteration path by number."""
        return HypoLoopIterPath(self.base / f"iter_{n}")


@dataclass(frozen=True)
class HypoLoopIterPath:
    """Hypothesis loop iteration path accessor."""

    base: Path

    @property
    def gen_hypo(self) -> SubstepPath:
        """Generate hypothesis substep accessor."""
        return SubstepPath(self.base / "gen_hypo", "gen_hypo")

    @property
    def review_hypo(self) -> SubstepPath:
        """Review hypothesis substep accessor."""
        return SubstepPath(self.base / "review_hypo", "review_hypo")


# ---------------------------------------------------------------------------
# Phase: invention_loop (iterated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventionLoopPath:
    """Invention loop phase path accessor."""

    base: Path

    def iter(self, n: int) -> InventionLoopIterPath:
        """Access iteration path by number."""
        return InventionLoopIterPath(self.base / f"iter_{n}")


@dataclass(frozen=True)
class InventionLoopIterPath:
    """Invention loop iteration path accessor."""

    base: Path

    @property
    def gen_strat(self) -> SubstepPath:
        """Generate strategy substep accessor."""
        return SubstepPath(self.base / "gen_strat", "gen_strat")

    @property
    def gen_plan(self) -> SubstepPath:
        """Generate plan substep accessor."""
        return SubstepPath(self.base / "gen_plan", "gen_plan")

    @property
    def gen_art(self) -> SubstepPath:
        """Generate art substep accessor."""
        return SubstepPath(self.base / "gen_art", "gen_art")

    @property
    def gen_paper_text(self) -> SubstepPath:
        """Generate paper text substep accessor."""
        return SubstepPath(self.base / "gen_paper_text", "gen_paper_text")

    @property
    def review_paper(self) -> SubstepPath:
        """Review paper substep accessor."""
        return SubstepPath(self.base / "review_paper", "review_paper")

    @property
    def upd_hypo(self) -> SubstepPath:
        """Update hypothesis substep accessor."""
        return SubstepPath(self.base / "upd_hypo", "upd_hypo")


# ---------------------------------------------------------------------------
# Phase: gen_paper_repo (flat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenPaperRepoPath:
    """Paper repo generation phase path accessor."""

    base: Path

    @property
    def gen_repo(self) -> SubstepPath:
        """Generate repository substep accessor."""
        return SubstepPath(self.base / "gen_repo", "gen_repo")

    @property
    def gen_viz(self) -> SubstepPath:
        """Generate visualizations substep accessor."""
        return SubstepPath(self.base / "gen_viz", "gen_viz")

    @property
    def gen_demos(self) -> SubstepPath:
        """Generate demos substep accessor."""
        return SubstepPath(self.base / "gen_demos", "gen_demos")

    @property
    def gen_full_paper(self) -> SubstepPath:
        """Generate full paper substep accessor."""
        return SubstepPath(self.base / "gen_full_paper", "gen_full_paper")

    @property
    def deploy_gh(self) -> SubstepPath:
        """Deploy to GitHub substep accessor."""
        return SubstepPath(self.base / "deploy_gh", "deploy_gh")


# ---------------------------------------------------------------------------
# Substep — leaf node with the substep name baked in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstepPath:
    """Substep path accessor."""

    base: Path
    name: str  # canonical substep name (gen_strat, gen_plan, etc.)

    @property
    def output_json(self) -> Path:
        """The substep's primary output file: ``<name>_output.json``."""
        return self.base / f"{self.name}_output.json"

    def task_dir(self, task_id: str) -> Path:
        """Per-task working directory inside the substep dir."""
        return self.base / task_id


__all__ = [
    "GenPaperRepoPath",
    "HypoLoopIterPath",
    "HypoLoopPath",
    "InventionLoopIterPath",
    "InventionLoopPath",
    "RunPath",
    "SeedHypoPath",
    "SubstepPath",
]
