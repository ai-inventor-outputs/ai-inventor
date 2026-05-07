#!/usr/bin/env python3
"""
Step 7: Generate Hypothesis Seeds.

Extracts seeds for hypothesis generation from the knowledge graph:
1. Topic Blind Spots - concepts a topic is missing from dissimilar topics

Input: data/_6_paper_triples/{run_id}/paper_triples_pr.json
Output: data/_7_hypo_seeds/{run_id}/
    - topic_blind_spots.json
"""

import json
import shutil
from pathlib import Path

from aii_lib.run import emit

from ._gen_hypo_seeds import generate_topic_blind_spots

__all__ = ["main"]


def load_papers(input_dir: Path) -> list:
    """Load papers from combined JSON file."""
    json_file = input_dir / "paper_triples_pr.json"

    if not json_file.exists():
        emit.status_public_error(f"Papers file not found: {json_file}")
        return []

    try:
        with open(json_file, encoding="utf-8") as f:
            papers = json.load(f)
        emit.status_public_success(f"Loaded {len(papers)} papers")
        return papers
    except Exception as e:
        emit.status_public_error(f"Failed to load papers: {e}")
        raise


def main(run_id: str, base_dir: Path, blind_spots_cfg: dict):
    """
    Main entry point for hypothesis seed extraction.

    Args:
        run_id: Run ID for pipeline orchestration.
        base_dir: Base directory for kg runs (passed by parent _1_seed_hypo).
        blind_spots_cfg: Plain dict mirroring kg ``gen_hypo_seeds.blind_spots``
            (min_shared_concepts, max_similarity, entity_types).
    """
    from ..constants import (
        STEP_6_PAPER_TRIPLES,
        STEP_7_HYPO_SEEDS,
    )
    from ..utils import get_run_dir

    emit.status_private_info(f"Run ID: {run_id}")

    input_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id, base_dir)
    output_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id, base_dir)

    emit.status_private_info(f"Input: {input_dir.relative_to(base_dir)}")
    emit.status_private_info(f"Output: {output_dir.relative_to(base_dir)}")

    # Check input exists before destroying output
    if not input_dir.exists():
        emit.status_public_error(f"Input directory not found: {input_dir}")
        emit.status_public_error("Run step 6 first")
        return 1

    # Clean up output directory
    if output_dir.exists():
        emit.status_private_info("Removing existing output directory")
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load papers
    emit.status_private_info("Loading papers...")
    papers = load_papers(input_dir)

    if not papers:
        emit.status_public_error("No papers loaded")
        return 1

    emit.status_public_info(f"Loaded {len(papers)} papers")

    results = {}

    # 1. Topic Blind Spots (sorted by score descending)
    emit.status_public_info("1. Finding Topic Blind Spots")
    results["blind_spots"] = generate_topic_blind_spots(
        papers,
        output_dir / "topic_blind_spots.json",
        min_shared_concepts=blind_spots_cfg.get("min_shared_concepts", 1),
        max_similarity=blind_spots_cfg.get("max_similarity", 1.0),
        entity_types=blind_spots_cfg.get("entity_types", []),  # Empty = all types
    )

    # Summary
    emit.status_public_info("Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    emit.status_public_info(f"Generated {success_count}/{total_count} seed types")

    # List output files
    output_files = list(output_dir.glob("*.json"))
    for f in output_files:
        size = f.stat().st_size
        emit.status_public_info(f"  {f.name} ({size:,} bytes)")

    emit.status_public_success("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
