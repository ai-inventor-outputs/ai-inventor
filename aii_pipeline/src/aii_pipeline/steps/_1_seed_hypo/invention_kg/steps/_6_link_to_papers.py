#!/usr/bin/env python3
"""
Step 6: Add Papers to Triples.

Combines clean paper data with Wikidata-enriched triples.

Input:
  - data/_3_papers_clean/{run_id}/ (clean paper.json files)
  - data/_5_wikidata/{run_id}/ (enriched triples.json files)

Output:
  - data/_6_paper_triples/{run_id}/paper_triples_pr.json

The combined output has structure:
{
  "index": 0,
  "paper": {...},  # Clean paper data
  "triples": {...}  # Enriched triples with wikidata
}
"""

import json
from pathlib import Path
from typing import Any

from aii_lib.run import emit


def load_paper_data(paper_dir: Path) -> dict[str, Any] | None:
    """
    Load paper.json from paper directory.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Paper data dictionary, or None if not found/invalid
    """
    paper_file = paper_dir / "paper.json"

    if not paper_file.exists():
        emit.status_public_warning(f"No paper.json found in {paper_dir.name}")
        return None

    try:
        with open(paper_file, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        emit.status_public_error(f"Failed to parse JSON in {paper_file}: {e}")
        return None
    except Exception as e:
        emit.status_public_error(f"Error loading {paper_file}: {e}")
        return None


def load_enriched_triples(triples_dir: Path) -> dict[str, Any] | None:
    """
    Load triples.json from enriched triples directory (step 5 output).

    Args:
        triples_dir: Path to the paper's triples directory

    Returns:
        Triples data, or None if not found/invalid
    """
    triples_file = triples_dir / "triples.json"

    if not triples_file.exists():
        return None

    try:
        with open(triples_file, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        emit.status_public_error(f"Failed to parse JSON in {triples_file}: {e}")
        return None
    except Exception as e:
        emit.status_public_error(f"Error loading {triples_file}: {e}")
        return None


def process_papers(clean_papers_dir: Path, enriched_triples_dir: Path) -> dict[str, Any]:
    """
    Process all paper directories and combine clean papers with enriched triples.

    Args:
        clean_papers_dir: Directory containing paper_XXXXX/ folders with paper.json
        enriched_triples_dir: Directory containing paper_XXXXX/ folders with triples.json

    Returns:
        Dictionary with combined papers and statistics
    """
    emit.status_public_info("Combining papers with enriched triples")

    # Get all paper directories from clean_papers_dir
    paper_dirs = sorted(clean_papers_dir.glob("paper_*"))
    if not paper_dirs:
        emit.status_public_warning(f"No paper directories found in {clean_papers_dir}")
        return {
            "combined_papers": [],
            "stats": {
                "total": 0,
                "not_enriched": 0,
                "no_triples": 0,
                "with_triples": 0,
            },
        }

    emit.status_public_info(f"Found {len(paper_dirs)} paper directories")

    combined_papers = []
    stats = {
        "total": len(paper_dirs),
        "not_enriched": 0,  # Papers not processed in step 5
        "no_triples": 0,  # Papers with empty triples
        "with_triples": 0,  # Papers with triples (saved)
    }

    for paper_dir in paper_dirs:
        # Extract index - handle both paper_XXXXX and paper_idxXXXXX formats
        paper_name = paper_dir.name
        paper_index = int(paper_name.replace("paper_idx", "").replace("paper_", ""))

        # Load clean paper
        clean_paper = load_paper_data(paper_dir)
        if clean_paper is None:
            emit.status_public_warning(f"Skipping {paper_dir.name}: no paper.json")
            continue

        # Load enriched triples from step 5
        # Use same folder name as clean papers (paper_XXXXX format)
        triples_dir = enriched_triples_dir / paper_name
        triples_data = load_enriched_triples(triples_dir)

        # Paper not enriched in step 5
        if triples_data is None:
            stats["not_enriched"] += 1
            continue

        # Check if triples were extracted
        triples = triples_data.get("triples", [])
        if not triples:
            stats["no_triples"] += 1
            continue

        # Has triples = success
        stats["with_triples"] += 1
        combined_papers.append(
            {"index": paper_index, "paper": clean_paper, "triples": triples_data}
        )

    return {"combined_papers": combined_papers, "stats": stats}


def main(run_id: str, base_dir: Path):
    """
    Main entry point.

    Args:
        run_id: Run ID for pipeline orchestration mode.
        base_dir: Base directory for kg runs (passed by parent _1_seed_hypo).
    """
    from aii_pipeline.steps._1_seed_hypo.invention_kg.constants import (
        STEP_3_PAPERS_CLEAN,
        STEP_5_WIKIDATA,
        STEP_6_PAPER_TRIPLES,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.utils import get_run_dir

    emit.status_private_info(f"Run ID: {run_id}")

    clean_papers_dir = get_run_dir(STEP_3_PAPERS_CLEAN, run_id, base_dir)
    enriched_triples_dir = get_run_dir(STEP_5_WIKIDATA, run_id, base_dir)
    output_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id, base_dir)

    emit.status_private_info(f"Clean papers: {clean_papers_dir.relative_to(base_dir)}")
    emit.status_private_info(f"Enriched triples: {enriched_triples_dir.relative_to(base_dir)}")
    emit.status_private_info(f"Output: {output_dir.relative_to(base_dir)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all papers
    result = process_papers(clean_papers_dir, enriched_triples_dir)
    combined_papers = result["combined_papers"]
    stats = result["stats"]

    # Save combined data
    output_file = output_dir / "paper_triples_pr.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_papers, f, indent=2, ensure_ascii=False)

        emit.status_public_success(f"Saved {len(combined_papers)} papers to {output_file.name}")
    except Exception as e:
        emit.status_public_error(f"Failed to save {output_file}: {e}")
        return 1

    # Print summary
    emit.status_public_info("Summary")
    emit.status_public_info(f"Total papers: {stats['total']}")
    emit.status_public_info(f"  Not enriched (step 5): {stats['not_enriched']}")
    enriched = stats["total"] - stats["not_enriched"]
    emit.status_public_info(f"  Enriched: {enriched}")
    emit.status_public_info(f" With triples (saved): {stats['with_triples']}")
    emit.status_public_info(f" No triples (excluded): {stats['no_triples']}")

    if stats["total"] > 0 and enriched > 0:
        enrich_rate = (enriched / stats["total"]) * 100
        extraction_rate = (stats["with_triples"] / enriched) * 100
        emit.status_public_info(f"Enrichment rate: {enrich_rate:.1f}%")
        emit.status_public_info(f"Triple extraction rate: {extraction_rate:.1f}%")

    emit.status_public_success("Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
