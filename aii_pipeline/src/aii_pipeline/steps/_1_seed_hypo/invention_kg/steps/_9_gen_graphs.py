#!/usr/bin/env python3
"""
Step 9: Generate Graphs.

Creates multiple graph representations:
1. Concepts graph: Co-occurrence of concepts
2. Concept Ontology: Wikidata hierarchy (subclass_of, part_of)
3. Paper to Concepts: Semantic KG with UMAP embeddings
4. Blind Spots: Topic gaps from hypo_seeds

Input: data/_6_paper_triples/{run_id}/, data/_7_hypo_seeds/{run_id}/
Output: data/_9_graphs/{run_id}/
"""

import shutil
from pathlib import Path

from aii_lib.run import emit

from ._gen_graphs import (
    generate_blind_spots_graph,
    generate_cooccurrence_graph,
    generate_ontology_graph,
    generate_semantic_graph,
    load_all_papers,
    load_papers_by_year,
)

__all__ = ["main"]


def main(run_id: str, base_dir: Path, temporal_windows: list[list[int]]):
    """
    Main entry point for graph generation.

    Args:
        run_id: Run ID for pipeline orchestration.
        base_dir: Base directory for kg runs (passed by parent _1_seed_hypo).
        temporal_windows: List of [start_year, end_year] pairs for graph slicing.
    """
    from ..constants import (
        STEP_6_PAPER_TRIPLES,
        STEP_7_HYPO_SEEDS,
        STEP_9_GRAPHS,
    )
    from ..utils import get_run_dir

    emit.status_private_info(f"Run ID: {run_id}")
    emit.status_private_info(f"Temporal windows: {temporal_windows}")

    input_dir = get_run_dir(STEP_6_PAPER_TRIPLES, run_id, base_dir)
    hypo_seeds_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id, base_dir)
    output_dir = get_run_dir(STEP_9_GRAPHS, run_id, base_dir)

    emit.status_private_info(f"Input (triples): {input_dir.relative_to(base_dir)}")
    emit.status_private_info(f"Input (hypo_seeds): {hypo_seeds_dir.relative_to(base_dir)}")
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
    papers = load_all_papers(input_dir)

    if not papers:
        emit.status_public_error("No papers loaded")
        return 1

    emit.status_public_info(f"Loaded {len(papers)} papers")

    # Generate all graphs
    results = {}

    # 1. Concepts graph (co-occurrence, all years + per-year)
    emit.status_public_info("1. Concepts graph (co-occurrence)")
    cooccur_dir = output_dir / "cooccurrence"
    cooccur_dir.mkdir(parents=True, exist_ok=True)

    results["concepts"] = generate_cooccurrence_graph(
        papers, cooccur_dir / "all.json", temporal_windows
    )

    # Per-year co-occurrence
    papers_by_year = load_papers_by_year(input_dir)
    by_year_dir = cooccur_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(papers_by_year.keys()):
        year_papers = papers_by_year[year]
        generate_cooccurrence_graph(year_papers, by_year_dir / f"{year}.json", temporal_windows)

    # 2. Concept Ontology graph
    emit.status_public_info("2. Concept Ontology graph")
    ontology_dir = output_dir / "ontology"
    results["ontology"] = generate_ontology_graph(papers, ontology_dir / "full.json")

    # 3. Paper to Concepts graph (semantic/UMAP)
    emit.status_public_info("3. Paper to Concepts graph (semantic)")
    semantic_dir = output_dir / "semantic"
    results["paper_concepts"] = generate_semantic_graph(papers, semantic_dir)

    # 4. Blind Spots graph (from hypo_seeds)
    emit.status_public_info("4. Blind Spots graph")
    derived_dir = output_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    if hypo_seeds_dir.exists():
        results["blind_spots"] = generate_blind_spots_graph(
            hypo_seeds_dir, derived_dir / "blind_spots.json"
        )
    else:
        emit.status_public_warning("Hypo seeds directory not found, skipping blind spots")
        results["blind_spots"] = False

    # Summary
    emit.status_public_info("Summary")
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    emit.status_public_info(f"Generated {success_count}/{total_count} graph types")

    # List output files
    emit.status_public_info("Output structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.rglob("*.json"))
            emit.status_public_info(f"  {subdir.name}/ ({len(files)} files)")

    emit.status_public_success("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
