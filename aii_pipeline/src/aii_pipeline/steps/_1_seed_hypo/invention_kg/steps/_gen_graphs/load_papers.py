#!/usr/bin/env python3
"""Paper loading functions for graph generation."""

import json
from pathlib import Path

from aii_lib.run import emit


def load_all_papers(combined_dir: Path) -> list[dict]:
    """Load all papers from combined JSON file."""
    json_file = combined_dir / "paper_triples_pr.json"

    if not json_file.exists():
        emit.status_public_error(f"Combined papers file not found: {json_file}")
        return []

    try:
        with open(json_file, encoding="utf-8") as f:
            papers = json.load(f)
        emit.status_public_info(f"Loaded {len(papers)} papers")
        return papers
    except Exception as e:
        emit.status_public_error(f"Failed to load {json_file}: {e}")
        return []


def load_papers_by_year(combined_dir: Path) -> dict[int, list[dict]]:
    """Load papers grouped by year."""
    all_papers = load_all_papers(combined_dir)
    if not all_papers:
        return {}

    papers_by_year = {}
    for paper_entry in all_papers:
        paper_data = paper_entry.get("paper", {})
        year = paper_data.get("publication_year")

        if year is not None:
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper_entry)

    emit.status_public_info(f"Loaded papers from {len(papers_by_year)} years")
    return papers_by_year
