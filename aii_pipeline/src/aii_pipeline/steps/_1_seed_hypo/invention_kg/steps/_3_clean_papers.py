#!/usr/bin/env python3
"""
Step 3: Clean Papers.

Extract minimal data needed for agent prompts from raw OpenAlex papers.

Input: data/_2_papers/{run_id}/topic_{id}/papers_{year}.json (raw OpenAlex data)
Output: data/_3_papers_clean/{run_id}/paper_XXXXX/paper.json (individual paper files)

Minimal format includes:
- index: Global paper index
- id: OpenAlex ID
- doi: Paper DOI
- title: Paper title
- abstract: Reconstructed from inverted index
- publication_year: Year published
- topic_id: OpenAlex topic ID the paper belongs to
- topic_name: Human readable topic name
"""

import json
from pathlib import Path
from typing import Any

from aii_lib.run import emit


def reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """
    Reconstruct abstract text from OpenAlex inverted index.

    Args:
        inverted_index: Dict mapping words to position indices

    Returns:
        Reconstructed abstract text
    """
    if not inverted_index:
        return ""

    # Create list of (position, word) tuples
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    # Sort by position and join
    word_positions.sort(key=lambda x: x[0])
    abstract = " ".join(word for _, word in word_positions)

    return abstract


def extract_minimal_paper(
    paper: dict[str, Any], global_index: int, topic_id: str, topic_name: str
) -> dict[str, Any]:
    """
    Extract fields needed for agent analysis and graph construction.

    Args:
        paper: Full OpenAlex paper object
        global_index: Global paper index across all years/topics
        topic_id: OpenAlex topic ID (e.g., "T10456")
        topic_name: Human readable topic name

    Returns:
        Paper dict with metadata, title, abstract, and author information
    """
    # Reconstruct abstract
    abstract = ""
    if paper.get("abstract_inverted_index"):
        abstract = reconstruct_abstract(paper["abstract_inverted_index"])

    return {
        "index": global_index,
        "id": paper.get("id", ""),
        "doi": paper.get("doi", ""),
        "title": paper.get("title", ""),
        "abstract": abstract,
        "publication_year": paper.get("publication_year"),
        "publication_date": paper.get("publication_date", ""),
        "cited_by_count": paper.get("cited_by_count", 0),
        "type": paper.get("type", ""),
        "language": paper.get("language", ""),
        "topic_id": topic_id,
        "topic_name": topic_name,
        "authorships": paper.get("authorships", []),
    }


def run_clean_papers(
    papers_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Clean all papers from the topic-based directory structure.

    Args:
        papers_dir: Path to papers directory (e.g., data/_2_papers/{run_id}/)
        output_dir: Directory to save individual paper folders

    Returns:
        Summary dict with counts
    """
    emit.status_public_progress("=== Step 3: Clean Papers ===")
    emit.status_private_info(f"Input: {papers_dir}")
    emit.status_private_info(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load topic summaries to get topic names
    topic_names = {}
    for topic_dir in papers_dir.glob("topic_*"):
        summary_file = topic_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, encoding="utf-8") as f:
                summary = json.load(f)
                topic_id = summary.get("topic_id", topic_dir.name.replace("topic_", ""))
                topic_names[topic_id] = summary.get("topic_name", "")

    # Process papers from all topic directories
    global_index = 0
    topic_stats = {}
    all_papers = []

    for topic_dir in sorted(papers_dir.glob("topic_*")):
        topic_id = topic_dir.name.replace("topic_", "")
        topic_name = topic_names.get(topic_id, topic_id)
        topic_papers = 0

        emit.status_public_info(f"Processing topic: {topic_name} ({topic_id})")

        # Process each year file in the topic
        for year_file in sorted(topic_dir.glob("papers_*.json")):
            year = year_file.stem.replace("papers_", "")

            with open(year_file, encoding="utf-8") as f:
                papers = json.load(f)

            for paper in papers:
                clean_paper = extract_minimal_paper(paper, global_index, topic_id, topic_name)

                # Create paper directory
                paper_dir = output_dir / f"paper_{global_index:05d}"
                paper_dir.mkdir(parents=True, exist_ok=True)

                # Save paper.json
                paper_file = paper_dir / "paper.json"
                with open(paper_file, "w", encoding="utf-8") as f:
                    json.dump(clean_paper, f, indent=2, ensure_ascii=False)

                all_papers.append(
                    {
                        "index": global_index,
                        "topic_id": topic_id,
                        "year": year,
                        "title": clean_paper["title"][:60],
                    }
                )

                global_index += 1
                topic_papers += 1

            emit.status_public_info(f"  {year}: {len(papers)} papers")

        topic_stats[topic_id] = {
            "name": topic_name,
            "papers": topic_papers,
        }
        emit.status_public_success(f"  Total: {topic_papers} papers for {topic_name}")

    # Save summary
    result = {
        "total_papers": global_index,
        "topic_stats": topic_stats,
        "papers": all_papers,
    }

    summary_file = output_dir / "clean_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Final summary
    emit.status_public_success("=== Clean Papers Complete ===")
    emit.status_public_info(f"Total papers cleaned: {global_index}")
    for stats in topic_stats.values():
        emit.status_public_info(f"  {stats['name']}: {stats['papers']} papers")
    emit.status_private_info(f"Output: {output_dir}")

    return result


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
