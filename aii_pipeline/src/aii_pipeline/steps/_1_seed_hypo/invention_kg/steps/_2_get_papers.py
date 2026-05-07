#!/usr/bin/env python3
"""
Step 2: Get Papers.

Fetch papers from OpenAlex for each selected topic.

Input: data/_1_sel_topics/{run_id}/topics.json (resolved topics from step 1)
Output: data/_2_papers/{run_id}/topic_{id}/papers_{year}.json

Loop structure: for each topic -> for each year -> fetch papers
"""

import json
import os
from pathlib import Path

from aii_lib.run import emit
from aii_lib.utils.retry import make_retry_log
from pyalex import Works
from pyalex import config as pyalex_config
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=make_retry_log(label="Semantic Scholar"),
    reraise=True,
)
def fetch_papers_for_topic_year(
    topic_id: str, year: int, limit: int = 100, sort_by: str = "cited_by_count"
) -> tuple[list[dict], int]:
    """
    Fetch papers for a specific topic and year from OpenAlex.

    Args:
        topic_id: OpenAlex topic ID (e.g., "T10456")
        year: Publication year
        limit: Maximum number of papers to fetch
        sort_by: Sort criterion (default: "cited_by_count")

    Returns:
        ``(papers, filtered_count)`` where ``filtered_count`` is the
        number of papers dropped for missing abstracts.
    """
    try:
        # Query OpenAlex for papers with this topic
        query = (
            Works()
            .filter(publication_year=year)
            .filter(topics={"id": topic_id})
            .filter(has_abstract=True)
            .sort(**{sort_by: "desc"})
            .select(
                [
                    # Core identification
                    "id",
                    "doi",
                    "title",
                    "publication_year",
                    "publication_date",
                    "cited_by_count",
                    "type",
                    "language",
                    # Abstract (inverted index format)
                    "abstract_inverted_index",
                    # Topics & concepts
                    "topics",
                    "primary_topic",
                    "keywords",
                    "concepts",
                    # Graph building
                    "referenced_works",
                    "related_works",
                    # Open access
                    "best_oa_location",
                    "open_access",
                    # Authorship
                    "authorships",
                    # Additional metadata
                    "biblio",
                ]
            )
        )

        # Paginate to get papers up to limit
        all_papers = []
        per_page = min(200, limit)

        pager = query.paginate(per_page=per_page, n_max=limit)

        for page in pager:
            all_papers.extend(page)
            if len(all_papers) >= limit:
                all_papers = all_papers[:limit]
                break

        # Filter out papers with null abstracts
        original_count = len(all_papers)
        all_papers = [p for p in all_papers if p.get("abstract_inverted_index")]
        filtered_count = original_count - len(all_papers)

        return all_papers, filtered_count

    except Exception as e:
        emit.status_public_error(f"Error fetching papers for topic {topic_id}, year {year}: {e}")
        raise


def fetch_papers_for_topic(
    topic: dict,
    start_year: int,
    end_year: int,
    papers_per_year: int,
    output_dir: Path,
    sort_by: str = "cited_by_count",
) -> dict:
    """
    Fetch papers for a single topic across all years.

    Args:
        topic: Topic dict with id, display_name, openalex_id
        start_year: First year to fetch
        end_year: Last year to fetch (inclusive)
        papers_per_year: Papers to fetch per year
        output_dir: Directory to save results (topic-specific)
        sort_by: Sort criterion

    Returns:
        Summary dict with counts
    """
    topic_name = topic["display_name"]
    topic_id = topic["openalex_id"]

    emit.status_public_info(f"=== Topic: {topic_name} ({topic_id}) ===")

    # Create topic output directory
    topic_dir = output_dir / f"topic_{topic_id}"
    topic_dir.mkdir(parents=True, exist_ok=True)

    total_papers = 0
    years_data = {}

    for year in range(start_year, end_year + 1):
        output_file = topic_dir / f"papers_{year}.json"

        # Skip if already exists
        if output_file.exists():
            emit.status_public_info(f"  {year}: already exists, skipping")
            with open(output_file, encoding="utf-8") as f:
                papers = json.load(f)
            years_data[year] = len(papers)
            total_papers += len(papers)
            continue

        # Fetch papers
        try:
            papers, filtered = fetch_papers_for_topic_year(
                topic_id=topic_id, year=year, limit=papers_per_year, sort_by=sort_by
            )

            if papers:
                # Save to file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)

                years_data[year] = len(papers)
                total_papers += len(papers)
                filter_note = f" (filtered {filtered})" if filtered else ""
                emit.status_public_info(f"  {year}: {len(papers)} papers{filter_note}")
            else:
                years_data[year] = 0
                emit.status_public_warning(f"  {year}: no papers found")

        except Exception as e:
            emit.status_public_error(f"  {year}: failed - {e}")
            years_data[year] = 0

    # Save topic summary
    summary = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "years": years_data,
        "total_papers": total_papers,
    }
    summary_file = topic_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    emit.status_public_success(f"  Total: {total_papers} papers for {topic_name}")
    return summary


def run_get_papers(
    topics_file: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    papers_per_topic_per_year: int,
    sort_by: str = "cited_by_count",
    email: str = os.environ.get("RESEARCHER_EMAIL", ""),
) -> dict:
    """
    Run the get papers step for all topics.

    Args:
        topics_file: Path to resolved topics JSON from step 1
        output_dir: Base output directory
        start_year: First year
        end_year: Last year (inclusive)
        papers_per_topic_per_year: Papers to fetch per topic per year
        sort_by: Sort criterion
        email: Email for OpenAlex API

    Returns:
        Summary dict
    """
    # Configure pyalex
    pyalex_config.email = email

    emit.status_public_progress("=== Step 2: Get Papers ===")

    # Load topics from step 1
    with open(topics_file, encoding="utf-8") as f:
        topics_data = json.load(f)

    topics = topics_data["topics"]
    emit.status_public_info(f"Topics: {len(topics)}")
    emit.status_public_info(f"Years: {start_year}-{end_year} ({end_year - start_year + 1} years)")
    emit.status_public_info(f"Papers per topic per year: {papers_per_topic_per_year}")

    expected_total = len(topics) * (end_year - start_year + 1) * papers_per_topic_per_year
    emit.status_public_info(f"Expected max papers: ~{expected_total:,}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch papers for each topic
    all_summaries = []
    grand_total = 0

    for topic in topics:
        summary = fetch_papers_for_topic(
            topic=topic,
            start_year=start_year,
            end_year=end_year,
            papers_per_year=papers_per_topic_per_year,
            output_dir=output_dir,
            sort_by=sort_by,
        )
        all_summaries.append(summary)
        grand_total += summary["total_papers"]

    # Save overall summary
    result = {
        "topics_file": str(topics_file),
        "start_year": start_year,
        "end_year": end_year,
        "papers_per_topic_per_year": papers_per_topic_per_year,
        "topic_summaries": all_summaries,
        "grand_total_papers": grand_total,
    }

    result_file = output_dir / "fetch_summary.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Final summary
    emit.status_public_success("=== Fetch Complete ===")
    emit.status_public_info(f"Total papers: {grand_total:,}")
    for s in all_summaries:
        emit.status_public_info(f"  {s['topic_name']}: {s['total_papers']:,}")

    return result


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
