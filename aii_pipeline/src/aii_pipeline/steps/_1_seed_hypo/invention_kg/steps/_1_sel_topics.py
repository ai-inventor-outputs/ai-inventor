#!/usr/bin/env python3
"""
Step 1: Select Topics.

Resolves topic names from config to OpenAlex topic IDs.
This step validates that the configured topics exist in OpenAlex
and retrieves their full metadata for use in subsequent steps.

Input: config.yaml sel_topics.topics (list of topic names)
Output: data/_1_sel_topics/{run_id}/topics.json (resolved topic metadata)
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from aii_lib.run import emit
from aii_lib.utils.retry import make_retry_log
from pyalex import Topics
from pyalex import config as pyalex_config
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=make_retry_log(label="OpenAlex topic"),
    reraise=True,
)
def resolve_topic(topic_name: str) -> dict | None:
    """
    Resolve a topic name to its OpenAlex metadata.

    Args:
        topic_name: Display name of the topic (e.g., "Multi-Agent Systems and Negotiation")

    Returns:
        Topic metadata dict if found, None otherwise
    """
    # Search for the topic
    results = list(Topics().search(topic_name).get())

    if not results:
        emit.status_public_warning(f"No results for topic: {topic_name}")
        return None

    # Find exact match or best match
    for topic in results:
        if topic["display_name"].lower() == topic_name.lower():
            emit.status_public_info(f"Found exact match: {topic['display_name']}")
            return topic

    # Return first result as best match
    best_match = results[0]
    emit.status_public_warning(
        f"No exact match for '{topic_name}', using best match: '{best_match['display_name']}'"
    )
    return best_match


def resolve_topics(
    topic_names: list[str], email: str = os.environ.get("RESEARCHER_EMAIL", "")
) -> list[dict]:
    """
    Resolve a list of topic names to OpenAlex topic metadata.

    Args:
        topic_names: List of topic display names
        email: Email for OpenAlex polite pool access

    Returns:
        List of resolved topic metadata dicts
    """
    # Configure pyalex
    pyalex_config.email = email

    emit.status_public_info(f"Resolving {len(topic_names)} topics from OpenAlex")

    resolved = []
    failed = []

    for name in topic_names:
        topic = resolve_topic(name)
        if topic:
            resolved.append(
                {
                    "display_name": topic["display_name"],
                    "id": topic["id"],
                    "openalex_id": topic["id"].split("/")[-1],  # e.g., "T12345"
                    "description": topic.get("description", ""),
                    "keywords": topic.get("keywords", []),
                    "works_count": topic.get("works_count", 0),
                    "domain": topic.get("domain", {}),
                    "field": topic.get("field", {}),
                    "subfield": topic.get("subfield", {}),
                }
            )
        else:
            failed.append(name)

    if failed:
        emit.status_public_error(f"Failed to resolve {len(failed)} topics: {failed}")

    emit.status_public_success(f"Resolved {len(resolved)}/{len(topic_names)} topics")
    return resolved


def run_sel_topics(
    topic_names: list[str],
    output_dir: Path,
    email: str = os.environ.get("RESEARCHER_EMAIL", ""),
) -> dict:
    """
    Run the topic selection step.

    Args:
        topic_names: List of topic names from config
        output_dir: Directory to save results
        email: Email for OpenAlex API

    Returns:
        Dict with resolved topics and metadata
    """
    emit.status_public_progress("=== Step 1: Select Topics ===")
    emit.status_public_info(f"Topics to resolve: {len(topic_names)}")
    for name in topic_names:
        emit.status_public_info(f"  - {name}")

    # Resolve topics
    resolved_topics = resolve_topics(topic_names, email=email)

    if not resolved_topics:
        raise ValueError("No topics could be resolved!")

    # Create output
    result = {
        "resolved_at": datetime.now(UTC).isoformat(),
        "requested_topics": topic_names,
        "resolved_count": len(resolved_topics),
        "topics": resolved_topics,
    }

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "topics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    emit.status_public_success(f"Saved resolved topics to {output_file}")

    # Log summary
    emit.status_public_info("Resolved Topics:")
    for topic in resolved_topics:
        emit.status_private_info(f"  Topic:    {topic['display_name']}")
        emit.status_private_info(f"  Domain:   {topic['domain'].get('display_name', '?')}")
        emit.status_private_info(f"  Field:    {topic['field'].get('display_name', '?')}")
        emit.status_private_info(f"  Subfield: {topic['subfield'].get('display_name', '?')}")
        emit.status_private_info(f"  Works:    {topic['works_count']:,}")

    return result


if __name__ == "__main__":
    raise SystemExit(
        "invention_kg steps are no longer runnable standalone — "
        "run via `aii_pipeline` so the main pipeline can supply base_dir, "
        "run_id, and per-user output routing."
    )
