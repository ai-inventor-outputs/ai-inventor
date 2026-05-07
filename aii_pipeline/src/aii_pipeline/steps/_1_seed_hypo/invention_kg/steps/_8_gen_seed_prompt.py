#!/usr/bin/env python3
"""
Step 8: Generate Seed Prompts.

Takes hypothesis seeds from step 7 and formats them into prompt snippets.
These snippets describe opportunities that can be inserted into a larger
LLM prompt elsewhere.

Input: data/_7_hypo_seeds/{run_id}/
    - topic_blind_spots.json

Output: data/_8_seed_prompt/{run_id}/
    - blind_spot_prompts.json
"""

import json
import shutil
from pathlib import Path
from typing import Any

from aii_lib.run import emit

from ._gen_seed_prompt import format_opportunity_prompt

__all__ = ["main"]


def extract_score_info(opp: dict[str, Any]) -> dict[str, Any]:
    """Extract score and breakdown from concept-centric blind spot opportunity."""
    # Get hierarchical scores from sub-objects
    topic_pair = opp.get("topic_pair", {})
    importance = opp.get("importance", {})
    transferability = opp.get("transferability", {})
    novelty = opp.get("novelty", {})

    return {
        "score": opp.get("seed_score", 0),
        "score_breakdown": {
            "topic_pair": topic_pair.get("score", 0),
            "importance": importance.get("score", 0),
            "transferability": transferability.get("score", 0),
            "novelty": novelty.get("score", 0),
        },
        "percentile_breakdown": {
            "topic_pair": topic_pair.get("percentile", 0),
            "importance": importance.get("percentile", 0),
            "transferability": transferability.get("percentile", 0),
            "novelty": novelty.get("percentile", 0),
        },
    }


def extract_topics(opp: dict[str, Any]) -> list[str]:
    """Extract topic names from concept-centric blind spot opportunity."""
    topics = []

    # In concept-centric format, blind_topic and ref_topic are strings
    blind_topic = opp.get("blind_topic")
    if blind_topic:
        if isinstance(blind_topic, dict):
            topics.append(blind_topic.get("name", ""))
        else:
            topics.append(str(blind_topic))

    ref_topic = opp.get("ref_topic")
    if ref_topic:
        if isinstance(ref_topic, dict):
            topics.append(ref_topic.get("name", ""))
        else:
            topics.append(str(ref_topic))

    return [t for t in topics if t]


def format_opportunities(opportunities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format blind spot opportunities into prompt snippets with scores."""
    emit.status_public_info(f"Formatting {len(opportunities)} blind spot opportunities")

    results = []
    for i, opp in enumerate(opportunities):
        try:
            prompt_text = format_opportunity_prompt(opp)
            score_info = extract_score_info(opp)
            topics = extract_topics(opp)

            # Use existing id from concept-centric format, or generate one
            opp_id = opp.get("id", f"blind_spot_idx{i}")

            # Extract blind_topic and ref_topic as separate fields
            blind_topic = opp.get("blind_topic", "")
            if isinstance(blind_topic, dict):
                blind_topic = blind_topic.get("name", "")
            ref_topic = opp.get("ref_topic", "")
            if isinstance(ref_topic, dict):
                ref_topic = ref_topic.get("name", "")

            results.append(
                {
                    "id": opp_id,
                    "type": "blind_spot",
                    "concept": opp.get("concept", ""),
                    "entity_type": opp.get("entity_type", ""),
                    "blind_topic": blind_topic,
                    "ref_topic": ref_topic,
                    "topics": topics,
                    "score": score_info["score"],
                    "score_breakdown": score_info["score_breakdown"],
                    "score_percentile": opp.get("score_percentile", 0),
                    "percentile_breakdown": score_info["percentile_breakdown"],
                    "prompt": prompt_text,
                }
            )
        except Exception as e:
            emit.status_public_warning(f"Failed to format opportunity {i}: {e}")

    emit.status_public_info(f"Formatted {len(results)}/{len(opportunities)} prompts")
    return results


def main(run_id: str, base_dir: Path):
    """
    Main entry point for seed prompt formatting.

    Args:
        run_id: Run ID for pipeline orchestration.
        base_dir: Base directory for kg runs (passed by parent _1_seed_hypo).
    """
    from ..constants import (
        STEP_7_HYPO_SEEDS,
        STEP_8_SEED_PROMPT,
    )
    from ..utils import get_run_dir

    emit.status_private_info(f"Run ID: {run_id}")

    input_dir = get_run_dir(STEP_7_HYPO_SEEDS, run_id, base_dir)
    output_dir = get_run_dir(STEP_8_SEED_PROMPT, run_id, base_dir)

    emit.status_private_info(f"Input: {input_dir.relative_to(base_dir)}")
    emit.status_private_info(f"Output: {output_dir.relative_to(base_dir)}")

    # Check input exists before destroying output
    if not input_dir.exists():
        emit.status_public_error(f"Input directory not found: {input_dir}")
        emit.status_public_error("Run step 7 first")
        return 1

    # Clean up output directory
    if output_dir.exists():
        emit.status_private_info("Removing existing output directory")
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load opportunities
    blind_spots_file = input_dir / "topic_blind_spots.json"

    if not blind_spots_file.exists():
        emit.status_public_error("No topic_blind_spots.json found")
        return 1

    with open(blind_spots_file) as f:
        blind_spots = json.load(f)
    emit.status_public_info(f"Loaded {len(blind_spots)} blind spot opportunities")

    if not blind_spots:
        emit.status_public_error("No opportunities found")
        return 1

    # Format blind spot prompts
    emit.status_public_info("1. Formatting Blind Spot Opportunities")
    all_prompts = format_opportunities(blind_spots)

    if all_prompts:
        # Sort by score_percentile descending (already computed in step 7)
        all_prompts.sort(key=lambda x: x.get("score_percentile", 0), reverse=True)

        with open(output_dir / "blind_spot_prompts.json", "w") as f:
            json.dump(all_prompts, f, indent=2, ensure_ascii=False)

    # Summary
    emit.status_public_info("Summary")
    emit.status_public_info(f"Total prompts formatted: {len(all_prompts)}")

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
