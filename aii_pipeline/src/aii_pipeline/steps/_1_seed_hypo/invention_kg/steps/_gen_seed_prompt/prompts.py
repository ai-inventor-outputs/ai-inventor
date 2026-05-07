#!/usr/bin/env python3
"""
Prompt formatters for hypothesis generation.

Uses templates from prompts/steps/_1_seed_hypo/_invention_kg/gen_hypo_seeds/hypo_templates.py.
Updated for concept-centric blind spot structure.
"""

from typing import Any

from aii_pipeline.prompts.steps._1_seed_hypo._invention_kg.gen_hypo_seeds.hypo_templates import (
    HYPO_TEMPLATES,
)


def _load_templates() -> dict[str, str]:
    """Return templates dict."""
    return HYPO_TEMPLATES


def format_blind_spot_prompt(opportunity: dict[str, Any]) -> str:
    """
    Format a concept-centric blind spot opportunity into a prompt.

    Args:
        opportunity: Blind spot opportunity JSON (concept-centric format)

    Returns:
        Formatted prompt string
    """
    templates = _load_templates()

    # Extract data from concept-centric format (hierarchical structure)
    concept = opportunity.get("concept", "Unknown concept")
    blind_topic = opportunity.get("blind_topic", "Unknown topic")
    ref_topic = opportunity.get("ref_topic", "Unknown topic")
    entity_type = opportunity.get("entity_type", "concept")

    # Get shared concepts from topic_pair sub-object
    topic_pair = opportunity.get("topic_pair", {})
    shared = topic_pair.get("shared_concepts", [])

    # Format shared concepts
    shared_str = ", ".join(shared[:5]) if shared else "various concepts"
    if len(shared) > 5:
        shared_str += "..."

    # Single concept as blind spot list
    blind_spot_list = f"- {concept} ({entity_type})"

    return templates["blind_spot"].format(
        blind_topic=blind_topic,
        ref_topic=ref_topic,
        shared_concepts=shared_str,
        blind_spot_list=blind_spot_list,
    )


def format_opportunity_prompt(opportunity: dict[str, Any]) -> str:
    """
    Format opportunity into a prompt string.

    Args:
        opportunity: Opportunity JSON (concept-centric blind spot)

    Returns:
        Formatted prompt string
    """
    # Concept-centric format - check for concept field
    if "concept" in opportunity:
        return format_blind_spot_prompt(opportunity)

    raise ValueError(f"Unknown opportunity type: {opportunity.get('opportunity_type', '')}")
