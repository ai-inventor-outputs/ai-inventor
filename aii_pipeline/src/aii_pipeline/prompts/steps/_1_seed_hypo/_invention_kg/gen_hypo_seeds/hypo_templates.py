"""Hypothesis prompt templates for gen_hypo_seeds step.

These templates are used to format opportunity seeds into prompt snippets.
Variables use Python string formatting: {variable_name}
"""

# Cross-field transfer opportunity (gap/blind spot)
BLIND_SPOT_TEMPLATE = """\
CROSS-FIELD TRANSFER OPPORTUNITY

Two research fields share common foundations but have developed different toolsets.
This represents an opportunity to transfer proven techniques from one field to another.

"{blind_topic}" and "{ref_topic}" both work with: {shared_concepts}

However, "{ref_topic}" has extensively adopted the following concepts that "{blind_topic}" has not yet utilized:

{blind_spot_list}

These concepts have proven valuable in {ref_topic}. Consider how they might address
unsolved challenges or enable new capabilities in {blind_topic}."""

# Breakthrough pattern opportunity
BREAKTHROUGH_TEMPLATE = """\
BREAKTHROUGH PATTERN

A high-impact research contribution emerged by combining existing concepts in a novel way.
Understanding this pattern may reveal similar opportunities in related domains.

The paper "{paper_title}" ({paper_year}, {paper_citations} citations) in {paper_topic}
achieved a breakthrough by proposing: {breakthrough_concept}

This was accomplished by:
- Building upon existing techniques: {uses}
- Extending prior work on: {extends}
- Introducing new concepts: {proposes}

Consider: What analogous combinations of established techniques could yield similar
breakthroughs in adjacent problem spaces or related domains?"""


HYPO_TEMPLATES = {
    "blind_spot": BLIND_SPOT_TEMPLATE,
    "breakthrough": BREAKTHROUGH_TEMPLATE,
}


__all__ = ["BLIND_SPOT_TEMPLATE", "BREAKTHROUGH_TEMPLATE", "HYPO_TEMPLATES"]
