"""Prompts and schemas for invention_kg knowledge graph extraction."""

# get_triples step exports
# gen_hypo_seeds step exports
from .gen_hypo_seeds import BLIND_SPOT_TEMPLATE, BREAKTHROUGH_TEMPLATE, HYPO_TEMPLATES
from .get_triples import (
    SYSTEM_PROMPT,
    Triple,
    Triples,
    build_retry_prompt,
    get_system_prompt,
    triples_prompt,
)

__all__ = [
    # get_triples
    "Triple",
    "Triples",
    "triples_prompt",
    "build_retry_prompt",
    "SYSTEM_PROMPT",
    "get_system_prompt",
    # gen_hypo_seeds
    "BLIND_SPOT_TEMPLATE",
    "BREAKTHROUGH_TEMPLATE",
    "HYPO_TEMPLATES",
]
