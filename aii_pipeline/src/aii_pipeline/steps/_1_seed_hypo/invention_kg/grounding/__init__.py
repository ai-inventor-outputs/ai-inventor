#!/usr/bin/env python3
"""
Grounding module for enriching triples with Wikidata.

Provides functions to:
- Fetch full Wikidata entity data from Wikipedia URLs
- Enrich triples with ALL Wikidata properties (stored under "wikidata" key)
"""

from .wikidata import (
    enrich_triples_with_wikidata,
    get_wikidata_from_wikipedia_url,
)

__all__ = [
    "enrich_triples_with_wikidata",
    "get_wikidata_from_wikipedia_url",
]
