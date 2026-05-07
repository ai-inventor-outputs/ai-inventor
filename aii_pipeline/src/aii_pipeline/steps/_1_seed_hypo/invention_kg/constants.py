#!/usr/bin/env python3
"""
Constants for Invention Knowledge Graph Pipeline.

This module defines all magic strings, directory names, and default values
used throughout the pipeline to ensure consistency and maintainability.

Note: "Triples" refers to the extracted (paper, relation, concept) tuples
from research papers. Relations include: uses (existing work), proposes (novel contributions).
"""

from pathlib import Path

# ============================================================================
# Base Directory
# ============================================================================
# kg outputs nest under: <base_dir>/<run_id>/1_seed_hypo/<step_name>/
# base_dir is passed explicitly by the parent _1_seed_hypo orchestrator on every
# step call (no singleton config). kg is not meant to run standalone.
SEED_HYPO_SUBDIR = "1_seed_hypo"  # Subfolder within each run for this module

# Local base dir for module-level files (prompts, configs)
BASE_DIR = Path(__file__).parent.resolve()  # invention_kg/

# ============================================================================
# Step Directory Names
# ============================================================================
# Directory names for each pipeline step's output
STEP_1_SEL_TOPICS = "_1_sel_topics"  # Selected topics from OpenAlex
STEP_2_PAPERS = "_2_papers"  # Raw papers from OpenAlex
STEP_3_PAPERS_CLEAN = "_3_papers_clean"  # Cleaned paper data
STEP_4_TRIPLES = "_4_triples"  # Triple extraction runs
STEP_5_WIKIDATA = "_5_wikidata"  # Triples enriched with Wikidata
STEP_6_PAPER_TRIPLES = "_6_paper_triples"  # Combined papers + enriched triples
STEP_7_HYPO_SEEDS = "_7_hypo_seeds"  # Hypothesis seeds (blind spots, breakthroughs)
STEP_8_SEED_PROMPT = "_8_seed_prompt"  # Generated seed prompts from blind spots
STEP_9_GRAPHS = "_9_graphs"  # All graph types
