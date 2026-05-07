#!/usr/bin/env python3
"""
Configuration loading for triple extraction.

Handles loading pipeline config, agent config, and paper data.
"""

import json
from pathlib import Path

import yaml
from aii_lib.run import emit


def load_pipeline_config(config_path: Path) -> dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Pipeline configuration dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_agent_config(config_path: Path) -> dict:
    """
    Load agent configuration from YAML file.

    Args:
        config_path: Path to agent config (e.g., bblocks_config.yaml)

    Returns:
        Agent configuration dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    emit.status_public_success(f"Loaded config from: {config_path}")
    return config


def load_papers_from_directory(
    papers_dir: Path, max_papers: int = -1
) -> tuple[list[dict], list[dict], int]:
    """
    Load all papers from paper_XXXXX/ directories.

    Args:
        papers_dir: Directory containing paper_XXXXX/ folders with paper.json
        max_papers: Maximum number of papers to load (-1 for all)

    Returns:
        Tuple of (all_papers, papers_to_process, num_paper_dirs) where:
        - all_papers: All papers loaded from all paper directories
        - papers_to_process: Papers after applying max_papers limit
        - num_paper_dirs: Number of paper directories found
    """
    # Find all paper_* directories and sort by index
    paper_dirs = sorted(papers_dir.glob("paper_*"))

    if not paper_dirs:
        raise FileNotFoundError(f"No paper directories found in {papers_dir}")

    emit.status_public_info(f"Found {len(paper_dirs)} paper directories")

    # Load all papers
    emit.status_private_info("Loading all papers...")
    all_papers = []
    for paper_dir in paper_dirs:
        paper_file = paper_dir / "paper.json"
        if not paper_file.exists():
            emit.status_public_warning(f"Skipping {paper_dir.name}: no paper.json found")
            continue

        try:
            with open(paper_file, encoding="utf-8") as f:
                paper = json.load(f)
                all_papers.append(paper)
        except Exception as e:
            emit.status_public_error(f"Error loading {paper_file}: {e}")
            continue

    emit.status_public_info(f"Loaded {len(all_papers)} papers from {len(paper_dirs)} directories")

    # Apply max_papers limit
    if max_papers == -1:
        papers_to_process = all_papers
    else:
        papers_to_process = all_papers[:max_papers]

    return all_papers, papers_to_process, len(paper_dirs)
