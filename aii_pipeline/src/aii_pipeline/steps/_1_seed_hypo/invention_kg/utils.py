#!/usr/bin/env python3
"""
Utility functions for Invention Knowledge Graph pipeline.

This module provides common utilities used across all pipeline steps,
including run ID management and path resolution helpers.

base_dir is always passed explicitly by the parent _1_seed_hypo orchestrator
(no singleton config). kg outputs nest under
<base_dir>/<run_id>/1_seed_hypo/<step_name>/.
"""

from pathlib import Path

# Get base directory of the invention_kg module (for module-local data files)
BASE_DIR = Path(__file__).parent.resolve()


# ============================================================================
# Run ID Management
# ============================================================================


def get_run_dir(
    step_name: str,
    run_id: str,
    base_dir: Path,
) -> Path:
    """
    Get the output directory path for a specific step and run.

    Structure: <base_dir>/<run_id>/1_seed_hypo/<step_name>/

    Args:
        step_name: Step directory name (e.g., '_1_sel_topics', '_7_hypo_seeds')
        run_id: Run ID string (e.g., 'novak_hypo_seed')
        base_dir: Base directory for runs (required).

    Returns:
        Path to step's run directory.
    """
    from .constants import SEED_HYPO_SUBDIR

    return base_dir / run_id / SEED_HYPO_SUBDIR / step_name
