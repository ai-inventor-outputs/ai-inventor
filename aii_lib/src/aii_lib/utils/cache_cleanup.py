"""
Cache cleanup utilities for run directories.

Cleans up temporary caches created during pipeline runs:
- venv directories (.venv, .skill_venv, .venv_img, .nb_env — agents pick names)
- .hf_cache directories (HuggingFace dataset downloads)
"""

import shutil
from pathlib import Path

# Every venv flavor an agent might create. Keep in sync with EXCLUDED_DIRS in
# aii_pipeline/.../_4_gen_paper_repo/utils/deploy.py — both lists guard against
# shipping/leaving these dirs around.
_VENV_DIR_NAMES = (
    ".venv",
    ".skill_venv",
    ".venv_img",
    ".nb_env",
    ".ability_client_venv",
)


def cleanup_run_caches(
    run_dir: Path | str,
    clear_venv: bool = True,
    clear_hf: bool = True,
) -> dict:
    """
    Clean up cache directories in a run folder.

    Args:
        run_dir: Path to the run directory (e.g., runs/20250112_123456/)
        clear_venv: Remove all venv-flavor directories (see _VENV_DIR_NAMES)
        clear_hf: Remove .hf_cache directories (HuggingFace downloads)

    Returns:
        Dict with cleanup stats: {removed: [...], total_size_mb: float}
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {"removed": [], "total_size_mb": 0.0}

    removed = []
    total_size_mb = 0.0

    if clear_venv:
        for name in _VENV_DIR_NAMES:
            for venv_dir in run_dir.rglob(name):
                if venv_dir.is_dir():
                    try:
                        size_mb = _get_dir_size_mb(venv_dir)
                        shutil.rmtree(venv_dir)
                        total_size_mb += size_mb
                        removed.append(
                            f"{name} ({size_mb:.1f} MB) at {venv_dir.relative_to(run_dir)}"
                        )
                    except Exception as e:
                        from loguru import logger as _logger

                        _logger.error(f"Failed to remove {name} at {venv_dir}: {e}")

    # Clean up .hf_cache directories
    if clear_hf:
        for hf_dir in run_dir.rglob(".hf_cache"):
            if hf_dir.is_dir():
                try:
                    size_mb = _get_dir_size_mb(hf_dir)
                    shutil.rmtree(hf_dir)
                    total_size_mb += size_mb
                    removed.append(f".hf_cache ({size_mb:.1f} MB) at {hf_dir.relative_to(run_dir)}")
                except Exception as e:
                    from loguru import logger as _logger

                    _logger.error(f"Failed to remove .hf_cache at {hf_dir}: {e}")

    return {"removed": removed, "total_size_mb": total_size_mb}


def _get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    except Exception:
        return 0.0
