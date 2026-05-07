"""Deploy helpers — paper copying, exclusion rules for file copying."""

from __future__ import annotations

import shutil
from pathlib import Path

from aii_lib.run import emit

# Inlined from the (now-deleted) aii_lib.agent_backend.utils.finalizer module.
# Files inside these dirs never get pushed to GitHub — env junk, build caches,
# and Claude session state. Update here if a new ignored category appears.
EXCLUDED_DIRS = {
    ".venv",
    "venv",
    ".skill_venv",
    ".venv_img",
    ".nb_env",
    ".ability_client_venv",
    "__pycache__",
    ".git",
    ".cache",
    "node_modules",
    "temp",
    "tmp",
    "dependencies",
    ".claude",
}
EXCLUDED_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    ".credentials.json",
}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".log", ".pt", ".pth", ".ckpt", ".safetensors"}


def make_copytree_ignore(max_file_size_bytes: int):
    """Create an ignore function for shutil.copytree."""

    def _ignore(directory, contents):
        ignored = set()
        for name in contents:
            full = Path(directory) / name
            if (
                name in EXCLUDED_DIRS
                or name in EXCLUDED_FILES
                or (full.is_file() and full.suffix in EXCLUDED_EXTENSIONS)
                or (full.is_file() and full.stat().st_size > max_file_size_bytes)
            ):
                ignored.add(name)
        return ignored

    return _ignore


def copy_paper_to_repo(
    repo_dir: Path,
    paper_pdf_path: Path | None,
    paper_latex_dir: Path | None,
    files_added: list[str],
    max_file_size_bytes: int = 50 * 1024 * 1024,
) -> tuple[bool, bool]:
    """Copy paper PDF and LaTeX source to repo. Returns (has_pdf, has_latex)."""
    has_pdf = False
    has_latex = False

    emit.status_private_info(
        f"paper_pdf_path: {paper_pdf_path} (exists={paper_pdf_path.exists() if paper_pdf_path else 'N/A'})"
    )

    if paper_pdf_path and paper_pdf_path.exists():
        dst = repo_dir / "paper.pdf"
        shutil.copy2(paper_pdf_path, dst)
        files_added.append("paper.pdf")
        has_pdf = True
        emit.status_public_success(
            f"   Copied paper.pdf ({paper_pdf_path.stat().st_size / 1024:.0f}KB)"
        )

    if paper_latex_dir and paper_latex_dir.exists() and paper_latex_dir.is_dir():
        dst = repo_dir / "paper_latex"
        shutil.copytree(
            paper_latex_dir,
            dst,
            dirs_exist_ok=True,
            ignore=make_copytree_ignore(max_file_size_bytes),
        )
        latex_files = list(dst.rglob("*"))
        for f in latex_files:
            if f.is_file():
                files_added.append(f"paper_latex/{f.relative_to(dst)}")
        has_latex = True
        emit.status_public_success(f"   Copied paper_latex/ ({len(latex_files)} files)")

    return has_pdf, has_latex
