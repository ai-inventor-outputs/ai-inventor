#!/usr/bin/env python3
"""
Resume functionality for triple extraction.

Provides functions to check which papers are already completed and skip them on restart.
"""

import shutil
from pathlib import Path

from aii_lib.run import emit

from .display import console


def get_completed_from_progress_file(parent_run_dir: Path) -> set[int]:
    """
    Read completed paper indices from progress file.

    This is the source of truth for what papers have been successfully completed.
    Much faster than scanning directories.

    Args:
        parent_run_dir: Output directory containing completed_papers.txt

    Returns:
        Set of paper indices that are successfully completed
    """
    progress_file = parent_run_dir / "completed_papers.txt"
    completed_indices = set()

    if not progress_file.exists():
        return completed_indices

    try:
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "paper_index" (one per line, only successful completions)
                try:
                    paper_index = int(line)
                    completed_indices.add(paper_index)
                except ValueError:
                    emit.status_public_warning(f"Invalid line in progress file: {line}")
    except Exception as e:
        emit.status_public_error(f"Error reading progress file {progress_file}: {e}")
        raise

    return completed_indices


def get_completed_paper_indices(
    parent_run_dir: Path,
) -> set[int]:
    """
    Get completed paper indices from progress file ONLY.

    The progress file (completed_papers.txt) is the single source of truth.
    Any paper folders not in the progress file are considered failed and will be deleted.

    This enables dynamic resume: we skip papers that are already done.

    Args:
        parent_run_dir: Output directory containing paper folders

    Returns:
        Set of paper indices that are successfully completed (from progress file only)
    """
    if not parent_run_dir.exists():
        return set()

    # Read from progress file (ONLY source of truth)
    completed_indices = get_completed_from_progress_file(parent_run_dir)

    # Check all existing paper folders
    paper_dirs = sorted(parent_run_dir.glob("paper_*"))
    if not paper_dirs:
        if completed_indices:
            console.print(
                f"\n[bold green]✓ Found {len(completed_indices)} completed papers from progress file[/bold green]"
            )
            console.print(
                "[yellow]⚠ No paper folders found - progress file may be stale[/yellow]\n"
            )
        return completed_indices

    # Extract indices from directories
    dir_indices = set()
    for paper_dir in paper_dirs:
        try:
            # Handle both paper_XXXXX and paper_idxXXXXX formats
            name = paper_dir.name
            paper_index = int(name.replace("paper_idx", "").replace("paper_", ""))
            dir_indices.add(paper_index)
        except ValueError:
            continue

    # Find directories that aren't in the progress file (failed/incomplete)
    failed_or_incomplete = dir_indices - completed_indices

    # If all directories are in progress file, we're done
    if not failed_or_incomplete:
        if completed_indices:
            console.print(
                f"\n[bold green]✓ Found {len(completed_indices)} completed papers from progress file[/bold green]"
            )
            console.print(f"[dim]All {len(dir_indices)} paper folders match progress file[/dim]\n")
        return completed_indices

    # Delete folders not in progress file (they failed or didn't complete)
    console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    console.print("[bold]Progress File Cleanup[/bold]")
    console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")

    if completed_indices:
        console.print(
            f"[bold green]✓ Found {len(completed_indices)} completed papers from progress file[/bold green]"
        )
    console.print(
        f"[bold yellow]⚠ Found {len(failed_or_incomplete)} paper folders NOT in progress file[/bold yellow]"
    )
    console.print("[dim]These folders will be deleted (failed or incomplete runs)[/dim]\n")

    deleted_count = 0
    for paper_dir in paper_dirs:
        try:
            # Handle both paper_XXXXX and paper_idxXXXXX formats
            name = paper_dir.name
            paper_index = int(name.replace("paper_idx", "").replace("paper_", ""))

            # Skip if in progress file (completed)
            if paper_index in completed_indices:
                continue

            # Delete folder - it's not in progress file so it failed
            console.print(f"  [red]✗[/red] Deleting Paper {paper_index:05d} (not in progress file)")
            emit.status_public_info(f"Paper {paper_index}: Not in progress file, deleting folder")

            try:
                shutil.rmtree(paper_dir)
                deleted_count += 1
                emit.status_private_debug(f"Deleted folder: {paper_dir}")
            except Exception as e:
                emit.status_public_error(f"Failed to delete {paper_dir}: {e}")

        except (ValueError, Exception) as e:
            emit.status_public_warning(f"Error checking {paper_dir.name}: {e}")
            continue

    # Print summary
    remaining_dirs = len(dir_indices) - deleted_count
    console.print(
        f"\n[bold green]Completed papers:[/bold green] {len(completed_indices)}/{len(dir_indices)}"
    )
    console.print(f"[bold red]Deleted folders:[/bold red] {deleted_count}")
    console.print(f"[bold cyan]Remaining folders:[/bold cyan] {remaining_dirs}")
    console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")

    emit.status_public_info(
        f"Progress file check complete: {len(completed_indices)} completed, {deleted_count} folders deleted"
    )

    return completed_indices
