#!/usr/bin/env python3
"""
Display utilities for triple extraction.

Provides formatted console output using Rich for progress tracking,
summaries, and status messages.
"""

import asyncio
from datetime import UTC, datetime

from rich.console import Console

# Global console for all output
console = Console()


def create_progress_tracker():
    """
    Create a simple logging-based progress tracker.

    Returns:
        An async function for displaying progress updates
    """
    print_lock = asyncio.Lock()

    async def update_task_display(description: str):
        """Log a paper's progress with timestamp."""
        async with print_lock:
            timestamp = datetime.now(UTC).strftime("%H:%M:%S")
            console.print(f"[dim]{timestamp}[/dim] {description}")

    return update_task_display


def print_summary(
    successful: int,
    failed: int,
    total_papers: int,
    total_time: float,
    title: str = "Processing Summary",
):
    """Print a summary of the processing results."""
    console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    console.print(
        f"[bold]{title}:[/bold]" if "Summary" in title else f"[bold magenta]{title}:[/bold magenta]"
    )
    console.print(f"  [green]Successful:[/green] {successful}/{total_papers}")
    console.print(f"  [red]Failed:[/red] {failed}/{total_papers}")
    console.print(f"  [blue]Total Time:[/blue] {total_time:.1f}s ({total_time / 60:.1f}m)")
    if total_papers > 0:
        console.print(f"  [magenta]Avg Time/Paper:[/magenta] {total_time / total_papers:.1f}s")
    console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")


def print_header(num_papers: int, max_concurrent: int):
    """Print processing header."""
    console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
    console.print(f"[bold]Processing {num_papers} papers (max {max_concurrent} concurrent)[/bold]")
    console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")


def print_completion():
    """Print completion message."""
    console.print("\n[bold green]✅ Processing complete![/bold green]")
