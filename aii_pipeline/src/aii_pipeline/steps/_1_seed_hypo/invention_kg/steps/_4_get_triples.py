#!/usr/bin/env python3
"""
Triple extraction orchestrator with parallel processing.

This script handles:
1. Loading configuration and paper data
2. Parallel processing of papers with concurrency control
3. Progress tracking and cost/timing statistics
4. Summary reporting

The core logic for processing a single paper is in _get_triples/get_triple.py
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from aii_lib.run import emit

from ._get_triples import (  # type: ignore[import-not-found]
    console,
    create_progress_tracker,
    get_completed_paper_indices,
    get_triples_for_paper,
    load_papers_from_directory,
    print_completion,
    print_header,
    print_summary,
    setup_logging,
)

# ============================================================================
# Main Processing Orchestration
# ============================================================================


async def process_year(
    year: int | str,  # noqa: ARG001 — kept for API parity; reintroduce when per-year filtering returns
    papers: list,
    parent_run_dir: Path,
    config: dict[str, Any],
    max_concurrent: int = 10,
):
    """
    Process all papers for a single year in parallel.

    Args:
        year: Year to process (int or "all_years" for combined processing)
        papers: List of paper dicts with title and abstract (must have 'index' field)
        parent_run_dir: Parent run directory where individual paper folders will be created
        config: Agent configuration from triples_config.yaml
        max_concurrent: Maximum number of concurrent paper processing tasks
    """
    start_time = time.time()

    print_header(len(papers), max_concurrent)

    # Semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress tracking file
    progress_file = parent_run_dir / "completed_papers.txt"
    progress_lock = asyncio.Lock()  # Lock only for file writes

    # Shared state with single lock to prevent race conditions
    successful: int = 0
    failed: int = 0
    in_progress: int = 0
    state_lock = asyncio.Lock()  # Single lock for all shared state

    # Helper function to write completed paper to progress file
    async def write_progress(paper_index: int):
        """Write successfully completed paper index to progress file with lock."""
        async with progress_lock:
            with open(progress_file, "a") as f:
                f.write(f"{paper_index}\n")

    # Create progress tracker (simple logging)
    update_task_display = create_progress_tracker()

    # Modified wrapper to use new display system
    async def process_paper_with_display(idx: int, paper: dict):
        """Process paper and update display."""
        nonlocal successful, failed, in_progress

        paper_index = paper["index"]
        title = paper["title"]
        abstract = paper["abstract"]

        async with semaphore:
            try:
                # Update status to running and calculate stats atomically
                async with state_lock:
                    in_progress += 1
                    done = successful + failed
                    pending = len(papers) - done - in_progress

                # Debug logging handled by update_task_display

                await update_task_display(
                    f"[bold blue]Started Paper {paper_index}[/bold blue] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
                )

                paper_start = time.time()
                result = await get_triples_for_paper(
                    paper_id=idx,
                    paper_index=paper_index,
                    title=title,
                    abstract=abstract,
                    parent_run_dir=parent_run_dir,
                    config=config,
                )
                paper_time = time.time() - paper_start

                # Write to progress file immediately (only successful completions, outside state_lock for minimal blocking)
                if result:
                    await write_progress(paper_index)

                # Remove from running and update stats atomically
                async with state_lock:
                    in_progress -= 1

                    if result:
                        successful += 1
                    else:
                        failed += 1

                    done = successful + failed
                    pending = len(papers) - done - in_progress

                # Debug logging handled by update_task_display

                if result:
                    await update_task_display(
                        f"[bold green]✓ Done Paper {paper_index}[/bold green] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} [cyan]{paper_time:.0f}s[/cyan]"
                    )
                else:
                    await update_task_display(
                        f"[bold red]✗ Failed Paper {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title}"
                    )

                return result

            except Exception as e:
                # Log full exception
                emit.status_public_error(
                    f"Error processing paper {paper_index} ('{title}'): {type(e).__name__}: {e!s}"
                )

                # Update state atomically
                async with state_lock:
                    in_progress -= 1
                    failed += 1
                    done = successful + failed
                    pending = len(papers) - done - in_progress

                await update_task_display(
                    f"[bold red]✗ Error Paper {paper_index}[/bold red] | Pending: {pending}, In Progress: {in_progress}, Done: {done} | {title} ({type(e).__name__})"
                )
                return None

    # Launch all tasks concurrently
    tasks = [process_paper_with_display(idx, paper) for idx, paper in enumerate(papers)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Allow time for subprocess cleanup before event loop closes
    # Multiple subprocesses need time to terminate gracefully
    await asyncio.sleep(3)

    # Calculate total time
    total_time = time.time() - start_time

    # Summary
    print_summary(successful, failed, len(papers), total_time, "Processing Summary")

    return total_time, successful, failed


async def main(run_id: str, base_dir: Path, config: dict[str, Any]):
    """
    Process papers from all years.

    Args:
        run_id: Run ID for pipeline orchestration mode.
        base_dir: Base directory for kg runs (passed by parent _1_seed_hypo).
        config: Plain dict mirroring the kg ``get_triples`` section (with
            ``claude_agent`` nested). Built from the typed ``InventionKGConfig``
            by the parent orchestrator and used by the agent wrapper that
            still expects dict access.
    """
    from aii_pipeline.steps._1_seed_hypo.invention_kg.constants import (
        STEP_3_PAPERS_CLEAN,
        STEP_4_TRIPLES,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.utils import get_run_dir

    # Get settings from config (passed from main pipeline config.yaml)
    step_config = config.get("get_triples", {})
    max_papers = step_config.get("max_papers", -1)
    max_concurrent = step_config.get("max_concurrent_agents", 10)

    # Agent config is now embedded in step_config (from config.yaml get_triples.claude_agent)
    agent_config = step_config  # Pass the full step_config which includes claude_agent

    emit.status_public_info("Starting triple extraction")
    emit.status_private_info(f"max_papers: {max_papers}, max_concurrent: {max_concurrent}")

    # Determine input/output directories using run_id
    emit.status_private_info(f"Run ID: {run_id}")

    papers_clean_dir = get_run_dir(STEP_3_PAPERS_CLEAN, run_id, base_dir)
    parent_run_dir = get_run_dir(STEP_4_TRIPLES, run_id, base_dir)
    is_resume = parent_run_dir.exists()

    # Set up the get_triples log dir under this run's output.
    parent_run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(parent_run_dir, resume_dir=None)

    if is_resume:
        emit.status_public_info(f"Resuming step 4 from: {parent_run_dir}")
    else:
        emit.status_public_info(f"Creating step 4 output: {parent_run_dir}")

    emit.status_private_info(f"Input directory: {papers_clean_dir}")
    emit.status_private_info(f"Output directory: {parent_run_dir}")

    # Load papers from directory (uses config.py module)
    try:
        all_papers, papers_to_process, num_year_files = load_papers_from_directory(
            papers_clean_dir, max_papers
        )
    except FileNotFoundError as e:
        emit.status_public_error(str(e))
        return 1

    # Create limit message for display
    if max_papers == -1:
        limit_msg = f"Processing all {len(papers_to_process)} papers"
    else:
        limit_msg = f"Processing {len(papers_to_process)} papers (max_papers={max_papers}, total available: {len(all_papers)})"

    # Check for already completed papers (dynamic resume) - only if resume mode is enabled
    if is_resume:
        # Validation check includes its own detailed logging and folder cleanup
        completed_indices = get_completed_paper_indices(parent_run_dir)

        if completed_indices:
            emit.status_private_info(
                f"Completed paper indices: {sorted(completed_indices)[:10]}..."
                if len(completed_indices) > 10
                else f"Completed paper indices: {sorted(completed_indices)}"
            )

            # Filter out completed papers
            papers_before = len(papers_to_process)
            papers_to_process = [
                p for p in papers_to_process if p["index"] not in completed_indices
            ]
            papers_skipped = papers_before - len(papers_to_process)

            emit.status_public_info(f"Skipping {papers_skipped} already completed papers")
            emit.status_public_info(f"Will process {len(papers_to_process)} remaining papers")
        else:
            emit.status_private_info(
                f"No completed papers found, will process all {len(papers_to_process)} papers"
            )
    else:
        emit.status_private_info(
            f"Starting fresh - will process all {len(papers_to_process)} papers"
        )

    emit.status_public_info(f"{limit_msg}")
    emit.status_private_info(f"Max concurrent: {max_concurrent}, Year files: {num_year_files}")

    # Process all papers at once
    overall_start_time = time.time()
    _total_time, successful, failed = await process_year(
        "all_years",
        papers_to_process,
        parent_run_dir,
        agent_config,
        max_concurrent=max_concurrent,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    overall_time = time.time() - overall_start_time

    # Count all folders in directory for reporting
    total_folders = len(list(parent_run_dir.glob("paper_*")))

    print_summary(
        successful,
        failed,
        total_folders,
        overall_time,
        "Final Summary",
    )

    print_completion()
    emit.status_public_info(
        f"Final results: {successful} successful, {failed} failed out of {total_folders} total papers"
    )

    # Print helpful message for next steps
    console.print("\n[bold cyan]Output saved to:[/bold cyan]")
    console.print(f"  {parent_run_dir}")

    return 0
