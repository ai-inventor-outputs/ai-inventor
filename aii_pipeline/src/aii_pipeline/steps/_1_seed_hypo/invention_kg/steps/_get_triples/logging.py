#!/usr/bin/env python3
"""Logging configuration for triple extraction."""

from datetime import UTC, datetime
from pathlib import Path

from aii_lib.run import emit


def setup_logging(base_dir: Path, resume_dir: Path | None = None):
    """
    Setup logging directory and return log file path.

    Note: Console logging is already configured via aii_lib.telemetry.
    This function just determines the log file path for reference.

    Args:
        base_dir: Base directory under which a "logs/" subdirectory will be
            created (e.g., the get_triples run dir).
        resume_dir: If resuming, the directory being resumed from (to reuse log file)

    Returns:
        Path to the log file (for reference, actual logging uses telemetry)
    """
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log file name
    if resume_dir is not None:
        # Extract timestamp from resume directory name (format: {num}_{timestamp})
        dir_name = resume_dir.name
        # Try to extract timestamp (anything after first underscore)
        parts = dir_name.split("_", 1)
        if len(parts) > 1:
            timestamp_part = parts[1]
            # Look for existing log file with this timestamp
            existing_log = log_dir / f"bblocks_{timestamp_part}.log"
            if existing_log.exists():
                log_file = existing_log
                emit.status_private_info(f"Resuming - appending to existing log file: {log_file}")
            else:
                # Log file doesn't exist, create new one with same timestamp
                log_file = existing_log
        else:
            # Couldn't parse timestamp, create new log file
            log_file = log_dir / f"bblocks_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    else:
        # Fresh start - create new log file
        log_file = log_dir / f"bblocks_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

    emit.status_private_info(f"Log directory: {log_dir}")
    return log_file
