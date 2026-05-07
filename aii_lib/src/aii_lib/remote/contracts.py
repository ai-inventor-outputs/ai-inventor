"""Shared constants for orchestrator ↔ worker file-size contracts."""

# GitHub hard limit — files >=100MB are rejected by git push. Single source for
# the project default; downstream modules import this instead of redeclaring 100.
DEFAULT_MAX_FILE_SIZE_MB = 100
