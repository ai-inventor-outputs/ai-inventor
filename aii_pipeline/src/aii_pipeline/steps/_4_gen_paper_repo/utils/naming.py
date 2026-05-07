"""Shared utilities for gen_paper_repo steps."""

import re

from aii_pipeline.steps._3_invention_loop.executors import sanitize_title

# Patterns that indicate the agent returned a status message instead of a real title
_BAD_TITLE_PATTERNS = [
    r"(?i)all\s+(output\s+)?files?\s+(verified|checked|created|generated)",
    r"(?i)^(success|done|complete|finished|ready|verified)",
    r"(?i)(under\s+size\s+limit|size\s+check|validation\s+pass)",
    r"(?i)^(task|step|process)\s+(complete|done|finished)",
    r"(?i)^(no\s+errors?|everything\s+(is\s+)?(ok|good|ready))",
]


def _is_bad_title(title: str) -> bool:
    """Check if a title looks like a status message rather than a real artifact title."""
    return any(re.search(pattern, title) for pattern in _BAD_TITLE_PATTERNS)


_TYPE_MAP = {
    "dataset": "dataset",
    "experiment": "experiment",
    "evaluation": "evaluation",
    "research": "research",
    "proof": "proof",
}
_TITLE_SUFFIX_MAX = 15


def _parse_artifact_id(artifact_id: str) -> tuple[str, str] | None:
    """Parse artifact_id into (type_prefix, iteration).

    ID format: gen_{type}_id{N}_it{iter}__{model}
    Example: "gen_experiment_id1_it1__opus" → ("experiment", "1")
    """
    m = re.match(
        r"(?:gen_)?(dataset|experiment|evaluation|proof|research)_id\d+_it(\d+)__\w+",
        artifact_id,
    )
    if m:
        atype = _TYPE_MAP.get(m.group(1), m.group(1))
        return atype, m.group(2)
    return None


def _short_title_suffix(title: str) -> str:
    """Extract a short informative suffix from a title (max 15 chars).

    E.g., "ISO-FIGS Benchmark Suite: 15-18 Tabular Datasets..." -> "iso_figs_benchm"
          "Breast Cancer Wisconsin Dataset Curation" -> "breast_cancer_w"
    """
    sanitized = sanitize_title(title, max_length=_TITLE_SUFFIX_MAX)
    return sanitized


def get_readable_folder_name(artifact_id: str, title: str = "") -> str:
    """Get human-readable folder name for an artifact in the repo.

    Format: {type}_iter{N}_{short_title}  (e.g., "dataset_iter1_iso_figs_benchm")

    Falls back to {type}_iter{N} if title is bad/missing, or sanitized artifact_id
    if the ID can't be parsed.

    Used by gen_art_demo and deploy_gh steps for consistent folder naming.

    Args:
        artifact_id: The artifact ID (e.g., "dataset_id1_it1__opus")
        title: Artifact title (from artifact.title)

    Returns:
        Sanitized folder name for use in repo structure
    """
    parsed = _parse_artifact_id(artifact_id)
    prefix = f"{parsed[0]}_iter{parsed[1]}" if parsed else None

    title_suffix = ""
    if title and not _is_bad_title(title):
        title_suffix = _short_title_suffix(title)

    if prefix and title_suffix:
        return f"{prefix}_{title_suffix}"
    if prefix:
        return prefix
    if title_suffix:
        return title_suffix
    return sanitize_title(artifact_id)


def build_github_code_mini_demo_data_url(repo_url: str | None, folder_name: str) -> str:
    """Build raw GitHub URL for mini_demo_data.json."""
    if not repo_url:
        return "UPDATE_THIS_URL_WITH_YOUR_REPO"
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    if "github.com/" in repo_url:
        parts = repo_url.split("github.com/")[-1]
        return (
            f"https://raw.githubusercontent.com/{parts}/main/{folder_name}/demo/mini_demo_data.json"
        )
    return "UPDATE_THIS_URL_WITH_YOUR_REPO"
