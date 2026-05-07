"""GitHub repo operations — create, check existence, resolve URL."""

from __future__ import annotations

import os
import re
import secrets
import subprocess

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aii_lib.run import emit
from aii_lib.utils.retry import make_retry_log


class GitHubRepoCreationError(Exception):
    """Raised when GitHub repo creation fails (retriable)."""


# =========================================================================
# Token isolation: gen_paper_repo uses AII_GH_TOKEN, not GH_TOKEN
# =========================================================================
# Paper repos are deployed under a dedicated GitHub account
# (``AII_GH_TOKEN`` in ``.env`` / RunPod secret store). Every other
# ``gh``/``git`` call in the repo (CI, this main repo, etc.) keeps using
# the parent process's ``GH_TOKEN``. The override is scoped to subprocess
# env dicts only — never mutates ``os.environ`` of the parent — so the
# two accounts can't trip over each other.


def gh_paper_token() -> str | None:
    """Return the token to use for paper-repo gh/git subprocess calls.

    Priority: ``AII_GH_TOKEN`` → ``GH_TOKEN`` → ``GITHUB_TOKEN``. Returns
    ``None`` if no token is set anywhere. Single-account setups (no
    ``AII_GH_TOKEN``) keep working via the ``GH_TOKEN`` fallback.
    """
    return (
        os.environ.get("AII_GH_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )


def gh_paper_env() -> dict[str, str]:
    """Subprocess env dict that pins ``GH_TOKEN`` to ``gh_paper_token()``.

    ``gh`` CLI reads ``GH_TOKEN`` (with ``GITHUB_TOKEN`` as a secondary).
    We always set both to the resolved paper-repo token so neither path
    can accidentally pick up the parent's main-account ``GH_TOKEN``.
    """
    env = os.environ.copy()
    tok = gh_paper_token()
    if tok:
        env["GH_TOKEN"] = tok
        env["GITHUB_TOKEN"] = tok
    return env


# =========================================================================
# REPO NAME GENERATION
# =========================================================================


def slugify(text: str, max_length: int = 40) -> str:
    """Convert text to URL-friendly slug.

    Args:
        text: Input text to slugify.
        max_length: Max slug length (default 40 to leave room for prefix/suffix
                    within GitHub's 100 char repo name limit).
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:max_length]


def generate_repo_name(title: str, prefix: str = "ai-invention") -> str:
    """Generate a unique repo name from a title.

    Format: {prefix}-{6_hex_chars}-{slugified_title}

    Args:
        title: Human-readable title (e.g. hypothesis title).
        prefix: Repo name prefix.

    Returns:
        Unique repo name like "ai-invention-a1b2c3-my-research-topic".
    """
    unique_id = secrets.token_hex(3)
    slug = slugify(title)
    name = f"{prefix}-{unique_id}-{slug}".rstrip("-")
    return name


def get_github_owner() -> str | None:
    """Get the authenticated GitHub username via gh CLI.

    Resolves whichever account ``AII_GH_TOKEN`` (then ``GH_TOKEN``)
    points to. Returns None if gh CLI is not available or not
    authenticated.
    """
    try:
        result = subprocess.run(
            ["gh", "api", "user", "-q", ".login"],
            capture_output=True,
            text=True,
            timeout=30,
            env=gh_paper_env(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_repo_url(title: str, prefix: str = "ai-invention") -> dict:
    """Generate repo name and resolve full GitHub URL.

    Does NOT create the repo — just determines what the URL will be.

    Args:
        title: Human-readable title for the repo.
        prefix: Repo name prefix.

    Returns:
        Dict with keys: repo_name, repo_url, repo_owner, error.
        repo_url is None if gh CLI is unavailable or not authenticated.
    """
    repo_name = generate_repo_name(title, prefix)
    result = {
        "repo_name": repo_name,
        "repo_url": None,
        "repo_owner": None,
        "error": None,
    }

    try:
        check_gh = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=gh_paper_env(),
        )
        if check_gh.returncode != 0:
            result["error"] = "gh CLI not installed"
            return result
    except FileNotFoundError:
        result["error"] = "gh CLI not installed"
        return result

    owner = get_github_owner()
    if not owner:
        result["error"] = "Not authenticated with gh CLI"
        return result

    result["repo_owner"] = owner
    result["repo_url"] = f"https://github.com/{owner}/{repo_name}"
    return result


# =========================================================================
# REPO CREATION
# =========================================================================


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=32),
    retry=retry_if_exception_type(GitHubRepoCreationError),
    before_sleep=make_retry_log(label="GitHub repo create"),
    reraise=True,
)
def _create_repo_with_retry(
    repo_name: str,
    description: str,
    private: bool = False,
) -> subprocess.CompletedProcess:
    """Create GitHub repo with tenacity retry on intermittent failures."""
    visibility = "--private" if private else "--public"
    result = subprocess.run(
        [
            "gh",
            "repo",
            "create",
            repo_name,
            visibility,
            "--description",
            description,
            "--clone=false",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=gh_paper_env(),
    )
    if result.returncode != 0 and "Repository creation failed" in result.stderr:
        raise GitHubRepoCreationError(result.stderr)
    return result


def create_repo(
    repo_name: str,
    description: str = "",
    private: bool = False,
) -> str | None:
    """Create a GitHub repo and return its URL, or None on failure.

    Args:
        repo_name: Full repo name (owner/name or just name for authenticated user).
        description: Repo description.
        private: Create as private repo (default: public).

    Returns:
        Repo URL (https://github.com/owner/name) or None.
    """
    from aii_lib.run import get_current_run

    run = get_current_run()

    if run is not None:
        emit.status_private_info(f"Creating GitHub repo: {repo_name}")
    try:
        result = _create_repo_with_retry(repo_name, description, private=private)
        if result.returncode == 0:
            owner = get_github_owner()
            name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
            url = f"https://github.com/{owner}/{name}" if owner else None
            if run is not None:
                emit.status_public_success(f"Created repo: {url}")
            return url
        if run is not None:
            emit.status_public_error(f"Failed to create repo: {result.stderr}")
        return None
    except GitHubRepoCreationError as e:
        if run is not None:
            emit.status_public_error(f"Failed to create repo after retries: {e}")
        return None


def ensure_repo_exists(
    repo_url: str,
    repo_name: str,
    description: str,
) -> tuple[bool, bool]:
    """Create the GitHub repo if it doesn't already exist.

    Returns ``(exists_or_created, created_now)``:
      * ``exists_or_created`` — True if the repo is usable for push (was
        already there or we just created it).
      * ``created_now`` — True if this call did the create; False if the
        repo was already present. Lets callers warn "we're about to
        overwrite an existing repo" instead of silently push-clobbering
        a previous run's contents, and gives ``repo_info.json`` the real
        ``created`` value instead of a hard-coded False.
    """
    parts = repo_url.rstrip("/").split("/")
    owner_repo = f"{parts[-2]}/{parts[-1]}"

    check = subprocess.run(
        ["gh", "repo", "view", owner_repo],
        capture_output=True,
        text=True,
        timeout=30,
        env=gh_paper_env(),
    )
    if check.returncode == 0:
        # Public warning (was private info): a repeat run with the same
        # hypothesis title resolves to the same repo name and silently
        # pushes over the previous paper / code. Surface this so the
        # user knows their prior contents are about to be replaced.
        emit.status_public_warning(f"Repo already exists, will overwrite contents: {repo_url}")
        return True, False

    url = create_repo(repo_name, description=description)
    return url is not None, url is not None
