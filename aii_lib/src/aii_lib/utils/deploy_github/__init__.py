"""GitHub repository deployment utilities.

Deployer that pushes files to a GitHub repo in phases.
Each phase is a telemetry module that runs in asyncio.gather.

Usage:
    from aii_lib.utils.deploy_github import GitHubDeployer, resolve_repo_url, create_repo

    # Resolve URL (doesn't create the repo yet)
    info = resolve_repo_url("My Research Topic")

    # Create the repo
    url = create_repo(info["repo_name"], description="...")

    # Deploy in phases (use in asyncio.gather alongside other work)
    deployer = GitHubDeployer(repo_url=url, clone_dir=Path(...))
    await deployer.start()
    await asyncio.gather(
        deployer.run_phase("PHASE_SRC", copy_src_fn, buffer_sequence=0),
        other_module(),
    )
    deployer.cleanup()
"""

from .deployer import GitHubDeployer
from .repo_ops import (
    GitHubRepoCreationError,
    create_repo,
    ensure_repo_exists,
    generate_repo_name,
    get_github_owner,
    resolve_repo_url,
    slugify,
)

__all__ = [
    "GitHubDeployer",
    "GitHubRepoCreationError",
    "create_repo",
    "ensure_repo_exists",
    "generate_repo_name",
    "get_github_owner",
    "resolve_repo_url",
    "slugify",
]
