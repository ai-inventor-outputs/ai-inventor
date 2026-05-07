"""Pipeline-specific incremental deploy — uses aii_lib.utils.deploy_github.

Sequential wrapper around GitHubDeployer used by step 5 (deploy_gh). Each
push phase shares the active deploy_gh module so all messages group
under one "deploy_gh" bucket. No background tasks, no buffered output —
the whole step is linear now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.remote.contracts import DEFAULT_MAX_FILE_SIZE_MB
from aii_lib.run import emit
from aii_lib.utils.deploy_github import GitHubDeployer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class PipelineDeployer:
    """Sequential wrapper around GitHubDeployer for the deploy_gh substep.

    Caller drives `start()` → `run_phase()` × N → `cleanup()`. Each phase
    runs under the active deploy_gh module (the caller already started
    it), so all push logs share that one bucket.
    """

    def __init__(
        self,
        repo_url: str,
        repo_name: str,
        repo_description: str,
        output_dir: Path,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        chunk_max_bytes: int = 1_000_000_000,
        push_timeout: int = 1200,
        min_push_interval: int = 2,
        commit_author_name: str = "AI Inventor",
        commit_author_email: str = "ai-inventor@noreply",
    ):
        self._repo_url = repo_url
        self._output_dir = output_dir
        self._max_file_size_mb = max_file_size_mb

        self._deployer = GitHubDeployer(
            repo_url=repo_url,
            clone_dir=output_dir / "_5_deploy_gh" / "_repo_clone",
            repo_name=repo_name,
            repo_description=repo_description,
            chunk_max_bytes=chunk_max_bytes,
            push_timeout=push_timeout,
            min_push_interval=min_push_interval,
            commit_author_name=commit_author_name,
            commit_author_email=commit_author_email,
        )

    async def start(self) -> bool:
        """Create repo on GitHub if needed, clone locally."""
        return await self._deployer.start()

    def cleanup(self) -> None:
        """Remove the local clone directory."""
        self._deployer.cleanup()

    async def run_phase(
        self,
        label: str,
        copy_fn: Callable[[Path], list[str]],
    ) -> None:
        """Run one copy → commit → push cycle.

        `label` is just a display tag for the phase ("src", "demos",
        "paper"); all output groups under the active module
        (we don't wrap a sub-module). Files written by `copy_fn` go to
        the clone dir, then `_push_files` adds + commits + pushes them.
        """
        files = copy_fn(self._deployer.clone_dir)
        if files:
            await self._deployer._push_files(label, files)  # type: ignore[attr-defined]
            emit.status_public_success(f"{label}: pushed {len(files)} files")
        else:
            emit.status_private_info(f"{label}: no files to push")
