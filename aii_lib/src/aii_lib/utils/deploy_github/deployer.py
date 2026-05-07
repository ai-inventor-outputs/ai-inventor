"""Generic incremental GitHub deployer.

Manages a persistent repo clone and pushes files in phases. Each phase
runs a copy_fn that writes into the clone, then commits and pushes.

Usage (sequential — used by gen_paper_repo's deploy_gh substep):
    deployer = GitHubDeployer(repo_url="https://github.com/...", clone_dir=Path(...))
    await deployer.start()
    await deployer.run_phase("deploy_src", copy_fn_src)
    await deployer.run_phase("deploy_paper", copy_fn_paper)
    deployer.cleanup()
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from aii_lib.run import emit

from .repo_ops import ensure_repo_exists, gh_paper_env, gh_paper_token

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_CHUNK_MAX_BYTES = 1_000_000_000  # 1GB per push
_MIN_PUSH_INTERVAL = 2  # seconds between pushes
_PUSH_TIMEOUT = 1200  # 20 minutes per push


class GitHubDeployer:
    """Incremental GitHub deployer.

    Phases run as regular async functions in asyncio.gather alongside other pipeline work.

    Args:
        repo_url: GitHub repo URL (https://github.com/owner/repo)
        clone_dir: Where to clone the repo (will be created/overwritten)
        repo_name: For repo creation (owner/name format). If None, repo must already exist.
        repo_description: Description for newly created repos.
        chunk_max_bytes: Max bytes per git push chunk.
        push_timeout: Timeout per push in seconds.
        min_push_interval: Min seconds between consecutive pushes (rate limiting).
    """

    def __init__(
        self,
        repo_url: str,
        clone_dir: Path,
        repo_name: str | None = None,
        repo_description: str = "",
        chunk_max_bytes: int = _CHUNK_MAX_BYTES,
        push_timeout: int = _PUSH_TIMEOUT,
        min_push_interval: int = _MIN_PUSH_INTERVAL,
        commit_author_name: str = "AI Inventor",
        commit_author_email: str = "ai-inventor@noreply",
    ):
        self._repo_url = repo_url
        self._clone_dir = clone_dir
        self._repo_name = repo_name
        self._repo_description = repo_description
        self._chunk_max_bytes = chunk_max_bytes
        self._push_timeout = push_timeout
        self._min_push_interval = min_push_interval
        self._commit_author_name = commit_author_name
        self._commit_author_email = commit_author_email

        self._last_push_time = 0.0
        self._push_lock = asyncio.Lock()  # serialize git operations on shared clone

    @property
    def clone_dir(self) -> Path:
        """Return the repository clone directory."""
        return self._clone_dir

    async def start(self) -> bool:
        """Create repo (if needed), clone. Returns True if setup succeeded."""
        if not self._repo_url:
            emit.status_public_error("Deploy ABORTED: No repo_url provided")
            return False

        # Check gh CLI (with the paper-repo token, not the parent's GH_TOKEN)
        try:
            check_gh = await asyncio.to_thread(
                subprocess.run,
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=30,
                env=gh_paper_env(),
            )
            if check_gh.returncode != 0:
                emit.status_public_warning("gh CLI not authenticated, skipping deployment")
                return False
        except FileNotFoundError:
            emit.status_public_warning("gh CLI not installed, skipping deployment")
            return False

        # Create repo if needed
        if self._repo_name:
            ok, created_now = ensure_repo_exists(
                repo_url=self._repo_url,
                repo_name=self._repo_name,
                description=self._repo_description or "AI-generated research project",
            )
            if not ok:
                emit.status_public_error("Cannot deploy — repo creation failed")
                return False
            self._created_now = created_now

        # Clone
        if self._clone_dir.exists():
            shutil.rmtree(self._clone_dir)

        emit.status_private_info("Cloning repository for incremental deploy...")
        clone_result = await asyncio.to_thread(
            subprocess.run,
            ["gh", "repo", "clone", self._repo_url, str(self._clone_dir)],
            capture_output=True,
            text=True,
            timeout=300,
            env=gh_paper_env(),
        )
        if clone_result.returncode != 0:
            emit.status_public_warning(f"Failed to clone: {clone_result.stderr}")
            return False

        # Per-clone git config (NOT --global) — keeps the developer's main
        # git identity untouched. Values come from the deploy_gh config
        # block in pipeline.yaml (``gen_paper_repo.github.commit_author_*``).
        for key, val in [
            ("user.name", self._commit_author_name),
            ("user.email", self._commit_author_email),
        ]:
            await asyncio.to_thread(
                subprocess.run,
                ["git", "config", key, val],
                cwd=self._clone_dir,
                capture_output=True,
                timeout=10,
            )

        # Inject the paper-repo token into the remote URL for push auth.
        # ``gh_paper_token()`` resolves AII_GH_TOKEN → GH_TOKEN → GITHUB_TOKEN
        # so single-account setups still work.
        gh_token = gh_paper_token()
        if gh_token:
            auth_url = self._repo_url.replace(
                "https://github.com/",
                f"https://x-access-token:{gh_token}@github.com/",
            )
            if not auth_url.endswith(".git"):
                auth_url += ".git"
            await asyncio.to_thread(
                subprocess.run,
                ["git", "remote", "set-url", "origin", auth_url],
                cwd=self._clone_dir,
                capture_output=True,
                timeout=10,
            )

        emit.status_public_success("Repository cloned — ready for incremental deploy")
        return True

    def cleanup(self) -> None:
        """Remove the repo clone directory."""
        if self._clone_dir.exists():
            shutil.rmtree(self._clone_dir, ignore_errors=True)

    async def run_phase(
        self,
        module_name: str,
        copy_fn: Callable[[Path], list[str]],
        buffer_sequence: int | None = None,  # noqa: ARG002
    ) -> None:
        """Run a deploy phase.

        ``buffer_sequence`` accepted for caller
        compat but unused — the v26 Run model handles parallel-module
        ordering.
        """
        files = copy_fn(self._clone_dir)

        if files:
            async with self._push_lock:
                await self._push_files(module_name, files)
            emit.status_public_success(f"{module_name} complete: {len(files)} files pushed")
        else:
            emit.status_private_info(f"{module_name}: no files to push")

    # =========================================================================
    # GIT PUSH
    # =========================================================================

    async def _push_files(self, label: str, files: list[str]) -> None:
        """Add, commit, and push a set of files in chunks."""
        repo_dir = self._clone_dir

        file_sizes = []
        for rel_path in files:
            full = repo_dir / rel_path
            size = full.stat().st_size if full.exists() else 0
            file_sizes.append((rel_path, size))

        total_bytes = sum(s for _, s in file_sizes)
        emit.status_public_info(
            f"[{label}] Pushing {len(files)} files ({total_bytes / (1024 * 1024):.0f}MB)"
        )

        # Partition into chunks
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_size = 0
        for rel_path_str, size in file_sizes:
            if current_chunk and current_size + size > self._chunk_max_bytes:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(rel_path_str)
            current_size += size
        if current_chunk:
            chunks.append(current_chunk)

        for chunk_idx, chunk_files in enumerate(chunks):
            chunk_bytes = sum(
                (repo_dir / p).stat().st_size for p in chunk_files if (repo_dir / p).exists()
            )
            chunk_label = f"{label} [{chunk_idx + 1}/{len(chunks)}]" if len(chunks) > 1 else label

            # Drop paths matched by .gitignore before staging. Without this, `git add`
            # exits non-zero on the first ignored path and aborts staging — but the
            # non-ignored paths are already partially in the index, so the next
            # commit silently picks them up (or, in the last phase, drops them on
            # cleanup). See `git add --pathspec-from-file` behavior.
            check_result = await asyncio.to_thread(
                subprocess.run,
                ["git", "check-ignore", "--stdin"],
                input="\n".join(chunk_files),
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            # rc 0 = some paths ignored, rc 1 = none ignored, rc 128 = error
            ignored = (
                set(check_result.stdout.splitlines())
                if check_result.returncode in (0, 1)
                else set()
            )
            chunk_files = [p for p in chunk_files if p not in ignored]
            if ignored:
                emit.status_private_info(
                    f"[{chunk_label}] Skipped {len(ignored)} .gitignore-matched paths"
                )
            if not chunk_files:
                emit.status_private_info(f"[{chunk_label}] All paths ignored — nothing to commit")
                continue

            # git add
            add_result = await asyncio.to_thread(
                subprocess.run,
                ["git", "add", "--pathspec-from-file=-"],
                input="\n".join(chunk_files),
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if add_result.returncode != 0:
                emit.status_public_warning(f"[{chunk_label}] git add failed: {add_result.stderr}")
                # Unstage any partial-add so the next chunk's commit doesn't pick
                # them up, and the final cleanup doesn't lose them silently.
                await asyncio.to_thread(
                    subprocess.run,
                    ["git", "reset", "HEAD", "--", *chunk_files],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                )
                continue  # skip commit+push for this chunk

            # git commit
            commit_msg = (
                f"Add {label} (part {chunk_idx + 1}/{len(chunks)})"
                if len(chunks) > 1
                else f"Add {label}"
            )
            commit_result = await asyncio.to_thread(
                subprocess.run,
                ["git", "commit", "-m", commit_msg],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if commit_result.returncode != 0:
                combined = (commit_result.stdout + commit_result.stderr).lower()
                if (
                    "nothing to commit" in combined
                    or "no changes added" in combined
                    or "nothing added" in combined
                ):
                    emit.status_private_info(f"[{chunk_label}] Nothing to commit — skipping push")
                    continue
                detail = (commit_result.stderr or commit_result.stdout or "").strip()
                emit.status_public_warning(
                    f"[{chunk_label}] git commit rc={commit_result.returncode}: {detail}"
                )
                continue  # skip push — don't push stale data

            # Rate-limit wait
            elapsed = time.time() - self._last_push_time
            if self._last_push_time > 0 and elapsed < self._min_push_interval:
                wait = self._min_push_interval - elapsed
                emit.status_private_info(f"[{chunk_label}] Rate-limit wait {wait:.0f}s...")
                await asyncio.sleep(wait)

            # git push with retry
            emit.status_private_info(f"[{chunk_label}] git push...")
            max_retries = 3
            for attempt in range(max_retries):
                push_result = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "push"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._push_timeout,
                )
                if push_result.returncode == 0:
                    self._last_push_time = time.time()
                    break

                stderr = push_result.stderr.lower()
                is_transient = any(
                    err in stderr
                    for err in [
                        "408",
                        "timeout",
                        "timed out",
                        "connection reset",
                        "unexpected disconnect",
                        "hung up",
                        "network",
                    ]
                )
                if is_transient and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    emit.status_public_warning(
                        f"[{chunk_label}] Push failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"Git push failed: {push_result.stderr}")

            emit.status_public_success(
                f"[{chunk_label}] Pushed ({chunk_bytes / (1024 * 1024):.0f}MB)"
            )
