"""GitHub CLI authentication helper.

Shared between aii_server and aii_pipeline for RunPod mode.
"""

import os
import subprocess
from pathlib import Path

from loguru import logger


def ensure_gh_auth(project_root: Path | None = None) -> None:
    """Authenticate GitHub CLI from GH_TOKEN env var or .env file."""
    result = subprocess.run(
        ["gh", "auth", "status"],
        capture_output=True,
        timeout=5,
    )
    if result.returncode == 0:
        logger.success("GitHub CLI already authenticated")
        return

    gh_token = os.environ.get("GH_TOKEN", "")
    if not gh_token and project_root:
        env_file = project_root / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GH_TOKEN="):
                    gh_token = line.split("=", 1)[1].strip().strip("'\"")
                    break

    if gh_token:
        result = subprocess.run(
            ["gh", "auth", "login", "--with-token"],
            input=gh_token,
            text=True,
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            subprocess.run(["gh", "auth", "setup-git"], capture_output=True, timeout=5)
            logger.success("GitHub CLI authenticated")
        else:
            logger.warning("GitHub CLI auth failed")
    else:
        logger.warning("GH_TOKEN not found — gh CLI unauthenticated")
