"""Server URL resolution — how aii_lib finds the aii_server.

Reads server settings from aii_config/server/server.yaml (single source of truth).
AII_SERVER_URL env var overrides for remote deployments.

Usage:
    from aii_lib.server_url import ability_service_url, SERVER_PORT, DEFAULT_SERVER_PORT
"""

import os
from pathlib import Path

# Hardcoded fallback when server.yaml is missing or has no port set.
# Single source for the project default — every other module imports this.
DEFAULT_SERVER_PORT = 8020

_CONFIG_FILE = Path(__file__).parents[3] / "aii_config" / "server" / "server.yaml"
from aii_lib.utils.config_overrides import load_config_with_overrides

_server_config: dict = load_config_with_overrides(_CONFIG_FILE)

SERVER_PORT = int(_server_config.get("server", {}).get("port", DEFAULT_SERVER_PORT))
SERVER_HOST = _server_config.get("server", {}).get("host", "localhost")


def ability_service_url() -> str:
    """Return the aii_server base URL (no path suffix).

    Priority: AII_SERVER_URL env var > localhost (port from server.yaml).
    Callers add their own path prefix (e.g. /abilities, /api).
    """
    url = os.environ.get("AII_SERVER_URL")
    if url:
        return url.rstrip("/")
    return f"http://localhost:{SERVER_PORT}"
