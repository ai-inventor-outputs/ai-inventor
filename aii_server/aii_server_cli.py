"""CLI entry point for aii_server — thin wrapper that imports the real main().

Setuptools discovers this as a top-level module (via where=["."] in pyproject.toml).
The actual server logic lives in aii_server.py.
"""

import sys
from pathlib import Path

# Ensure aii_server/ is on sys.path so Django can import config, dashboard, etc.
#
# NOTE: sys.path.insert normally violates rule #5 (no ugly hacks), but is
# unavoidable here. When launched via setuptools entry point, the CWD and
# sys.path don't contain aii_server/'s siblings (config/, dashboard/,
# agent_abilities/), which Django loads by top-level name (DJANGO_SETTINGS_MODULE
# = "config.settings"). There's no way to register these as proper packages
# without moving them under aii_server/ and rewriting every Django import.
_server_dir = str(Path(__file__).resolve().parent)
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from aii_server import main  # noqa: E402

__all__ = ["main"]
