"""Module-level invariants for ``aii_launcher.deploy``.

Verifies the constants and path derivations that other code (server-side
tmux session naming, log paths, orchestrator launch) silently depends on.
A regression in any of these would silently break deployment with
confusing downstream failures, so pin them here.
"""

from __future__ import annotations

import sys

from aii_launcher import deploy as dmod


def test_project_root_points_to_repo_root():
    """PROJECT_ROOT must be the repo root (4 levels up from deploy.py)."""
    # deploy.py is at <repo>/aii_launcher/src/aii_launcher/deploy.py
    # PROJECT_ROOT = parent.parent.parent.parent must be <repo>.
    assert dmod.PROJECT_ROOT.is_dir(), "PROJECT_ROOT must exist"
    assert (dmod.PROJECT_ROOT / "aii_launcher").is_dir()
    assert (dmod.PROJECT_ROOT / "aii_launcher" / "pyproject.toml").is_file()
    assert (dmod.PROJECT_ROOT / "aii_server").is_dir()


def test_session_names_are_distinct_and_stable():
    """Server / pipeline session names must be distinct stable strings."""
    assert dmod.SERVER_SESSION == "aii-server"
    assert dmod.PIPELINE_SESSION == "aii-pipeline"
    assert dmod.SERVER_SESSION != dmod.PIPELINE_SESSION


def test_py_returns_venv_when_present(tmp_path, monkeypatch):
    """``_py()`` returns the project's .venv interpreter when it exists."""
    fake_root = tmp_path
    fake_python = fake_root / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("")
    monkeypatch.setattr(dmod, "PROJECT_ROOT", fake_root)

    assert dmod._py() == str(fake_python)


def test_py_falls_back_to_sys_executable_when_no_venv(tmp_path, monkeypatch):
    """When no .venv/bin/python exists, ``_py()`` falls back to sys.executable."""
    monkeypatch.setattr(dmod, "PROJECT_ROOT", tmp_path)
    # No .venv created → fallback path.
    assert dmod._py() == sys.executable
