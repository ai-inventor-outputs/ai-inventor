"""RUNPOD_API_KEY guard rails on ``deploy_runpod`` / ``resume_stream``.

Both helpers refuse to do anything when the key is missing — and they must
return exit-code ``1`` (not raise) so the CLI can ``sys.exit(...)`` cleanly.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from aii_launcher import deploy as dmod


@pytest.fixture(autouse=True)
def _stub_aii_runpod(monkeypatch):
    """Stub aii_runpod when not installed (public/open-source builds).

    deploy_runpod / resume_stream now start with ``import aii_runpod`` so
    public builds get a friendly error instead of a raw ModuleNotFoundError.
    These tests target the API-key guard one step later, so we register a
    fake module when the real one isn't available.
    """
    if "aii_runpod" not in sys.modules:
        monkeypatch.setitem(sys.modules, "aii_runpod", types.ModuleType("aii_runpod"))


def _run(coro):
    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if asyncio.get_event_loop_policy().get_event_loop().is_running()
        else asyncio.run(coro)
    )


def test_deploy_runpod_returns_1_without_api_key(monkeypatch, capsys):
    """deploy_runpod must short-circuit with rc=1 when RUNPOD_API_KEY is empty."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = asyncio.run(dmod.deploy_runpod())
    assert rc == 1
    captured = capsys.readouterr()
    assert "RUNPOD_API_KEY" in captured.out


def test_deploy_runpod_returns_1_when_api_key_blank(monkeypatch, capsys):
    """An empty-string env var is treated the same as missing."""
    monkeypatch.setenv("RUNPOD_API_KEY", "")
    rc = asyncio.run(dmod.deploy_runpod())
    assert rc == 1
    assert "RUNPOD_API_KEY" in capsys.readouterr().out


def test_resume_stream_returns_1_without_api_key(monkeypatch, capsys):
    """resume_stream guards on the same env var."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = asyncio.run(dmod.resume_stream("some-pod"))
    assert rc == 1
    assert "RUNPOD_API_KEY" in capsys.readouterr().out
