"""``stop_all()`` collects fixed sessions + dynamic pipeline-run sessions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aii_launcher import deploy as dmod


def test_stop_all_returns_0_when_nothing_running(capsys):
    """If no sessions exist, returns 0 with 'Nothing running' message."""
    with (
        patch("aii_lib.utils.tmux.session_exists", return_value=False),
        patch("aii_lib.utils.tmux.kill_session"),
        patch.object(dmod.subprocess, "run") as mock_run,
        patch.object(dmod.time, "sleep"),
    ):
        # tmux list-sessions returns no pipeline-run sessions
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        rc = dmod.stop_all()

        assert rc == 0
        assert "Nothing running" in capsys.readouterr().out


def test_stop_all_discovers_pipeline_run_sessions(capsys):
    """Pipeline-run sessions (``aii-<id>``) from ``tmux list-sessions`` get stopped.

    Excludes static services (``aii-server``, ``aii-storybook``, etc.) — those
    have their own slots in ``stop_all``'s fixed-session list and shouldn't
    double-up via the dynamic pipeline-run sweep.
    """
    fake_running = {"aii-foo", "aii-bar"}

    def session_exists(name):
        return name in fake_running

    list_result = MagicMock(
        returncode=0,
        stdout="aii-foo\naii-bar\nunrelated-session\n",
    )

    with (
        patch("aii_lib.utils.tmux.session_exists", side_effect=session_exists),
        patch("aii_lib.utils.tmux.kill_session"),
        patch.object(dmod.subprocess, "run") as mock_run,
        patch.object(dmod.time, "sleep"),
    ):
        mock_run.return_value = list_result

        rc = dmod.stop_all()

        assert rc == 0
        out = capsys.readouterr().out
        assert "Stopped" in out
        assert "aii-foo" in out
        assert "aii-bar" in out

        sent_to = [
            call.args[0][3]  # tmux send-keys -t <session>
            for call in mock_run.call_args_list
            if call.args[0][:3] == ["tmux", "send-keys", "-t"]
        ]
        assert "aii-foo" in sent_to
        assert "aii-bar" in sent_to


def test_stop_all_skips_when_tmux_missing(capsys):
    """If ``tmux list-sessions`` raises FileNotFoundError, fall back to fixed list."""
    with (
        patch("aii_lib.utils.tmux.session_exists", return_value=False),
        patch("aii_lib.utils.tmux.kill_session"),
        patch.object(dmod.subprocess, "run", side_effect=FileNotFoundError),
        patch.object(dmod.time, "sleep"),
    ):
        # No session_exists → no stops → "Nothing running"
        rc = dmod.stop_all()
        assert rc == 0
        assert "Nothing running" in capsys.readouterr().out
