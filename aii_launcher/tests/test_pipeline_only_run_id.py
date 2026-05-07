"""``deploy_pipeline_only`` derives session name + orphan pattern from
the ``--fork-new-run-id=<id>`` / ``--resume-run-id=<id>`` args passed to
the pipeline.

Both equals form (``--fork-new-run-id=foo``) and two-token form
(``--fork-new-run-id foo``) must work.
"""

from __future__ import annotations

from unittest.mock import patch

from aii_launcher import deploy as dmod


def _setup_mocks():
    """Stub out everything that touches the system (tmux, fs, pid lookup, signal)."""
    return [
        patch("aii_lib.utils.tmux.launch_in_tmux"),
        patch("aii_lib.utils.tmux.session_exists", return_value=True),
        patch("aii_lib.utils.tmux.get_pane_pid", return_value=1234),
        patch(
            "aii_lib.utils.tmux.pipeline_session_name",
            side_effect=lambda rid: f"aii-{rid.replace('.', '-')}",
        ),
        patch(
            "aii_lib.utils.paths.logs_dir",
            side_effect=lambda c: __import__("pathlib").Path("/tmp") / c,
        ),
        patch.object(dmod.time, "sleep"),
        # ``deploy_pipeline_only`` does ``os.kill(pid, 0)`` as a liveness
        # probe; pid=1234 isn't real in the test process, so stub it out.
        patch.object(dmod.os, "kill"),
    ]


def _enter(mocks):
    return [m.__enter__() for m in mocks]


def _exit(mocks):
    for m in mocks:
        m.__exit__(None, None, None)


def test_fork_new_run_id_parsed_from_equals_form():
    """``--fork-new-run-id=run-001`` → session ``aii-run-001``."""
    mocks = _setup_mocks()
    handles = _enter(mocks)
    try:
        launch_mock = handles[0]
        rc = dmod.deploy_pipeline_only(pipeline_args=["--fork-new-run-id=run-001"])
        assert rc == 0
        kw = launch_mock.call_args.kwargs
        assert kw["session"] == "aii-run-001"
        assert "-id=run-001" in kw["orphan_pattern"]
    finally:
        _exit(mocks)


def test_resume_run_id_parsed_from_space_form():
    """``--resume-run-id run-002`` (two tokens) is also supported."""
    mocks = _setup_mocks()
    handles = _enter(mocks)
    try:
        launch_mock = handles[0]
        rc = dmod.deploy_pipeline_only(
            pipeline_args=["--resume-run-id", "run-002", "--other", "x"],
        )
        assert rc == 0
        kw = launch_mock.call_args.kwargs
        assert kw["session"] == "aii-run-002"
        assert "-id=run-002" in kw["orphan_pattern"]
    finally:
        _exit(mocks)


def test_no_run_id_falls_back_to_default_session_and_no_orphan_pattern():
    """Without --fork-new-run-id/--resume-run-id, we use the static PIPELINE_SESSION."""
    mocks = _setup_mocks()
    handles = _enter(mocks)
    try:
        launch_mock = handles[0]
        rc = dmod.deploy_pipeline_only(pipeline_args=["--foo", "bar"])
        assert rc == 0
        kw = launch_mock.call_args.kwargs
        assert kw["session"] == dmod.PIPELINE_SESSION
        # orphan_pattern is None when run_id couldn't be derived — this
        # avoids killing concurrent runs whose ids we can't disambiguate.
        assert kw["orphan_pattern"] is None
    finally:
        _exit(mocks)


def test_explicit_session_name_overrides_derivation():
    """An explicit ``session_name`` arg wins over derivation from pipeline_args."""
    mocks = _setup_mocks()
    handles = _enter(mocks)
    try:
        launch_mock = handles[0]
        rc = dmod.deploy_pipeline_only(
            pipeline_args=["--fork-new-run-id=should-be-ignored"],
            session_name="my-custom-session",
        )
        assert rc == 0
        kw = launch_mock.call_args.kwargs
        # session_name passes through verbatim …
        assert kw["session"] == "my-custom-session"
        # … but the orphan pattern still tracks the parsed run_id, since
        # that's what the pipeline.cli process actually has in argv.
        assert "-id=should-be-ignored" in kw["orphan_pattern"]
    finally:
        _exit(mocks)


def test_returns_1_when_session_fails_to_start():
    """If get_pane_pid returns None (no PID = no session), return 1."""
    with (
        patch("aii_lib.utils.tmux.launch_in_tmux"),
        patch("aii_lib.utils.tmux.session_exists", return_value=False),
        patch("aii_lib.utils.tmux.get_pane_pid", return_value=None),
        patch(
            "aii_lib.utils.tmux.pipeline_session_name",
            side_effect=lambda rid: f"aii-{rid}",
        ),
        patch(
            "aii_lib.utils.paths.logs_dir",
            side_effect=lambda c: __import__("pathlib").Path("/tmp") / c,
        ),
        patch.object(dmod.time, "sleep"),
        patch.object(dmod.os, "kill"),
    ):
        rc = dmod.deploy_pipeline_only(pipeline_args=["--fork-new-run-id=ghost"])
        assert rc == 1
