"""``deploy.py`` calls ``launch_in_tmux`` with the AII pipeline env defaults.

Every aii server/pipeline tmux launch must carry ``CLAUDE_CONFIG_DIR``
(NFS-shared creds across pods) and ``AII_BUFFER_TRACE`` (drain diagnostics);
the helper ``_aii_pipeline_env`` is the single source of truth for that
dict. A regression here silently breaks RunPod cred sharing or hides
buffer-drain stalls, so verify both the env preamble and the orphan-pattern
pass-through to ``launch_in_tmux``.
"""

from __future__ import annotations

from unittest.mock import patch

from aii_launcher import deploy as dmod


def test_aii_pipeline_env_carries_required_keys():
    """The shared env dict pins CLAUDE_CONFIG_DIR + AII_BUFFER_TRACE."""
    env = dmod._aii_pipeline_env()
    assert env["AII_BUFFER_TRACE"] == "1"
    cfg = env["CLAUDE_CONFIG_DIR"]
    assert cfg.endswith("/aii_data/.claude")
    # Anchored under PROJECT_ROOT so RunPod's NFS mount picks it up.
    assert str(dmod.PROJECT_ROOT) in cfg


def test_pipeline_only_passes_aii_env_to_launch_in_tmux():
    """``deploy_pipeline_only`` must forward ``_aii_pipeline_env()`` verbatim."""
    with (
        patch("aii_lib.utils.tmux.launch_in_tmux") as mock_launch,
        patch("aii_lib.utils.tmux.session_exists", return_value=True),
        patch("aii_lib.utils.tmux.get_pane_pid", return_value=1234),
        patch(
            "aii_lib.utils.paths.logs_dir",
            side_effect=lambda c: __import__("pathlib").Path("/tmp") / c,
        ),
        patch.object(dmod.time, "sleep"),
        patch.object(dmod.os, "kill"),
    ):
        rc = dmod.deploy_pipeline_only(pipeline_args=["--fork-new-run-id=run-001"])
        assert rc == 0

        kw = mock_launch.call_args.kwargs
        assert kw["extra_env"] == dmod._aii_pipeline_env()
        # session + orphan_pattern derived from the run-id are still threaded
        assert kw["session"] == "aii-run-001"
        assert "-id=run-001" in kw["orphan_pattern"]
