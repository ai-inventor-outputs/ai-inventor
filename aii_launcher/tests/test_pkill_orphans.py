"""``aii_lib.utils.tmux.pkill_orphans`` is a tmux-aware orphan reaper.

Verifies the wrapper passes the pattern through to ``pgrep -f``, filters out
PIDs whose ``/proc/<pid>/comm`` is ``tmux:*`` (the daemon inherits the
matched cmdline), and signals survivors with SIGTERM → sleep → SIGKILL so
multiprocessing Manager children that swallow SIGTERM still die.
"""

from __future__ import annotations

import signal
import subprocess
from unittest.mock import patch

from aii_lib.utils import tmux as tmod


def test_pkill_orphans_skips_tmux_daemon_and_two_phase_kills_others():
    pattern = "aii_pipeline.cli.*run_name=run-001"
    pgrep_stdout = "1234\n5678\n9012\n"

    fake_pgrep = subprocess.CompletedProcess(
        args=["pgrep", "-f", pattern], returncode=0, stdout=pgrep_stdout, stderr=""
    )

    # 1234 is a real Python orphan; 5678 is the tmux daemon (which inherits the
    # matched cmdline and must be skipped); 9012 is another orphan.
    comm_by_pid = {1234: "python", 5678: "tmux: server", 9012: "python"}

    def fake_read_text(self, *args, **kwargs):
        for pid, comm in comm_by_pid.items():
            if str(self).endswith(f"/proc/{pid}/comm"):
                return comm + "\n"
        raise OSError("unknown pid")

    with (
        patch.object(tmod.subprocess, "run", return_value=fake_pgrep) as mock_run,
        patch.object(tmod, "os") as mock_os,
        patch.object(tmod.time, "sleep") as mock_sleep,
        patch.object(tmod.Path, "read_text", new=fake_read_text),
    ):
        tmod.pkill_orphans(pattern)

        # Two pgrep invocations: one before SIGTERM, one before SIGKILL.
        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            argv = call.args[0]
            assert argv == ["pgrep", "-f", pattern]
            assert call.kwargs["check"] is False

        # SIGTERM and SIGKILL each fire once per non-tmux PID (1234, 9012).
        # The daemon PID 5678 is skipped because its /proc/.../comm is "tmux: server".
        kill_calls = mock_os.kill.call_args_list
        kill_targets = [(c.args[0], c.args[1]) for c in kill_calls]
        assert (1234, signal.SIGTERM) in kill_targets
        assert (9012, signal.SIGTERM) in kill_targets
        assert (1234, signal.SIGKILL) in kill_targets
        assert (9012, signal.SIGKILL) in kill_targets
        # 5678 must NEVER be signalled.
        assert all(target_pid != 5678 for target_pid, _ in kill_targets)

        # Two sleeps: between SIGTERM/SIGKILL, and after SIGKILL before the
        # replacement process tries to bind the port.
        assert mock_sleep.call_count == 2
        for call in mock_sleep.call_args_list:
            assert 0 < call.args[0] <= 1.0
