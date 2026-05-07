"""Agent hooks — time-remaining and project skills link setup."""

from __future__ import annotations

import os
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import HookMatcher

from aii_lib.run import emit

if TYPE_CHECKING:
    from claude_agent_sdk.types import SyncHookJSONOutput

    from ..models import AgentOptions


def ensure_project_skills_link(options: AgentOptions) -> None:
    """Ensure .claude/ is reachable from the agent's CWD.

    The Claude SDK discovers skills by walking up from CWD to find .claude/.
    When the agent CWD differs from the project root (e.g. /workspace/runs/...
    vs /ai-inventor/), skills won't be found. This creates a .claude symlink in the
    agent's CWD pointing to the project root's .claude/.
    """
    cwd = Path(str(options.cwd)).resolve() if options.cwd else None
    if not cwd:
        return

    # Already has .claude/ — nothing to do
    if (cwd / ".claude").exists():
        return

    # Find project root's .claude/
    project_root = os.environ.get("AII_PROJECT_ROOT", "")
    if not project_root:
        return
    project_claude = Path(project_root) / ".claude"
    if not project_claude.is_dir():
        return

    # CWD is already under the project root — SDK will find it naturally
    try:
        cwd.relative_to(Path(project_root).resolve())
        return
    except ValueError:
        pass  # CWD is outside project root — need symlink

    # Create symlink in CWD
    target = cwd / ".claude"
    try:
        cwd.mkdir(parents=True, exist_ok=True)
        target.symlink_to(project_claude.resolve())
    except OSError:
        pass  # Best-effort — don't crash the agent over this


class _TimeRemainingHook:
    """PostToolUse hook injecting time-remaining system messages.

    Implemented as a top-level callable class (not a closure) so the
    hook is pickleable. Closures defined inside a function can't be
    resolved by ``pickle.dumps`` (``Can't get local object
    'foo.<locals>.bar'``), which would break any pickle round-trip of
    the agent graph (subprocess boundary, multiprocessing, snapshot).

    State:
      - ``deadlines``: shared mutable dict the agent updates as it
        progresses (epoch-second deadlines for prompt/agent/container).
      - ``_no_deadlines_warned``: instance flag so the "no deadlines
        set" warning fires at most once per agent.
    """

    BUFFER_MINUTES = 5
    URGENT_THRESHOLD = 15

    def __init__(self, deadlines: dict[str, float | None], task_id: str = "") -> None:
        self.deadlines = deadlines
        self.task_id = task_id
        self._no_deadlines_warned = False

    async def __call__(
        self, hook_input: Any, matched_tool_name: str, hook_context: Any
    ) -> SyncHookJSONOutput:
        now = _time.time()
        remaining_list = [d - now for d in self.deadlines.values() if d is not None]
        if not remaining_list:
            if not self._no_deadlines_warned:
                emit.status_public_warning(
                    "Time-remaining hook active but no deadlines set (prompt/agent/container all None)",
                )
                self._no_deadlines_warned = True
            return {}
        raw_minutes = max(0.0, min(remaining_list)) / 60
        reported_minutes = max(0.0, raw_minutes - self.BUFFER_MINUTES)

        if reported_minutes < self.URGENT_THRESHOLD:
            msg = (
                f"<system-reminder>WARNING: You have {reported_minutes:.1f} minutes "
                f"remaining. Finish what you are doing now — do not start new work. "
                f"Wrap up and produce your final output.</system-reminder>"
            )
        else:
            msg = (
                f"<system-reminder>You have {reported_minutes:.1f} minutes "
                f"remaining to finish all your tasks.</system-reminder>"
            )

        # Emit an agent_hook journal event so consumers (console, OTel,
        # clone) see exactly what the hook injected into the agent's
        # stream. ``text`` IS the system-reminder body verbatim — no
        # paraphrase.
        if self.task_id:
            emit.agent_hook(
                self.task_id,
                hook_type="PostToolUse",
                text=msg,
                extras={"tool": matched_tool_name or ""},
            )

        result: SyncHookJSONOutput = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": msg,
            },
        }
        return result


def install_time_remaining_hook(options: AgentOptions, deadlines: dict[str, float | None]) -> None:
    """Install a PostToolUse hook that injects time-remaining system messages.

    The hook reads from ``deadlines`` (mutable dict with epoch timestamps)
    and reports min(remaining) across prompt/agent/container timeouts.

    Implemented as :class:`_TimeRemainingHook` rather than an inner
    closure so the hook is pickleable — see the class docstring for
    the rationale.
    """
    hook = _TimeRemainingHook(deadlines, task_id=options.run_id or "")
    hook_matcher = HookMatcher(hooks=[hook])
    if options.hooks is None:
        options.hooks = {}
    existing = options.hooks.get("PostToolUse", [])
    options.hooks["PostToolUse"] = [*existing, hook_matcher]
