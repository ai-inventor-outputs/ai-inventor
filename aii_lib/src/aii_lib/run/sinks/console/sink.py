"""ConsoleRunSink — colorized stdout for the Run bus.

Wraps an internal :class:`TaskSequencer` so parallel-task output
displays sequentially (one task contiguously, then the next), and
applies a per-sink filter so only the events the operator wants to
see reach stdout.

Lifted from the previous ``aii_lib.telemetry.sinks.console.ConsoleSink``
— the colorization + JSON-key tinting + truncation + summary
formatting logic stays intact; the input shape just changes from
``TelemetryMessage`` to typed :class:`BaseMessage` (we ``model_dump``
inside ``_format_and_print``).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from aii_lib.run.sink import RunSink

from ..utils import TaskSequencer
from .colors import (
    SERVER_SOURCE_COLORS,
    Colors,
    colorize_json_keys,
    get_color,
)
from .format import format_json_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from aii_lib.run.messages import BaseMessage


# ---------------------------------------------------------------------------
# Console filter policy
# ---------------------------------------------------------------------------

# Message types always shown when ``log_llm_messages=False``. Anything
# outside this set requires ``log_llm_messages=True`` to reach stdout.
_CONSOLE_ALWAYS_SHOW_TYPES: frozenset[str] = frozenset(
    {
        "module_start",
        "module_end",
        "mdgroup_start",
        "mdgroup_end",
        "iteration_start",
        "iteration_end",
        "status_public_progress",
        "status_public_info",
        "status_private_info",
        "status_public_success",
        "status_public_warning",
        "status_public_error",
        "status_public_interim_summary",
        "status_private_debug",
        "task_start",
        "task_end",
        "agent_user_prompt",
        "agent_system_prompt",
        "llm_user_prompt",
        "llm_system_prompt",
        "agent_retry",
        "llm_retry",
        "agent_schema_error",
        "llm_schema_error",
        "run_start",
        "run_end",
        "module_output",
        "agent_config",
        "agent_summary",
        "llm_summary",
        "server_request",
        "server_event",
        "server_error",
    }
)

# Leaf summary types still go through ``_format_summary_message`` for the
# multi-line per-call breakdown (cost/tokens/tool dicts). The structural
# end events (task_end / module_end / mdgroup_end / iteration_end /
# run_end) carry their summary inline in ``text`` (formatted by the
# dispatcher) and just print as-is.
_SUMMARY_TYPES = frozenset(
    {
        "agent_summary",
        "llm_summary",
    }
)

# Types never truncated — short, important status rows + the end-event
# rollup summaries (which carry the full ``Cost | In-Out | Runtime`` line
# in their ``text`` field).
_NO_TRUNCATE_TYPES = frozenset(
    {
        "status_public_progress",
        "status_public_info",
        "status_private_info",
        "status_public_success",
        "status_public_warning",
        "status_public_error",
        "status_public_interim_summary",
        "status_private_debug",
        "agent_summary",
        "llm_summary",
        "agent_hook",
        "task_start",
        "task_end",
        "module_start",
        "module_end",
        "mdgroup_start",
        "mdgroup_end",
        "iteration_start",
        "iteration_end",
        "run_start",
        "run_end",
        "agent_start",
        "agent_end",
        "server_event",
        "server_error",
    }
)


class ConsoleRunSink(RunSink):
    """Run-bus subscriber: colorized stdout with parallel-task sequencing.

    Construction parameters:

      - ``truncation`` — max chars for content (default ``None`` = no truncation)
      - ``log_llm_messages`` — when ``False``, only the
        :data:`_CONSOLE_ALWAYS_SHOW_TYPES` set reaches stdout.

    The internal :class:`TaskSequencer` reorders parallel-task output
    so each task's stream displays contiguously. Errors
    (``status_public_error``) bypass buffering inside the sequencer.
    """

    def __init__(
        self,
        truncation: int | None = 150,
        log_llm_messages: bool = True,
        include_private_messages: bool = False,
        *,
        sequence_lookup: Callable[[str], int | None] | None = None,
    ) -> None:
        self.truncation = truncation
        self.log_llm_messages = log_llm_messages
        self.include_private_messages = include_private_messages
        self.module_colors: dict[str, str] = {}
        # Ambient scope stack — tracks open structural events so
        # scopeless messages (status_public_*, etc.) can be attributed
        # to the most recent live scope when their wire payload doesn't
        # carry parent_id / task_id. Stack of (event_type, node_id)
        # tuples; structural starts push, ends pop the matching pair.
        self._scope_stack: list[tuple[str, str]] = []
        self._sequencer = TaskSequencer(
            forward=self._format_and_print,
            sequence_lookup=sequence_lookup,
        )

    def _ancestor_chain(self, msg_dict: dict) -> list[str]:
        """Walk the structural lineage for ``msg_dict``.

        Returns names in walk order: the message's owning node first
        (e.g. the module for ``module_start``, the task for an
        ``agent_*`` event), then each ancestor up to the Run.

        Resolution order for the starting node:

        1. ``module_id`` / ``iteration_id`` / ``group_id`` — structural
           events that create or end the named node. The node itself
           heads the chain.
        2. ``task_id`` — task lifecycle + agent_* / llm_* family. The
           task heads the chain.
        3. ``parent_id`` — generic structural pointer.
        4. The console sink's ambient scope stack — for messages with
           none of the above (status_public_info etc. emitted ad-hoc
           inside a scope).
        """
        from aii_lib.run import get_current_run

        run = get_current_run()
        if run is None:
            return []

        start_id = (
            msg_dict.get("module_id")
            or msg_dict.get("iteration_id")
            or msg_dict.get("group_id")
            or msg_dict.get("task_id")
            or msg_dict.get("parent_id")
            or ""
        )
        if not start_id and self._scope_stack:
            start_id = self._scope_stack[-1][1]
        if not start_id:
            return []

        chain: list[str] = []
        seen: set[str] = set()
        node_id = start_id
        while node_id and node_id not in seen:
            seen.add(node_id)
            node = run.find_node(node_id)
            if node is None:
                break
            name = getattr(node, "name", "") or ""
            if name:
                chain.append(name)
            node_id = getattr(node, "parent_id", "") or ""
            if node_id == run.node_id:
                run_name = getattr(run, "name", "") or run.node_id
                chain.append(run_name)
                break
        return chain

    def _track_scope(self, msg_dict: dict) -> None:
        """Push/pop the ambient scope stack on structural events.

        The stack's top is the most recently-opened structural node;
        scopeless messages between events attribute to that scope.
        """
        mtype = msg_dict.get("type") or ""
        if mtype in ("mdgroup_start", "iteration_start", "module_start", "task_start"):
            nid = (
                msg_dict.get("module_id")
                or msg_dict.get("iteration_id")
                or msg_dict.get("group_id")
                or msg_dict.get("task_id")
                or ""
            )
            if nid:
                self._scope_stack.append((mtype, nid))
        elif mtype in ("mdgroup_end", "iteration_end", "module_end", "task_end"):
            start_marker = mtype.replace("_end", "_start")
            for i in range(len(self._scope_stack) - 1, -1, -1):
                if self._scope_stack[i][0] == start_marker:
                    self._scope_stack.pop(i)
                    break

    # ------------------------------------------------------------------
    # RunSink hooks
    # ------------------------------------------------------------------

    def flush(self, event: BaseMessage) -> None:
        """Feed an event into the task sequencer."""
        msg_type = getattr(event, "type", "") or ""
        if not self._should_show(msg_type):
            return
        self._sequencer.feed(event)

    def close(self) -> None:
        """Flush any pending buffered messages."""
        try:
            self._sequencer.flush_pending()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Filter policy
    # ------------------------------------------------------------------

    def _should_show(self, msg_type: str) -> bool:
        if not self.include_private_messages and msg_type.startswith("status_private_"):
            return False
        if self.log_llm_messages:
            return True
        return msg_type in _CONSOLE_ALWAYS_SHOW_TYPES

    # ------------------------------------------------------------------
    # Formatting (lifted from ConsoleSink._format_and_print)
    # ------------------------------------------------------------------

    def _format_and_print(self, event: BaseMessage) -> None:
        msg_dict = event.model_dump(mode="json")
        message_type = msg_dict.get("type", "unknown")
        # Update ambient scope BEFORE chain resolution so the message
        # being printed sees the right "owning scope" — e.g. a
        # module_start finds itself at the top of the stack and reads
        # its own node, not the parent iteration.
        self._track_scope(msg_dict)
        message_text = msg_dict.get("text", "")
        tool_name = msg_dict.get("tool", "") or ""
        # Sub-task events (agent_*, llm_*) no longer carry their own
        # task_name field — the canonical name lives on the Task node
        # only. Look it up by task_id for display purposes.
        agent_context = msg_dict.get("task_name", "") or ""
        if not agent_context:
            tid = msg_dict.get("task_id", "") or ""
            if tid:
                from aii_lib.run import get_current_run

                _run = get_current_run()
                if _run is not None:
                    _task = _run.find_task(tid)
                    if _task is not None and _task.name:
                        agent_context = _task.name
        is_error = msg_dict.get("is_error", False)

        if message_type == "module_output":
            extras = msg_dict.get("extras") or {}
            outputs = msg_dict.get("outputs") or extras.get("outputs") or []
            module_name = msg_dict.get("name") or extras.get("name") or "?"
            message_text = f"{len(outputs)} output(s) from {module_name}"

        if message_type == "run_start":
            extras = msg_dict.get("extras") or {}
            rd = extras.get("aii_prompt") or msg_dict.get("aii_prompt") or ""
            if rd:
                message_text = f"{message_text}\n   Research: {rd[:150]}"
        if message_type == "run_end":
            extras = msg_dict.get("extras") or {}
            status = extras.get("status") or msg_dict.get("status") or ""
            if status:
                message_text = f"{message_text} ({status})"

        if message_type in _SUMMARY_TYPES:
            message_text = self._format_summary_message(msg_dict)

        should_truncate = self.truncation and message_type not in _NO_TRUNCATE_TYPES

        if (tool_name and tool_name.endswith("_OUT")) or message_type in (
            "agent_tool_result",
            "llm_tool_result",
        ):
            message_text = format_json_output(message_text)

        if should_truncate and len(message_text) > self.truncation:
            truncated_amount = len(message_text) - self.truncation
            message_text = (
                message_text[: self.truncation]
                + f"\n... (+{truncated_amount} chars truncated in console)"
            )

        display_label = message_type
        display_name = message_type

        # Color priority: server source → module color → type/tool → suffix default.
        module = msg_dict.get("module", "")
        server_source = msg_dict.get("source", "")

        content_color = Colors.RESET
        if message_type.startswith("server_") and server_source:
            content_color = SERVER_SOURCE_COLORS.get(server_source, Colors.SRV_DEFAULT)
            if message_type == "server_error":
                content_color = Colors.ERROR

        if content_color == Colors.RESET:
            content_color = (
                self.module_colors.get(module.upper(), Colors.RESET) if module else Colors.RESET
            )
        if content_color == Colors.RESET:
            content_color = get_color(display_label)
        if content_color == Colors.RESET:
            content_color = get_color(message_type)
        if content_color == Colors.RESET:
            if display_label.endswith("_IN"):
                content_color = Colors.TOOL_CALL
            elif display_label.endswith("_OUT"):
                content_color = Colors.TOOL_RESULT

        is_task_header = "TASK_HEADER:" in message_text
        clean_content = ""
        if is_task_header:
            clean_content = message_text[message_text.index("TASK_HEADER:") + 12 :]

        if is_error:
            message_text = f"Error: {message_text}"

        if is_task_header:
            if "\n" in clean_content:
                header, prompt = clean_content.split("\n", 1)
                colorized_prompt = colorize_json_keys(prompt, content_color)
                formatted_content = (
                    f"{Colors.YELLOW}{header}{Colors.RESET}\n"
                    f"{content_color}{colorized_prompt}{Colors.RESET}"
                )
            else:
                formatted_content = ""
        else:
            colorized_text = colorize_json_keys(message_text, content_color)
            formatted_content = f"{content_color}{colorized_text}{Colors.RESET}"

        is_log_level = message_type.startswith("status_")
        is_server = message_type.startswith("server_")

        # Walk the parent chain so the line shows the full lineage:
        # message_type | parent | grandparent | … | run | <content>
        # Type stays in WHITE/grey; ancestors in BRIGHT_RED tint.
        ancestor_color = content_color if is_log_level else Colors.BRIGHT_RED
        chain = self._ancestor_chain(msg_dict)
        chain_segments = "".join(f"{ancestor_color}{name}{Colors.RESET}|" for name in chain)

        if is_server and server_source:
            line = (
                f"{Colors.WHITE}{display_name}{Colors.RESET}|"
                f"{content_color}{server_source}{Colors.RESET}|"
                f"{chain_segments}\n"
                f"{formatted_content}\n"
            )
        else:
            line = (
                f"{Colors.WHITE}{display_name}{Colors.RESET}|"
                f"{chain_segments}\n"
                f"{formatted_content}\n"
            )

        # Direct stdout write — this sink's whole purpose is to stream
        # run events to the operator's tmux pane. Using sys.stdout.write
        # instead of print() so ruff's T201 no-print rule won't strip
        # it (regression from 57c33e3f7).
        sys.stdout.write(line)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Summary-row formatter (lifted)
    # ------------------------------------------------------------------

    def _format_summary_message(self, msg_dict: dict) -> str:
        metadata = msg_dict.get("extras") or {}
        data = {**msg_dict, **metadata}

        total_cost = data.get("total_cost", 0.0) or 0.0
        token_cost = data.get("token_cost", total_cost)
        if token_cost is None:
            token_cost = total_cost
        tool_cost = data.get("tool_cost", 0.0) or 0.0

        model = data.get("model", "")
        status = data.get("status", "")
        is_aggregated = data.get("is_aggregated", False)
        is_error = data.get("is_error", False)

        runtime_seconds = data.get("runtime_seconds", 0.0) or 0.0
        llm_time_seconds = data.get("llm_time_seconds", 0.0) or 0.0
        num_calls = data.get("num_calls", 1) or 1

        usage = data.get("usage") or {}
        input_tokens = (
            data.get("input_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or 0
        )
        output_tokens = (
            data.get("output_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        reasoning_tokens = (
            data.get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or usage.get("thoughts_token_count")
            or 0
        )
        cache_write = (
            data.get("cache_write_tokens")
            if data.get("cache_write_tokens") is not None
            else usage.get("cache_creation_input_tokens")
        )
        cache_read = (
            data.get("cache_read_tokens")
            if data.get("cache_read_tokens") is not None
            else usage.get("cache_read_input_tokens") or usage.get("cached_content_token_count")
        )

        tool_calls = data.get("tool_calls", {}) or {}
        tool_costs_dict = data.get("tool_costs", {}) or {}

        lines = [""]

        if tool_cost > 0:
            lines.append(
                f"Total: ${total_cost:.4f} (Tokens: ${token_cost:.4f} + Tools: ${tool_cost:.4f})"
            )
        else:
            lines.append(f"Total: ${total_cost:.4f}")

        status_str = f"{status} (error)" if is_error else status
        if is_aggregated:
            lines.append(f"Model: {model} | Calls: {num_calls}")
        else:
            lines.append(f"Model: {model} | Status: {status_str}")

        runtime_str = self._format_time(runtime_seconds)
        llm_time_str = self._format_time(llm_time_seconds)
        if is_aggregated:
            avg_time = llm_time_seconds / num_calls if num_calls > 0 else 0
            avg_str = self._format_time(avg_time)
            lines.append(f"Runtime: {runtime_str} | LLM Time: {llm_time_str} | Avg/Call: {avg_str}")
        else:
            lines.append(f"Runtime: {runtime_str} | Turns: {num_calls}")
        lines.append("")

        total_in = input_tokens + (cache_write or 0) + (cache_read or 0)
        token_line = f"Ctx In: {total_in:,} | Out: {output_tokens:,}"
        if reasoning_tokens > 0:
            token_line += f" | Reasoning: {reasoning_tokens:,}"
        if token_cost > 0:
            token_line += f" | Cost: ${token_cost:.4f}"
        lines.append(token_line)

        cache_write_str = f"{cache_write:,}" if cache_write is not None else "N/A"
        cache_read_str = f"{cache_read:,}" if cache_read is not None else "N/A"
        uncached_str = f"{input_tokens:,}"
        lines.append(
            f"Uncached: {uncached_str} | "
            f"Cache write: {cache_write_str} | Cache read: {cache_read_str}"
        )

        active_tools = {k: v for k, v in tool_calls.items() if v > 0}
        if active_tools:
            tool_strs = [f"{name}: {count}" for name, count in sorted(active_tools.items())]
            lines.append(f"Tools: {', '.join(tool_strs)}")
        else:
            lines.append("Tools: N/A")

        if tool_costs_dict:
            cost_parts = []
            for tool_name, cost_info in tool_costs_dict.items():
                if isinstance(cost_info, dict):
                    count = cost_info.get("count", 0)
                    unit = cost_info.get("unit", 0)
                    total = cost_info.get("total", 0)
                    cost_parts.append(f"{tool_name}: {count} × ${unit:.4f} = ${total:.4f}")
            if cost_parts:
                lines.append(f"Tool Cost: {', '.join(cost_parts)}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds <= 0:
            return "N/A"
        if seconds >= 60:
            return f"{seconds / 60:.1f}m"
        return f"{seconds:.0f}s"


__all__ = ["ConsoleRunSink"]
