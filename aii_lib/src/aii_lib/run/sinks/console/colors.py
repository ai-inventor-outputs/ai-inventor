"""ANSI color codes and utilities for ConsoleRunSink output.

Lifted from the now-deleted ``aii_lib.telemetry.utils.colors`` so the
sink owns its own palette + JSON-key tinting helpers.
"""

import re


class Colors:
    """ANSI color code palette for console output."""

    # Basic colors
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    RESET = "\033[0m"

    # Semantic colors
    ERROR = "\033[31m"
    WARNING = "\033[33m"
    SUCCESS = "\033[32m"
    INFO = "\033[34m"
    DEBUG = "\033[36m"

    # Message type colors
    SYSTEM = "\033[38;5;240m"
    ASSISTANT = "\033[38;5;214m"
    THINKING = "\033[38;5;201m"
    TOOL_CALL = "\033[38;5;245m"
    TOOL_RESULT = "\033[38;5;28m"
    SUMMARY = "\033[38;5;81m"
    TASK_IN = "\033[38;5;208m"
    TASK_OUT = "\033[38;5;28m"
    TODO_IN = "\033[38;5;82m"
    TODO_OUT = "\033[38;5;82m"
    READ_OUT = "\033[38;5;130m"
    PROMPT = "\033[38;5;156m"
    S_PROMPT = "\033[38;5;140m"

    # Hooks
    HOOK = "\033[38;5;220m"

    # Mid-stream usage updates (agent_message_delta / llm_message_delta)
    MESSAGE_DELTA = "\033[38;5;213m"  # bright pink

    # Server request sources
    SRV_ABILITIES = "\033[38;5;246m"  # gray
    SRV_DASHBOARD = "\033[38;5;75m"  # blue
    SRV_AUTH = "\033[38;5;177m"  # magenta/purple
    SRV_SSE = "\033[38;5;43m"  # teal/cyan
    SRV_STARTUP = "\033[38;5;114m"  # green
    SRV_DB = "\033[38;5;179m"  # warm yellow
    SRV_STATIC = "\033[38;5;240m"  # dim gray
    SRV_DEFAULT = "\033[38;5;246m"  # gray fallback

    # Module/aggregation summaries
    MOD_SUM = "\033[38;5;81m"
    RUN_SUM = "\033[38;5;81m"

    # JSON key highlighting
    JSON_KEY = "\033[38;5;81m"


# Color map — lowercase keys for case-insensitive lookup
_COLOR_MAP = {
    # Status
    "status_public_progress": "\033[38;5;249m",
    "status_public_info": "\033[38;5;249m",
    "status_private_info": Colors.GRAY,
    "status_public_warning": Colors.WARNING,
    "status_public_error": Colors.ERROR,
    "status_public_success": Colors.SUCCESS,
    "status_private_debug": Colors.DEBUG,
    "status_public_published": Colors.SUMMARY,
    "status_public_interim_summary": Colors.SUMMARY,
    # Agent types
    "agent_response": Colors.ASSISTANT,
    "agent_think": Colors.THINKING,
    "agent_tool_call": Colors.TOOL_CALL,
    "agent_tool_result": Colors.TOOL_RESULT,
    "agent_config": Colors.SYSTEM,
    "agent_system_prompt": Colors.S_PROMPT,
    "agent_user_prompt": Colors.PROMPT,
    "agent_hook": Colors.HOOK,
    "agent_summary": Colors.SUMMARY,
    "agent_retry": Colors.WARNING,
    "agent_schema_error": Colors.ERROR,
    "agent_message_delta": Colors.MESSAGE_DELTA,
    # LLM types
    "llm_response": Colors.ASSISTANT,
    "llm_think": Colors.THINKING,
    "llm_tool_call": Colors.TOOL_CALL,
    "llm_tool_result": Colors.TOOL_RESULT,
    "llm_system": Colors.SYSTEM,
    "llm_system_prompt": Colors.S_PROMPT,
    "llm_user_prompt": Colors.PROMPT,
    "llm_summary": Colors.SUMMARY,
    "llm_retry": Colors.WARNING,
    "llm_schema_error": Colors.ERROR,
    "llm_message_delta": Colors.MESSAGE_DELTA,
    # Module/group lifecycle — ``*_end`` events carry the per-node
    # ``Cost | In-Out | Runtime`` summary text formatted by the dispatcher.
    "module_start": Colors.TASK_IN,
    "module_end": Colors.MOD_SUM,
    "mdgroup_start": Colors.TASK_IN,
    "mdgroup_end": Colors.SUMMARY,
    "iteration_start": Colors.TASK_IN,
    "iteration_end": Colors.SUMMARY,
    "task_start": Colors.TASK_IN,
    "task_end": Colors.SUMMARY,
    "agent_start": Colors.TASK_IN,
    "agent_end": Colors.TASK_OUT,
    "run_start": Colors.INFO,
    "run_end": Colors.RUN_SUM,
    "module_output": Colors.TOOL_RESULT,
    # Server types (default color — source-based override below)
    "server_request": Colors.SRV_DEFAULT,
    "server_event": Colors.SRV_STARTUP,
    "server_error": Colors.ERROR,
}

# Source-based color overrides for server_request / server_event
SERVER_SOURCE_COLORS = {
    "abilities": Colors.SRV_ABILITIES,
    "dashboard": Colors.SRV_DASHBOARD,
    "auth": Colors.SRV_AUTH,
    "sse": Colors.SRV_SSE,
    "startup": Colors.SRV_STARTUP,
    "db": Colors.SRV_DB,
    "dist": Colors.SRV_STATIC,
}


def get_color(message_type: str) -> str:
    """Get color for a message type."""
    return _COLOR_MAP.get(message_type.lower(), Colors.RESET)


def colorize(text: str, color: str) -> str:
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.RESET}"


def colorize_json_keys(text: str, content_color: str) -> str:
    """Colorize JSON keys in text with a distinct color."""
    pattern = r'("[\w_-]+")(\s*:\s*)'

    def replace_key(match: object) -> str:
        key = match.group(1)
        colon_space = match.group(2)
        return f"{Colors.JSON_KEY}{key}{Colors.RESET}{content_color}{colon_space}"

    return re.sub(pattern, replace_key, text)
