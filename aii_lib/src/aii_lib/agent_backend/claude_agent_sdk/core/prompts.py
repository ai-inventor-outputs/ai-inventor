"""Prompt templates and builders for Agent retry/continue/force-output flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import AgentOptions

# ── Templates ──────────────────────────────────────────────────────────

FORCE_OUTPUT_GENERAL_TEMPLATE = """STOP. You have reached the maximum number of turns.

Do NOT use any more tools. Finish what you are doing and provide your final output NOW.

Use whatever information you have gathered so far to produce the best response possible.
"""

_EXPECTED_FILES_FEEDBACK_TEMPLATE = """The following required files are missing:

{missing_files}

Create these files now. The task is not complete until all required files exist.

IMPORTANT: When providing your structured output (title, summary, etc.), describe the ARTIFACT you built — NOT the file verification status. Your title and summary must describe what you created, not that you verified files.
"""

_EXPECTED_FILES_NO_PATHS_FEEDBACK_TEMPLATE = """Your structured output did not include the expected file paths.

Issue: {detail}

Expected file fields in `{field}`:
{expected_fields}

You MUST include the `{field}` field in your structured output with ALL expected file paths filled in (as relative paths from your workspace). Also ensure those files actually exist in your workspace.

IMPORTANT: When providing your structured output (title, summary, etc.), describe the ARTIFACT you built — NOT the file verification status. Your title and summary must describe what you created, not that you verified files.
"""

DIAG_PREFIX = "__DIAG__:"

# ── Prompt builders ────────────────────────────────────────────────────

# Reason → (context template with {timeout_val} placeholder)
_CONTINUE_TEMPLATES: dict[str, str] = {
    "message_timeout": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded "
        "the {timeout_val}s message timeout. Each individual operation must complete "
        "within {timeout_val}s. Do NOT mock, skip, or compromise your execution — "
        "still do the real work. Try to make operations run faster if possible. "
        "If a command genuinely takes longer than {timeout_val}s, split it into "
        "sequential parts that each complete within the time limit."
    ),
    "seq_prompt_timeout": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The entire prompt execution exceeded "
        "the {timeout_val}s prompt timeout. The total work for this prompt must complete "
        "within {timeout_val}s. Do NOT mock, skip, or compromise your execution — "
        "still do the real work. Reuse any partial results from the previous attempt. "
        "Try to be more efficient — cut non-essential steps, but do not sacrifice "
        "the quality of the core task."
    ),
    "agent_timeout": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The entire agent run exceeded "
        "the {timeout_val}s agent timeout. This is the final timeout level — "
        "you have {timeout_val}s total. Do NOT mock, skip, or compromise your execution — "
        "still do the real work. Use whatever partial work exists from previous attempts. "
        "Do not start over or repeat completed steps. Focus only on what remains "
        "and produce the required output."
    ),
    "connection_error": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: A transient network/API error occurred "
        "(connection reset, rate limit, or service unavailability). This was not your fault. "
        "Continue exactly where you left off — the connection has been restored."
    ),
    "validation_error": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The output failed schema validation. "
        "Review the required output format carefully and ensure your response matches "
        "the expected JSON schema exactly."
    ),
    "structured_output_missing": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The required structured output was not "
        "produced. Ensure you complete the task AND produce the expected structured output "
        "in the required format."
    ),
    "subscription_error": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The Claude subscription/access was temporarily "
        "unavailable. This was not your fault — access has been restored. Continue exactly "
        "where you left off."
    ),
    "process_error": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The agent subprocess terminated unexpectedly. "
        "This was a transient infrastructure error, not your fault. Continue exactly where "
        "you left off."
    ),
    "agent_failed": (
        "YOUR PREVIOUS SESSION WAS INTERRUPTED: The agent returned an error. "
        "Review the task requirements and try again, using any partial work from "
        "the previous attempt."
    ),
}

_CONTINUE_FALLBACK = (
    "YOUR PREVIOUS SESSION WAS INTERRUPTED due to an unexpected error. Continue where you left off."
)


def build_continue_prompt(
    original_prompt: str,
    failure_reason: str | None,
    options: AgentOptions,
) -> str:
    """Build a context-aware continue prompt based on what caused the retry."""
    # Pick template and fill timeout value if needed
    template = _CONTINUE_TEMPLATES.get(failure_reason or "", _CONTINUE_FALLBACK)
    timeout_map = {
        "message_timeout": options.message_timeout,
        "seq_prompt_timeout": options.seq_prompt_timeout,
        "agent_timeout": options.agent_timeout,
    }
    timeout_val = timeout_map.get(failure_reason or "", "")
    context = template.format(timeout_val=timeout_val) if "{timeout_val}" in template else template

    # Append last messages for context about what was happening before the failure
    last_msgs = _get_last_messages(options)
    if last_msgs:
        context += "\n\nLast messages before interruption:"
        for msg in last_msgs:
            context += f"\n  - {msg}"

    return f"{context}\n\nCONTINUE FOLLOWING THESE INSTRUCTIONS:\n\n{original_prompt}"


def build_force_output_prompt(options: AgentOptions) -> str:
    """Build force output prompt when max_turns is exceeded."""
    return options.force_output_prompt or FORCE_OUTPUT_GENERAL_TEMPLATE


def build_expected_files_feedback(
    missing: list[str],
    expected_files_field: str | None,
    get_expected_fields_fn: Any,
) -> str:
    """Build feedback prompt for missing files retry.

    If missing contains a structured-output diagnostic (prefixed with
    DIAG_PREFIX), uses the no-paths template instead.
    """
    if missing and missing[0].startswith(DIAG_PREFIX):
        detail = missing[0][len(DIAG_PREFIX) :]
        field = expected_files_field or "out_expected_files"
        expected_fields = get_expected_fields_fn()
        return _EXPECTED_FILES_NO_PATHS_FEEDBACK_TEMPLATE.format(
            detail=detail,
            field=field,
            expected_fields=expected_fields,
        )
    return _EXPECTED_FILES_FEEDBACK_TEMPLATE.format(
        missing_files="\n".join(f"- {m}" for m in missing)
    )


def _get_last_messages(options: AgentOptions) -> list[str]:
    """Get last formatted message lines from the live Run for retry context.

    Uses ``options.run_id`` (the runtime task node_id) as the lookup
    key — ``options.agent_context`` is a display name and
    ``find_task`` would always miss it. See the matching fix in
    ``core/retry.py``'s ``build_agent_retry_context``.
    """
    from aii_lib.run import get_current_run

    run = get_current_run()
    if run is None:
        return []
    return run.get_recent_message_text(
        task_id=options.run_id or None,
        n=options.retry_context_messages,
    )
