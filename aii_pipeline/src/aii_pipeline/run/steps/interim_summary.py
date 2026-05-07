"""Periodic interim narrative summary as DBOS steps + an in-workflow loop.

Replaces the legacy ``InterimSummarySink`` Run-bus subscriber. The
loop spawns from ``run_pipeline_workflow`` as an ``asyncio.create_task``
(Python's asyncio inherits the parent's :class:`ContextVar`s, so the
DBOS workflow_id is in scope inside the task — every step call lands
in the parent run's journal). The loop:

  1. waits ``initial_delay_s`` so very-short runs never see a summary;
  2. polls the journal for new events since the last cursor;
  3. fires :func:`generate_interim_summary_step` if at least
     ``min_new_messages`` events have arrived;
  4. emits the result via :func:`emit.status_public_interim_summary`
     so consumer sinks (subscribed to the journal tailer) pick it up.

Cancellation: ``run_pipeline_workflow``'s ``finally`` block cancels
the task and awaits it. Per-iteration exceptions are logged and the
loop continues — a single bad summary doesn't kill the cadence.
"""

from __future__ import annotations

from aii_lib.run.journal import decode_output, query_events
from aii_lib.workflows.summarize import summarize
from dbos import DBOS
from loguru import logger

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You write a 2-paragraph dashboard summary of an AI research pipeline run.

Output format (strict): exactly two paragraphs separated by ONE blank line.
Paragraph 1 = what is happening RIGHT NOW. Paragraph 2 = what has been
COMPLETED so far. If nothing meaningful is completed yet, omit paragraph 2
and the blank line — just output paragraph 1.

NEVER describe what YOU are doing (e.g. "Analyzing pipeline logs to compose…",
"Producing a summary of…", "Here is a narrative of…"). The reader sees the
finished summary directly. Write the summary itself, not a description of
the summarization.

LENGTH (count carefully):
- Paragraph 1: 30-150 chars (~6-30 words).
- Paragraph 2: up to 150 chars (~30 words), or omit it.

STYLE:
- Concise. Short direct sentences. No filler.
- No subject ("I", "the agent", "the pipeline") — describe events directly.
- Specific: include hypothesis names, dataset names, counts, metrics when available.
- Spell out acronyms in full. Technical jargon is fine if spelled out.
- Plain text only: no JSON, no curly braces, no quotes around the output, no
  markdown, no headers, no bullets, no bold/italic, no emojis.

Output ONLY the two paragraphs (or one if nothing is completed). Nothing else.
"""


# ---------------------------------------------------------------------------
# Pure helpers (no side effects — safe to call from anywhere)
# ---------------------------------------------------------------------------


def _format_messages(messages: list[dict], max_chars_per_msg: int) -> str:
    """Format journal events (oldest→newest) into one prompt-ready blob.

    Per-message clipping at ``max_chars_per_msg`` keeps any single
    monster message from dominating; no post-join cap is applied —
    :func:`summarize` truncates per-tier (drops oldest, keeps tail) as
    it walks the chain so each tier sees as much context as its
    window allows.
    """
    lines = []
    for msg in messages:
        msg_type = msg.get("type", "unknown")
        text = msg.get("text") or msg.get("message_text") or ""
        module = msg.get("module") or ""
        ts = msg.get("end_at") or msg.get("ts") or ""

        if len(text) > max_chars_per_msg:
            text = text[:max_chars_per_msg] + "... [truncated]"

        time_str = ts.split("T")[1].split(".")[0] if "T" in str(ts) else ""
        prefix = f"[{time_str}]" if time_str else ""
        mod_prefix = f"[{module}]" if module and module != "PIPELINE" else ""

        lines.append(f"{prefix}{mod_prefix} {msg_type}: {text}")

    return "\n".join(lines)


def _build_prompt(formatted: str, previous_summary: str) -> str:
    """Build the user-message body for the summarize LLM call.

    Includes the previous summary when available so the model can
    refresh-in-place rather than restating.
    """
    if previous_summary:
        return (
            f"Previous summary:\n\n{previous_summary}\n\n"
            f"Latest pipeline logs:\n\n{formatted}\n\n"
            f"Write the updated 2-paragraph summary directly. Keep what hasn't "
            f"changed, refresh what has progressed. Output only the summary."
        )
    return (
        f"Pipeline logs:\n\n{formatted}\n\n"
        f"Write the 2-paragraph summary directly. Output only the summary."
    )


# ---------------------------------------------------------------------------
# DBOS steps
# ---------------------------------------------------------------------------


@DBOS.step()
def collect_new_journal_events_step(
    workflow_id: str,
    after_function_id: int,
    max_chars_per_msg: int,
    limit: int = 1000,
) -> tuple[str, int, int]:
    """Page the journal for events newer than ``after_function_id``.

    Returns ``(formatted_text, new_cursor_function_id, count)``.
    """
    rows = query_events(
        [workflow_id],
        after_ts_ms=0,
        after_function_id=after_function_id,
        limit=limit,
    )
    msg_dicts: list[dict] = []
    for _wf_id, _fid, _ts_ms, raw_output in rows:
        msg = decode_output(raw_output)
        if msg is None:
            continue
        msg_dicts.append(msg.model_dump(mode="json"))
    new_cursor = rows[-1][1] if rows else after_function_id
    formatted = _format_messages(msg_dicts, max_chars_per_msg)
    return formatted, new_cursor, len(msg_dicts)


@DBOS.step()
async def generate_interim_summary_step(
    events_text: str,
    previous_summary: str = "",
    timeout_s: float = 20.0,
    reasoning_effort: str = "medium",
) -> str:
    """Generate a 2-paragraph narrative summary of recent activity.

    DBOS journals the input + output so workflow replays reuse the
    cached summary instead of re-running the LLM.
    """
    if not events_text.strip():
        return ""
    prompt = _build_prompt(events_text, previous_summary)
    try:
        result = summarize(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            timeout=timeout_s,
            reasoning_effort=reasoning_effort,
        )
        return (result.get("text") or "").strip()
    except Exception:
        logger.opt(exception=True).warning("interim_summary: summarize() failed")
        return ""


__all__ = [
    "collect_new_journal_events_step",
    "generate_interim_summary_step",
]
