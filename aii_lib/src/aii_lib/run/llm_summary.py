"""LLM-summary buffer — pre-record summary attachment for the Run bus.

Sits between :meth:`Run._on` and :meth:`Run._record`. When an eligible
typed :class:`BaseMessage` arrives (an ``agent_*`` / ``llm_*`` /
specific ``status_public_*`` event without an existing ``summary``),
the buffer holds it, fires off an LLM summary call via the executor,
and only releases the message to ``_record`` once the summary is
attached. Drain order is FIFO insertion order — a late-completing
summary cannot jump ahead of an earlier in-flight message.

The prompt + eligibility predicate + output scrub live HERE, next to
the buffer that uses them. The chain-walking infrastructure lives in
:mod:`aii_lib.workflows.summarize` (``summarize``); this module
just builds the prompt and consumes the result.

If the entire fallback chain fails (or times out at
:data:`SummaryBuffer.DRAIN_TIMEOUT_S`), the message drains *without*
a summary and the caller-supplied ``on_summary_failed`` callback
fires so the failure is observable (Run wires it to a
``status_public_warning`` whose own ``summary`` is pre-populated to
prevent recursion through the buffer).
"""

from __future__ import annotations

import collections
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Callable

    from .messages import BaseMessage


# ---------------------------------------------------------------------------
# Optional buffer-trace instrumentation (gated on AII_BUFFER_TRACE=1)
# ---------------------------------------------------------------------------

import json as _bt_json
import os as _bt_os
import threading as _bt_threading

_BUFFER_TRACE_ENABLED = _bt_os.environ.get("AII_BUFFER_TRACE") == "1"
_buffer_trace_lock = _bt_threading.Lock()
_buffer_trace_fp = None


def _buffer_trace_init() -> None:
    """Open the trace file lazily on first event."""
    global _buffer_trace_fp
    if _buffer_trace_fp is not None:
        return
    try:
        from ..utils.paths import logs_dir as _logs_dir

        d = _logs_dir("buffer_trace")
    except Exception:
        from pathlib import Path

        d = Path("/tmp/aii_buffer_trace")
    d.mkdir(parents=True, exist_ok=True)
    _buffer_trace_fp = (d / f"{_bt_os.getpid()}.jsonl").open("a", buffering=1)


def _buffer_trace(event: str, **fields: Any) -> None:
    """Emit one JSONL line. No-op if trace is disabled."""
    if not _BUFFER_TRACE_ENABLED:
        return
    fields["event"] = event
    fields["ts"] = time.monotonic()
    fields["thread"] = _bt_threading.current_thread().name
    with _buffer_trace_lock:
        if _buffer_trace_fp is None:
            _buffer_trace_init()
        try:
            _buffer_trace_fp.write(_bt_json.dumps(fields) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt + eligibility (lifted from the old workflows/summarize/sum_msg.py)
# ---------------------------------------------------------------------------

_MAX_MESSAGE_CHARS = 3000
_DEFAULT_TIMEOUT = 10.0  # per-tier wall-clock budget; sum_msg fires per dashboard event

# Types where the dashboard wants a one-line gist beyond agent_*/llm_*.
_LLM_SUMMARY_TYPES = {
    "status_public_info",
    "status_public_error",
    "status_public_progress",
    "status_public_success",
    "status_public_published",
    "run_start",
}

# agent_*/llm_* types that are pure metadata — the FE renders these
# structurally rather than from a one-line gist. Skip to avoid wasting
# LLM calls on rows nobody reads.
#
# ``agent_start`` / ``agent_end`` are structural LLM-call open/close
# brackets carrying only ``task_id`` (text=""), so the LLM has nothing
# to summarize — it either hallucinates a plausible-looking row or
# fails outright. The slim-channel filter drops them outright; this
# skip-set keeps the summary buffer in lockstep.
#
# ``agent_hook`` is SDK infrastructure chatter (permission prompts,
# time-remaining pings) that the slim-channel filter drops — no point
# burning a summary call on a row the FE never sees.
#
# ``agent_message_delta`` is the per-token streaming delta; the slim
# channel drops it and summarizing every token chunk would torch the
# executor for no user-visible gain.
_SKIP_TYPES = {
    "agent_start",
    "agent_end",
    "agent_hook",
    "agent_message_delta",
}

_SYSTEM_PROMPT_TEMPLATE = """\
You summarize AI research pipeline events for a live dashboard feed.

FORMAT (strict):
  "<Noun phrase>: <short detail phrase>"   — when extra detail is meaningful
  "<Noun phrase>"                           — when the noun alone says it all
                                              (no colon, no trailing period)

The NOUN phrase names the artifact/event produced by this message. It is the
subject of the row. Pick a concise, specific noun from the list below or its
close analogue. Start with a capital letter.

NOUN LIBRARY (by module / message type)
  hypo step                 → "Hypothesis draft" | "Hypothesis revision"
  rev_hypo step             → "Hypothesis review"
  upd_hypo step             → "Hypothesis update"
  strat step                → "Strategy" | "Strategy update"
  plan step (per task)      → "Research plan" | "Experiment plan" | "Dataset plan"
                              | "Evaluation plan" | "Proof plan"
  art step (per task)       → "Research" | "Experiment" | "Dataset"
                              | "Evaluation" | "Proof"
  paper_text step           → "Paper draft" | "Paper section"
  rev_paper step            → "Paper review"
  viz / demo / full_paper / repo → "Figure" | "Demo" | "Final paper" | "Repository"

  agent_response            → a noun naming the decision/claim
                              ("Review submitted", "Hypothesis accepts 4/5")
                              — NEVER start with "Agent"/"Response" (those
                              are the pill labels).
  agent_tool_call / agent_tool_result  (REQUIRED format: "<ToolName>: <detail>")
    The slim wire dropped the per-tool field, so every tool message renders
    with a generic [Tool] pill — your prefix carries the type signal.
    Use the canonical tool name verbatim (Bash, Read, Write, Edit, Glob,
    Grep, WebSearch, WebFetch, Task, Agent, Skill, TodoRead, TodoWrite),
    capitalised exactly as listed.

    Bash                    → "Bash: <specific action>"
                              ("Bash: pytest eval module",
                               "Bash: pip install numpy")
    Read                    → "Read: <what's examined>"
                              ("Read: pipeline config")
    Write / Edit            → "Write: <change>" / "Edit: <change>"
                              ("Edit: fix circular import")
    Glob / Grep             → "Glob: <pattern>" / "Grep: <pattern>"
                              ("Grep: *_test.py across src/")
    WebSearch / WebFetch    → "WebSearch: <topic>" / "WebFetch: <topic>"
                              ("WebSearch: ArXiv papers on Lévy flights")
    Task / Agent            → "Task: <subagent activity>" / "Agent: <activity>"
                              ("Task: mathlib lemma search")
    Skill                   → "Skill: <which + why>"
                              ("Skill: Arxiv-fetch for 3 DOIs")
    StructuredOutput        → EXCEPTION: use the DOMAIN noun from input
                              fields, not "StructuredOutput:". The structured
                              payload IS the artifact, so the domain noun is
                              more useful: "Hypothesis draft: ...",
                              "Hypothesis review: ...", "Research plan: ...",
                              "Experiment plan: ...", "Strategy: ...".
                              NEVER "Output:", "Structured output",
                              "StructuredOutput".
  status_public_warning     → cause + recovery     ("Schema retry 2/3 —
                              summary too short", "Auth 401 — 60s retry").
                              NEVER start with "Warning" / "Retry".
  status_public_info        → the fact, numbers first ("Score 6/10 — 4
                              major"). Noun comes from the content, never
                              a meta word.
  status_public_error       → what failed + impact ("Validation failed —
                              missing fields"). NEVER start with "Error".
  agent_config / llm_config → just the setting     ("Opus 4.7 — low effort",
                              "Sonnet 4.5 — review iter 2"). NEVER start
                              with "Config" — that's the pill label.
  agent_system_prompt /     → describe the agent's ROLE/SETUP — what it's
  llm_system_prompt           been told it is and what it has access to.
                              NEVER fabricate an output (don't invent a
                              "Hypothesis review: 6/10 — …" score, don't
                              quote a Strategy, don't draft a Hypothesis).
                              The system prompt is the SETUP, not the
                              result; the result lands in a later
                              ``agent_response`` row. Examples:
                                "Hypothesis-generation agent — Haiku"
                                "Hypothesis-review agent — 4 criteria"
                                "Strategy-generation agent — invention loop iter 2"
                                "Paper-review agent — conference-style"
                              NEVER start with "System" — that's the pill
                              label. Numbers/scores ONLY if they are
                              constants stated in the prompt itself
                              (criteria count, max iterations); never
                              invent an evaluation score.
  run_start                 → ONLY the research topic (no colon, no detail)

HARD RULES
- Length: {min_chars}-{max_chars} characters. Aim for 15-45 total.
- At most ONE colon (the one between noun and detail). Never use colons in
  the detail itself — use an em-dash " — " if you need a secondary separator.
- Plain text only: no JSON, brackets, markdown, code fences.
- Character set: ASCII letters/digits/spaces and basic punctuation only
  (.,;:!?'"-/—). NO emojis, decorative dingbats, fancy quotes, math symbols,
  arrows, or non-Latin scripts. The dashboard renders these as gibberish.
- Never include opaque IDs (toolu_..., UUIDs, session hashes, raw task IDs
  like "t1_hypo_it3__opus"). Describe the artifact instead.
- Numbers as digits. Abbreviate only standard terms (ML, LLM, API, GPU, CPU).
- Minutes rounded to integers. DROP vague time-remaining mentions
  ("113 min left", "X minutes noted") — they're internal budget hints that
  the dashboard doesn't display anywhere.

BANNED LEAD WORDS — absolute rule (non-negotiable)
  The pill label is already displayed before your text. For most events
  the pill spells out the category — your summary must NOT repeat it.

  Banned first words for non-tool events (no case variants):
    "Output", "Input", "Agent", "Response", "Warning", "Retry", "Config",
    "Error", "Tool".

  And banned as the noun in "NOUN: detail":
    Any of the above, with or without the colon.

  EXCEPTION — tool_call / tool_result events: the pill renders generically
  as [Tool] (the slim wire no longer carries the tool name), so the tool
  NAME is the REQUIRED prefix. Start with "Bash:", "Read:", "WebSearch:",
  etc. — see the per-tool table above. StructuredOutput keeps its
  domain-noun convention ("Hypothesis draft: ...") since the structured
  payload IS the artifact.

  INSTEAD: pretend the label word is already there and write what comes AFTER.
    Raw event: agent_config  model=claude-opus-4-7 effort=low
      Pill renders:  [Config] _______
      What fills the blank? "Opus 4.7 — low effort"  ✓
      NOT "Config: Opus 4.7 — low effort"             ✗ (duplicates label)

    Raw event: status_public_warning  schema retry 2/3
      Pill: [Warning] _______
      Fill: "Schema retry 2/3 — summary too short"   ✓
      NOT "Warning: schema retry 2/3"                  ✗

    Raw event: tool_call WebSearch "Lévy flight"
      Pill renders generically:  [Tool] _______
      Fill: "WebSearch: ArXiv papers on Lévy flight optimization"  ✓
      NOT  "ArXiv papers on Lévy flight"                            ✗ (no tool prefix)
      NOT  "Web search: Lévy flight"                                 ✗ (use "WebSearch:" verbatim, not "Web search:")

  EXCEPTION: when the noun is a pipeline-domain artifact that is NOT a label
  word, the colon form is REQUIRED. These domain nouns are allowed before
  the colon: "Hypothesis draft", "Hypothesis revision", "Hypothesis review",
  "Hypothesis update", "Strategy", "Research plan", "Experiment plan",
  "Dataset plan", "Evaluation plan", "Proof plan", "Research", "Experiment",
  "Dataset", "Evaluation", "Proof", "Paper draft", "Paper section",
  "Paper review", "Figure", "Demo", "Final paper", "Repository", "Score".
  Use them with the colon:  "Hypothesis review: 6/10 — 4 major".

BANNED FILLER WORDS — remove and restructure:
  "successfully", "delivered", "provided", "starting", "beginning",
  "completed", "running", "executing", "using", "has been", "was",
  "process", "handling", "trying to", "attempting", "performing",
  "invoking", "initialized", "notes", "noted".

GOOD EXAMPLES (domain noun + colon OR bare detail)
  Hypothesis draft: quorum-sensing gates           ← domain noun ok
  Hypothesis review: 6/10 — 4 major, 3 minor
  Hypothesis revision: 5 critiques addressed
  Research plan: stochastic-resonance sensor array
  Experiment plan: 2×2 factorial on GSM-Hard
  Research: literature review on Lévy flights
  Evaluation: accuracy vs. token-cost curves
  Strategy: 3 mechanistic directions
  Paper draft: intro + related work
  Paper review: 7/10 — novelty solid, stats weak
  Score: 6/10 — 4 major
  Hypothesis submitted                              ← no detail

  For non-tool, non-domain events, NO colon — just the bare detail:
  Opus 4.7 — low effort                             ← [Config] label
  Schema retry 2/3 — summary too short              ← [Warning] label
  Auth 401 — 60s retry                              ← [Warning] label
  Rate limit — waiting for capacity                 ← [Warning] label
  Validation failed — missing strengths, critiques  ← [Error] label
  Review acknowledged                               ← [Agent] label

  For tool_call / tool_result events, REQUIRED format is "<ToolName>: <detail>":
  Bash: pytest eval module                          ← [Tool] pill (generic)
  Bash: pip install numpy
  Read: pipeline config — model selection
  Edit: fix circular import in telemetry init
  Glob: *_test.py across src/
  Grep: TODO across telemetry/
  WebSearch: ArXiv papers on Lévy flight optimization
  WebFetch: arxiv.org/abs/2310.12345
  Task: mathlib lemma search for L_p bounds
  Skill: Arxiv-fetch — 3 DOIs
  Hypothesis draft: quorum-sensing gates            ← StructuredOutput → domain noun
  Hypothesis review: 6/10 — 4 major, 3 minor        ← StructuredOutput → domain noun

BAD EXAMPLES (label echo or missing tool prefix = forbidden)
  Output: schema validation failed     — [Output] is the banned lead word
  Structured output provided           — label echo + filler
  Output does not match required schema — starts with "Output"
  Agent: acknowledged review           — [Agent] is the label
  Agent acknowledged review            — starts with "Agent"
  Config: Opus 4.7 — low effort        — [Config] is the label
  Warning: schema retry 2/3            — [Warning] is the label
  Retry: auth 401 — 60s                — [Warning] label, and "Retry" too
  Error: validation failed             — [Error] is the label
  toolu_01Wsoi3...: hypothesis         — raw id
  Hypothesis: 113 min remaining        — time-hint noise

  Tool-specific bad examples (missing or wrong prefix):
  pytest eval module                   — missing "Bash: " prefix on a Bash call
  Command: pytest eval module          — use "Bash:" not "Command:"
  Bash command: pytest eval module     — drop "command", just "Bash: pytest..."
  File: pipeline.yaml                  — use "Read:" not "File:"
  Web search: Lévy flight              — use "WebSearch:" verbatim, no space
  Tool call: WebSearch foo             — use the tool name directly, not "Tool call:"

Return ONLY the summary text itself. No JSON. No quotes. No curly braces.
No markdown, no code fences, no labels like "Summary:" — just the bare line."""


def _is_eligible_dict(message: dict) -> bool:
    """Eligibility predicate operating on the dict form of a message.

    ``agent_*`` / ``llm_*`` / specific ``status_public_*`` types
    qualify; metadata rows (``agent_summary``) are skipped.
    """
    msg_type = message.get("type", "")
    if msg_type in _SKIP_TYPES:
        return False
    return msg_type.startswith(("agent_", "llm_")) or msg_type in _LLM_SUMMARY_TYPES


def _format_message_for_llm(message: dict, max_chars: int) -> str:
    """Build the user-message body the summarizer LLM sees."""
    msg_type = message.get("type", "unknown")
    text = message.get("text") or ""
    module = message.get("module") or ""
    task_name = message.get("task_name") or ""
    tool = message.get("tool") or ""

    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    parts = [f"type: {msg_type}"]
    if module:
        parts.append(f"module: {module}")
    if task_name:
        parts.append(f"task: {task_name}")
    if tool:
        parts.append(f"tool: {tool}")
    parts.append(f'text: "{text}"')

    tool_input = message.get("input")
    if tool_input:
        parts.append(f'input: "{str(tool_input)[:500]}"')

    tool_output = message.get("output")
    if tool_output:
        parts.append(f'output: "{str(tool_output)[:500]}"')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SummaryBufferConfig:
    """Tunables for the per-message summary buffer."""

    def __init__(
        self,
        min_chars: int = 30,
        max_chars: int = 50,
        max_concurrent: int = 10,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_concurrent = max_concurrent


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class SummaryBuffer:
    """FIFO queue that holds eligible messages until LLM summary lands.

    Owned by ``Run``. ``Run._on`` checks :meth:`is_eligible`; if yes it
    calls :meth:`submit` with ``Run._record`` as the on-ready
    callback. The buffer mutates ``event.summary`` in-place once the
    LLM summary is ready, then forwards to the callback in submission
    order.

    Thread-safety: ``_queue_lock`` protects queue mutation;
    ``_emit_lock`` ensures only one thread drains at a time so a burst
    of completing futures can't reorder the emit sequence.

    Worker self-deadlock guard: if ``_try_drain`` is called from a
    worker thread (name starts with ``llm_summary``), it does NOT
    block waiting for the head — that would deadlock when the
    blocking thread IS the one needed to make the head ready. Workers
    pop ready prefix only.
    """

    DRAIN_TIMEOUT_S: float = 20.0

    def __init__(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        config: SummaryBufferConfig,
        on_summary_failed: Callable[[BaseMessage], None] | None = None,
    ) -> None:
        self._executor = executor
        self._config = config
        self._on_summary_failed = on_summary_failed

        self._queue: collections.deque[
            tuple[threading.Event, Callable[[BaseMessage], None], list, BaseMessage]
        ] = collections.deque()
        self._queue_lock = threading.Lock()
        self._emit_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def is_eligible(self, event: BaseMessage) -> bool:
        """Should ``event`` be held until an LLM summary is attached?

        Messages that already carry a non-empty ``summary`` are NOT
        eligible (replay, a prior buffer pass, or the warning fallback
        has filled it in already).
        """
        if getattr(event, "summary", "") != "":
            return False
        return _is_eligible_dict(event.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Submit + drain
    # ------------------------------------------------------------------

    def submit(
        self,
        event: BaseMessage,
        on_ready: Callable[[BaseMessage], None],
    ) -> None:
        """Enqueue ``event`` and kick off its LLM-summary future.

        ``on_ready`` is called once with ``event`` after the message's
        turn at the queue head AND its summary is attached (or the
        DRAIN_TIMEOUT_S safety valve fires).
        """
        msg_dict = event.model_dump(mode="json")
        # Belt-and-braces: callers should have routed via ``is_eligible``
        # first, but if not we fall back to the same predicate here.
        eligible = _is_eligible_dict(msg_dict)

        ready = threading.Event()
        result_holder: list[Any] = [None]

        if not eligible:
            ready.set()  # passthrough: drains as soon as it's at the head

        with self._queue_lock:
            self._queue.append((ready, on_ready, result_holder, event))
            qlen_after = len(self._queue)

        _buffer_trace(
            "submit",
            msg_id=id(event),
            type=getattr(event, "type", "?"),
            text_len=len(getattr(event, "text", "") or ""),
            eligible=eligible,
            queue_len_after=qlen_after,
        )

        if eligible:
            self._executor.submit(
                self._generate,
                event,
                msg_dict,
                ready,
                result_holder,
            )

        self._try_drain()

    def _generate(
        self,
        event: BaseMessage,
        msg_dict: dict,
        ready: threading.Event,
        result_holder: list,
    ) -> None:
        """Worker-thread body: build the prompt, call the LLM, signal ready.

        The chain (model tiers + per-tier timeout, no within-tier
        retries) is owned by ``summarize``; this method only sees
        a sync function returning ``{"text": ..., ...}``.
        """
        exec_start_t = time.monotonic()
        err_kind: str | None = None
        try:
            from ..workflows.summarize import summarize

            formatted = _format_message_for_llm(msg_dict, _MAX_MESSAGE_CHARS)
            system = _SYSTEM_PROMPT_TEMPLATE.format(
                min_chars=self._config.min_chars,
                max_chars=self._config.max_chars,
            )
            prompt = f"Summarize this pipeline event for a live dashboard feed:\n\n{formatted}"

            try:
                result = summarize(
                    prompt=prompt,
                    system=system,
                    timeout=_DEFAULT_TIMEOUT,
                    reasoning_effort="low",
                )
                summary = (result.get("text") or "").strip()
            except Exception as e:
                err_kind = f"{type(e).__name__}: {str(e)[:80]}"
                summary = ""

            # Truncate to the requested max with an explicit ellipsis so
            # the dashboard renders "..." instead of a silent mid-word clip.
            if summary and len(summary) > self._config.max_chars:
                summary = summary[: self._config.max_chars - 3] + "..."

            if summary:
                result_holder[0] = summary
        except Exception as e:
            err_kind = f"{type(e).__name__}: {str(e)[:80]}"
        finally:
            _buffer_trace(
                "exec_end",
                msg_id=id(event),
                type=getattr(event, "type", "?"),
                exec_duration_s=time.monotonic() - exec_start_t,
                got_summary=bool(result_holder[0]),
                err=err_kind,
            )
            ready.set()
            self._try_drain()

    def _try_drain(self) -> None:
        """Pop and emit the ready prefix from the head.

        If the head isn't ready, wait up to ``DRAIN_TIMEOUT_S`` for it.
        If still not ready after the wait, emit it without a summary
        and fire ``on_summary_failed`` so the failure is observable.

        Workers (thread name ``llm_summary*``) skip the wait — they
        ARE the path that makes the head ready, so blocking deadlocks
        themselves. They drain only the already-ready prefix.
        """
        from_worker = _bt_threading.current_thread().name.startswith("llm_summary")
        self._emit_lock.acquire()
        try:
            to_emit: list[tuple[Callable[[BaseMessage], None], str, BaseMessage]] = []

            with self._queue_lock:
                while self._queue:
                    ready, on_ready, result_holder, event = self._queue[0]
                    if ready.is_set():
                        self._queue.popleft()
                        to_emit.append((on_ready, result_holder[0] or "", event))
                    else:
                        break

                if self._queue and not self._queue[0][0].is_set() and not from_worker:
                    head_ready = self._queue[0][0]
                    self._queue_lock.release()
                    drain_t0 = time.monotonic()
                    try:
                        head_ready.wait(timeout=self.DRAIN_TIMEOUT_S)
                    finally:
                        drain_elapsed = time.monotonic() - drain_t0
                        if drain_elapsed > 15.0:
                            _buffer_trace(
                                "drain_blocked",
                                blocked_s=drain_elapsed,
                                queue_len=len(self._queue),
                                head_ready=head_ready.is_set(),
                            )
                        self._queue_lock.acquire()

                    while self._queue:
                        ready, on_ready, result_holder, event = self._queue[0]
                        if ready.is_set():
                            self._queue.popleft()
                            to_emit.append((on_ready, result_holder[0] or "", event))
                        elif to_emit:
                            break
                        else:
                            self._queue.popleft()
                            to_emit.append((on_ready, "", event))
        finally:
            self._emit_lock.release()

        # Outside both locks: attach summaries, then forward.
        for on_ready, summary_text, event in to_emit:
            if summary_text and hasattr(event, "summary"):
                event.summary = summary_text
            on_ready(event)

            # Only warn when an *eligible* event drained without a summary
            # (LLM call timed out or errored). Non-eligible passthroughs
            # never had a summary requested in the first place — warning on
            # them produces 1:1 noise per status emit.
            if (
                not summary_text
                and self._on_summary_failed is not None
                and _is_eligible_dict(event.model_dump(mode="json"))
            ):
                try:
                    self._on_summary_failed(event)
                except Exception:
                    # Never let the warning path break the main drain.
                    pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Block until every queued summary lands (or its wait elapses), then drain."""
        with self._queue_lock:
            entries = list(self._queue)
        for ready, _, _, _ in entries:
            if not ready.wait(timeout=timeout):
                break
        self._try_drain()


__all__ = ["SummaryBuffer", "SummaryBufferConfig"]
