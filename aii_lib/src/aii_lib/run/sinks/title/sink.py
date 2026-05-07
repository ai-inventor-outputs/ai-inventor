"""TitleGeneratorSink — one-shot run-name derivation.

Watches the Run bus. On the first :class:`AgentUserPromptMessage` OR
:class:`LlmUserPromptMessage` to flow through (whichever arrives first,
when ``run.name`` is empty), kicks off the title summarizer on a daemon
thread and emits a :class:`RunTitleMessage` with the result. The
dispatcher copies the text onto ``run.name``, so other sinks (Clone,
ToApp, …) pick it up via the same event.

Best-effort + async: the LLM round-trip never blocks the bus. A failed
or empty title is silently swallowed — the run just keeps an empty
name rather than retrying.

The prompt lives in this module, alongside the sink that uses it. The
chain-walking infrastructure lives in :mod:`aii_lib.workflows.summarize`
(``summarize``).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

from aii_lib.run import emit
from aii_lib.run.messages import (
    AgentUserPromptMessage,
    BaseMessage,
    LlmUserPromptMessage,
    RunTitleMessage,
)
from aii_lib.run.sink import RunSink

if TYPE_CHECKING:
    from pathlib import Path

    from aii_lib.run.run import Run


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Per-tier wall-clock budget. Title generation is on the run-create hot
# path so each tier needs to fail fast; with 4 tiers the worst-case
# walk is ~40s but typical first-tier success is <1s. Bumped from 3s
# to 10s to absorb tail latency on the gpt-oss-120b tiers (Groq p90
# ~10s under load) without prematurely walking past a tier that would
# have succeeded.
_TIMEOUT = 10.0

_TITLE_PROMPT = """Generate a short title (10-25 characters) for this research topic.

RULES:
- Use ONLY letters, numbers, and spaces.
- No punctuation, no quotes, no accents, no JSON, no markdown.
- Short noun phrase describing the core topic.
- Title case (capitalize each word).
- 10-25 characters total.

Examples of GOOD titles:
Levy Flight Methods
Transformer Pruning
Multi Agent Systems
GNN Expressiveness

Examples of BAD titles (DO NOT do this):
"Levy flights" (quotes)
Dependency-Distance (hyphens)
ML/AI Research (slashes)
Topic: Transformers (prefix)
Computational Linguistics Research on Dependency Distance Minimization (too long)
GNN Study (too short)

Output the title text only. Nothing else.
"""


def generate_title(text: str) -> str | None:
    """Generate a short title from ``text``. Returns ``None`` on failure.

    Plain-text call. The model is steered toward a 10-25 char alphanumeric
    title via the prompt, but the function does not enforce or normalize
    anything — whatever the LLM produces (with surrounding whitespace and
    quotes stripped) becomes the title. Walks the project default
    fallback chain (gpt-oss-20b/Groq → gpt-oss-120b/Cerebras → …).

    Public — both :class:`TitleGeneratorSink` (in this module) and the
    aii_server start-run endpoint (which pre-computes the title before
    spawning the pipeline) call this. Sync because both call sites
    are blocking-thread contexts where async would force a per-call
    asyncio loop cold-start.
    """
    log = logger.bind(module="title_gen")
    prompt = f"Generate a title for:\n\n{text[:2000]}"

    try:
        from aii_lib.workflows.summarize import summarize

        result = summarize(
            prompt=prompt,
            system=_TITLE_PROMPT,
            timeout=_TIMEOUT,
            reasoning_effort="high",
        )
        return ((result.get("text") or "").strip().strip("\"'")) or None

    except Exception as e:
        log.warning(f"title_gen failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class TitleGeneratorSink(RunSink):
    """Run-bus subscriber that derives ``run.name`` once."""

    def __init__(self, run: Run, run_dir: Path) -> None:
        self._run = run
        self._run_dir = run_dir
        self._fired = False
        self._lock = threading.Lock()
        emit.status_private_info("TitleGeneratorSink started")

    def flush(self, event: BaseMessage) -> None:
        """Fire title generation on first user prompt; persist on RunTitleMessage."""
        # When a RunTitleMessage flows through (either fresh from
        # ``_generate_and_emit`` below OR replayed from parent's clone
        # log during fork boot), persist the human-readable title to
        # ``sinks/title/.title``. The runs-sidebar reads this file as
        # ``llm_gen_title`` — so forks pick up parent's title without
        # re-firing the LLM, since the same RunTitleMessage that set
        # ``run.name`` during dispatch also lands here.
        if isinstance(event, RunTitleMessage):
            text = (event.text or "").strip()
            if text:
                _persist_title(self._run_dir, text)
            return
        if not isinstance(event, (AgentUserPromptMessage, LlmUserPromptMessage)):
            return
        if self._run.name:
            return
        # Forks inherit parent's title via the parent's ``RunTitleMessage``
        # being replayed through the bus during fork-boot replay. Don't
        # fire fresh title-gen here — it'd race the replayed title and
        # cost a wasted LLM call (worst case: fork ends up with a
        # different title than parent because async title-gen wins).
        if getattr(self._run, "forked_from_run_id", None):
            return
        with self._lock:
            if self._fired:
                return
            self._fired = True

        prompt_text = event.text
        threading.Thread(
            target=_generate_and_emit,
            args=(self._run, self._run_dir, prompt_text),
            name="run-title-gen",
            daemon=True,
        ).start()


def _persist_title(run_dir: Path, title: str) -> None:
    """Write the human-readable title to ``sinks/title/.title``.

    Idempotent for replays — same title means same file content. The
    runs-sidebar reads this file as ``llm_gen_title``; dispatch's
    ``_apply_run_title`` slugifies the same text into ``run.name``.
    """
    title_path = run_dir / "sinks" / "title" / ".title"
    title_path.parent.mkdir(parents=True, exist_ok=True)
    title_path.write_text(title, encoding="utf-8")


def _generate_and_emit(run: Run, run_dir: Path, prompt_text: str) -> None:
    title = (generate_title(prompt_text) or "").strip()
    if not title:
        return
    # Emit the human-readable title — the title sink's flush persists
    # the file from the same RunTitleMessage that flows here, and
    # dispatch's ``_apply_run_title`` slugifies for ``run.name``. One
    # source of truth (the message), no asymmetry between fresh
    # runs and replayed forks.
    run._on(RunTitleMessage(parent_id=run.node_id, text=title))


__all__ = ["TitleGeneratorSink", "generate_title"]
