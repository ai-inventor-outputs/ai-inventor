"""Title generation as a ``@DBOS.step``.

Eager one-shot step invoked at the top of ``run_pipeline_workflow``.
Inputs are JSON-safe (``prompt_text`` + ``run_dir``); the output is
the derived title (empty string on failure). DBOS journals the
result so workflow replays re-use the cached title without burning
a second LLM round-trip.

Side effect: writes the title to ``<run_dir>/sinks/title/.title``
— the runs-sidebar reader picks it up from there as
``llm_gen_title``.
"""

from __future__ import annotations

from pathlib import Path

from aii_lib.run.sinks.title.sink import generate_title
from dbos import DBOS


@DBOS.step()
async def generate_title_step(prompt_text: str, run_dir: str) -> str:
    """Generate a short title from ``prompt_text`` and persist it.

    Returns the derived title or an empty string on failure. The
    caller sets ``run.name`` from the result and emits a
    :class:`RunTitleMessage` so dispatch's ``_apply_run_title``
    slugifies into ``run.name`` consistently.
    """
    if not prompt_text or not prompt_text.strip():
        return ""
    title = (generate_title(prompt_text) or "").strip().strip("\"'")
    if not title:
        return ""
    title_path = Path(run_dir) / "sinks" / "title" / ".title"
    title_path.parent.mkdir(parents=True, exist_ok=True)
    title_path.write_text(title, encoding="utf-8")
    return title
