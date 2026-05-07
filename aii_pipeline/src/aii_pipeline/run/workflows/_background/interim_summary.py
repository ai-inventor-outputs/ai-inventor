"""``interim_summary_workflow`` — periodic narrative summaries of a run.

Polls the **parent** workflow's journal, generates a 2-paragraph
summary via an LLM, and emits a ``status_public_interim_summary``
event into THIS workflow's journal. The events endpoint merges this
workflow's journal into the parent's timeline (via the
``f"{run_id}-summary"`` id derived from
:func:`.send_message_recv.summary_workflow_id`).

Why a separate workflow: same reason as ``send_message_recv``.
``@DBOS.step`` calls (``collect_new_journal_events_step``,
``generate_interim_summary_step``, ``journal_event_step`` from emit)
inside an ``asyncio.create_task`` sibling of the parent would
interleave function ids with the parent's body and break replay
determinism. Hosting the loop in its own workflow gives it its own
function-id space.

Lifecycle:

  * Parent starts via :func:`dbos.DBOS.start_workflow_async` with the
    deterministic id :func:`.send_message_recv.summary_workflow_id`.

  * Parent's ``finally`` cancels via :func:`dbos.DBOS.cancel_workflow_async`.
    The next ``DBOS.sleep`` or ``@DBOS.step`` call raises
    :class:`DBOSWorkflowCancelledError`; we exit cleanly.
"""

from __future__ import annotations

from aii_lib.run import emit
from dbos import DBOS
from dbos._error import DBOSWorkflowCancelledError
from loguru import logger

from aii_pipeline.run.steps.interim_summary import (
    collect_new_journal_events_step,
    generate_interim_summary_step,
)


@DBOS.workflow()
async def interim_summary_workflow(
    parent_run_id: str,
    cfg_dict: dict,
) -> None:
    """Periodic poll → summarise → emit loop.

    ``cfg_dict`` is :class:`InterimSummaryStepConfig.model_dump()` so
    DBOS can journal the workflow's input cleanly (the dataclass /
    Pydantic config object is reconstructed inside).

    Each iteration:
      1. ``collect_new_journal_events_step`` pages the **parent**'s
         journal for events newer than the local cursor;
      2. if at least ``cfg.min_new_messages`` new events arrived,
         ``generate_interim_summary_step`` calls the LLM;
      3. ``emit.status_public_interim_summary`` writes the summary
         event to **this** workflow's journal (the events endpoint
         union pulls it back into the run's timeline).

    Sleeps ``cfg.interval_s`` between iterations. Cancellation
    propagates through the next ``DBOS.sleep`` / step call and the
    coroutine returns cleanly.
    """
    from aii_pipeline.run.config import InterimSummaryStepConfig

    cfg = InterimSummaryStepConfig.model_validate(cfg_dict)

    previous_summary = ""
    last_cursor_fid = 0

    try:
        await DBOS.sleep_async(cfg.initial_delay_s)
    except DBOSWorkflowCancelledError:
        return

    while True:
        try:
            events_text, new_cursor, n_new = collect_new_journal_events_step(
                workflow_id=parent_run_id,
                after_function_id=last_cursor_fid,
                max_chars_per_msg=cfg.max_chars_per_msg,
            )
            if n_new >= cfg.min_new_messages and events_text:
                summary = await generate_interim_summary_step(
                    events_text=events_text,
                    previous_summary=previous_summary,
                    timeout_s=cfg.timeout_s,
                    reasoning_effort=cfg.reasoning_effort,
                )
                if summary:
                    emit.status_public_interim_summary("", summary=summary)
                    previous_summary = summary
                    last_cursor_fid = new_cursor
        except DBOSWorkflowCancelledError:
            return
        except Exception:
            logger.opt(exception=True).warning("interim_summary_workflow: iteration failed")

        try:
            await DBOS.sleep_async(cfg.interval_s)
        except DBOSWorkflowCancelledError:
            return


__all__ = ["interim_summary_workflow"]
