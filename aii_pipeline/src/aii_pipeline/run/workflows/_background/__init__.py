"""Background sibling workflows for ``run_pipeline_workflow``.

Each module here defines one ``@DBOS.workflow`` that runs CONCURRENTLY
with the parent pipeline. They are NOT children of the parent — they
have their own ``workflow_uuid`` so their step calls don't interleave
with the parent's step sequence (which would break replay
determinism). Parent starts them via :func:`dbos.DBOS.start_workflow_async`
with deterministic ids derived from its own ``run_id`` (e.g.
``f"{run_id}-recv"``), and cancels them in its ``finally`` block via
:func:`dbos.DBOS.cancel_workflow_async`.

  * :mod:`.send_message_recv` — receives ``DBOS.send_async`` injects
    from the server's ``/send_message`` endpoint and routes prompts
    into the SDK sessions of the target module's child tasks.

  * :mod:`.interim_summary` — periodically summarises the parent's
    journal via an LLM and emits a ``status_public_interim_summary``
    event. Its events live in this workflow's own journal; the FE
    events endpoint queries both the parent's and this workflow's
    journal to render them in the same timeline.
"""

from .interim_summary import interim_summary_workflow
from .send_message_recv import (
    SEND_MESSAGE_TOPIC,
    recv_workflow_id,
    send_message_recv_workflow,
    summary_workflow_id,
)

__all__ = [
    "SEND_MESSAGE_TOPIC",
    "interim_summary_workflow",
    "recv_workflow_id",
    "send_message_recv_workflow",
    "summary_workflow_id",
]
