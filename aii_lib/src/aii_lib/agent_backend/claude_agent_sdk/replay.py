"""Replay synthesis — reconstruct an :class:`AgentResponse` from a recorded Task.

Stage 4 of the v27 replay-execute architecture. When ``Run._playback_mode``
is ``"replay"`` and ``Agent.run`` is invoked against a Task whose status is
already terminal, we don't dispatch the SDK — we synthesize the response
from what was recorded during the original run:

  - ``structured_output``  ← ``task.output.model_dump()`` (typed payload
    the original substep emitted via ``module_output`` / ``task_output``)
  - ``failed``             ← ``task.status == NodeStatus.FAILED``
  - ``error_message``      ← last ``AgentEndMessage.text`` on the task
  - ``final_response``     ← concatenation of recorded
    ``AgentResponseMessage`` text in original order
  - ``expected_files_valid`` ← ``True`` (deferred — see §0.1.1 audit)

The audit at REPLAY_EXECUTE_AUDIT_0_1.md confirmed every consumer of
``AgentResponse`` can be satisfied from these four primary fields plus
the ``expected_files_valid`` default. Production substeps mostly read
``structured_output`` (typed parse) + ``failed`` (fail-fast gate);
``final_response`` is doc-string only and ``error_message`` log-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.run.node import NodeStatus

from .models.responses import AgentResponse

if TYPE_CHECKING:
    from aii_lib.run.task import Task


def synthesize_agent_response_from_task(task: Task) -> AgentResponse:
    """Build an :class:`AgentResponse` from a fully-recorded Task.

    Returns a response that mirrors what ``Agent.run`` would have
    returned during the original execution. Caller decides whether
    the synthesized response is appropriate (replay mode + Task is
    already terminal) — this function does NOT itself enforce mode
    or status preconditions.

    Edge cases:
    - Task has no ``output`` set → ``structured_output=None``, but
      ``failed=False`` if status is DONE (caller will get an empty
      structured response, same as the original execution would
      have produced).
    - Task is FAILED → ``failed=True``, ``error_message`` from the
      most recent ``agent_end`` event's text.
    - Task has no recorded ``agent_response`` events → ``final_response=""``;
      this is fine because the only production-code consumer of
      ``final_response`` is doc-strings (verified by §0.1 audit).
    """
    structured = None
    raw_output = getattr(task, "output", None)
    if raw_output is not None:
        if hasattr(raw_output, "model_dump"):
            structured = raw_output.model_dump()
        elif isinstance(raw_output, dict):
            structured = raw_output

    response_chunks: list[str] = []
    error_message: str | None = None
    expected_files_valid = True

    for msg in task.messages or []:
        msg_type = getattr(msg, "type", None)
        if msg_type == "agent_response":
            txt = getattr(msg, "text", None)
            if txt:
                response_chunks.append(txt)
        elif msg_type == "agent_end":
            err = getattr(msg, "text", None)
            if err:
                error_message = err
            efv = getattr(msg, "expected_files_valid", None)
            if efv is not None:
                expected_files_valid = bool(efv)

    failed = task.status == NodeStatus.FAILED
    final_response = "\n".join(response_chunks)

    return AgentResponse(
        final_response=final_response,
        structured_output=structured,
        expected_files_valid=expected_files_valid,
        failed=failed,
        error_message=error_message if failed else None,
    )
