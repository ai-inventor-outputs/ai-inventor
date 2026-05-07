"""Run context — ContextVars for the live Run aggregate and per-node ctx.

Two ContextVars live here:

  * :data:`_current_run` — the single live Run for the duration of
    execution. Set once at pipeline entry; read anywhere via
    :func:`current_run` so call sites don't have to thread Run through
    every function signature.

  * :data:`_ctx` — the active per-node execution context, narrowing as
    the pipeline descends the run tree. ``run_pipeline`` pushes a
    top-level ``StepContext``; each phase's ``execute`` builds and
    pushes its own narrower phase ctx (e.g. ``LoopCtx``) for substep
    children to read; same recursion at substep level. Read via
    :func:`current_ctx`, write via :func:`ctx_scope` (auto-pop on exit).

Mirrors the existing ``aii_lib.telemetry.context`` pattern for the
telemetry singleton.

Usage::

    from aii_lib.run.context import set_current_run, current_run

    # At pipeline entry:
    run = Run(node_id=run_id)
    set_current_run(run)

    # Anywhere in the pipeline (state-mutating + journal write via emit.*):
    from aii_lib.run import emit
    tid = emit.start_task(name="gen_strat_it1__opus_001", parent_module_id=mid)
    emit.end_task(tid, status="done")

    # Per-node context flow:
    with ctx_scope(step_ctx):
        for phase in run.children:
            await phase.execute()  # phase pushes its own narrower ctx
                                   # via ctx_scope inside execute()

The chained-syntax handles (``current_task(id).start(...)`` etc.) were
removed when the project consolidated all state mutation onto Run's
flat methods.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .run import Run


_current_run: ContextVar[Run | None] = ContextVar("aii_run", default=None)


_ctx: ContextVar[Any] = ContextVar("aii_ctx", default=None)
"""Active per-node execution context.

Narrows as the pipeline descends the run tree:

  * ``run_pipeline`` pushes a top-level ``StepContext`` (config +
    run_dir). Phase results live on the run tree as
    ``MdGroup.output`` and are read via ``Run.find_group_by_name``,
    not threaded through this ctx.
  * Each phase's ``execute`` reads parent's ctx via :func:`current_ctx`,
    builds its own typed phase ctx (``LoopCtx``, ``GenPaperCtx``, ...)
    via the phase's ``get_context`` method, and pushes it via
    :func:`ctx_scope` for the duration of the phase body.
  * Same recursion at substep level: each substep's ``execute`` pushes
    its own narrower typed substep ctx for downstream consumers.

The ContextVar is type-erased (``Any``); call sites annotate the read
type at the destructuring site (``parent: LoopCtx = current_ctx()``)."""


def set_current_run(run: Run) -> None:
    """Bind the live Run aggregate as the process-wide current run.

    Call once at pipeline start.
    """
    _current_run.set(run)


def current_run() -> Run:
    """Return the current live Run aggregate.

    Raises ``LookupError`` if no run has been set — that's a programmer
    error (calling pipeline code before the Run is established).
    """
    run = _current_run.get()
    if run is None:
        raise LookupError(
            "No current Run set. Call set_current_run(run) at pipeline "
            "entry before calling pipeline-side mutation methods."
        )
    return run


def get_current_run() -> Run | None:
    """Return the current live Run, or ``None`` if not set.

    Use this for code paths where the absence of a Run is acceptable
    (e.g. cli warm-up before run entry, or test code).
    """
    return _current_run.get()


def current_ctx() -> Any:
    """Active execution context for the current node.

    Narrows as you descend the run tree. At the pipeline top-level it
    returns the ``StepContext`` pushed by ``run_pipeline``; inside a
    phase's ``execute`` it returns the phase's typed ctx (e.g. ``LoopCtx``);
    inside a substep's ``execute`` it returns the substep's typed ctx —
    provided each layer pushed its own narrower ctx via :func:`ctx_scope`.

    Returns ``None`` when nothing has been pushed (typical for tests
    that exercise tree assembly without running the pipeline). Mirrors
    :func:`current_run` for the per-node execution layer.
    """
    return _ctx.get()


@contextmanager
def ctx_scope(value: Any):
    """Push value as the active context for the duration of the block.

    Restored to the previous value on exit (normal or exception). Inside
    ``with ctx_scope(x):`` every call to :func:`current_ctx` returns ``x``.
    Nested ``ctx_scope`` calls stack — each push is paired with a pop on
    exit via the stdlib ContextVar token mechanism, so phase / substep ctxs
    unwind cleanly when the pipeline returns up the tree.
    """
    token = _ctx.set(value)
    try:
        yield value
    finally:
        _ctx.reset(token)


__all__ = [
    "ctx_scope",
    "current_ctx",
    "current_run",
    "get_current_run",
    "set_current_run",
]
