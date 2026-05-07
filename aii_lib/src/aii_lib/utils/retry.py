"""``make_retry_log`` — tenacity ``before_sleep`` callback that surfaces retries through the live Run's status events."""

from __future__ import annotations


def make_retry_log(max_retries: int | None = None, label: str | None = None) -> object:
    """Tenacity ``before_sleep`` callback that logs retries via the live Run.

    Used by retry decorators on agent / LLM call sites — emits a
    ``status_public_warning`` when a tier fails so the dashboard /
    console surfaces "Retrying X (n/m) in Ws: <error>".
    """

    def _before_sleep(retry_state: object) -> None:
        attempt = retry_state.attempt_number
        wait = retry_state.next_action.sleep if retry_state.next_action else 0

        total = max_retries
        if total is None:
            stop = getattr(retry_state.retry_object, "stop", None)
            total = getattr(stop, "max_attempt_number", "?")

        try:
            exc = retry_state.outcome.exception()
        except Exception:
            exc = None
        err = str(exc).split("\n")[0] if exc else "Unknown error"

        name = label or (getattr(retry_state.fn, "__name__", "?") if retry_state.fn else "?")

        from aii_lib.run import emit, get_current_run

        run = get_current_run()
        if run is not None:
            emit.status_public_warning(f"Retrying {name} ({attempt}/{total}) in {wait:.0f}s: {err}")

    return _before_sleep
