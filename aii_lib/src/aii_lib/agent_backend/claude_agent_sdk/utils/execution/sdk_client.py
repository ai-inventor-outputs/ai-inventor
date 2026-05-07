"""Low-level streaming executor wrapping ClaudeSDKClient.

Each invocation of :meth:`StreamingExecutor.execute` opens one or more
``ClaudeSDKClient`` instances back-to-back, each handling exactly one
user prompt (initial OR a /send_message injection that arrived during
the previous instance). The executor:

* Opens an instance, calls ``client.query(prompt)`` once with a string,
  drains ``receive_response`` until ``ResultMessage`` arrives.
* Captures the ``ResultMessage`` and emits ONE ``agent_summary`` event at
  instance close, carrying that instance's authoritative cumulative cost
  and tokens (from the SDK).
* If a /send_message injection landed during the run, repeats with a new
  ``ClaudeSDKClient`` that ``resume=``s the previous session_id (plain
  resume, no fork) — preserves the model's conversation context while
  starting a fresh per-instance cost counter. NodeStats sums each
  instance's ``agent_summary`` via :func:`apply_leaf_summary`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    UserMessage,
)
from claude_agent_sdk._errors import ProcessError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class AgentProcessError(Exception):
    """Raised when the Claude agent subprocess terminates unexpectedly.

    This wraps ProcessError to provide a cleaner exception that can be caught
    and retried by the agent retry logic.
    """


class SubscriptionAccessError(Exception):
    """Raised when the Claude subscription/access is unavailable.

    Detected from SDK messages with error="authentication_failed" or
    text matching "does not have access to Claude". The agent should
    poll and wait (like threshold exceeded) rather than burn retries.
    """


_AUTH_CRASH_PATTERNS = [
    "oauth token has expired",
    "authentication_error",
    "failed to authenticate",
    "does not have access to claude",
    "organization does not have access",
]


def _is_auth_crash(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(p in error_text for p in _AUTH_CRASH_PATTERNS)


def _check_stderr_for_auth(stderr_lines: list[str]) -> bool:
    text = " ".join(stderr_lines).lower()
    return any(p in text for p in _AUTH_CRASH_PATTERNS)


def _extract_text_from_user_message(msg: UserMessage) -> str:
    """Return the prompt text from a SDK ``UserMessage`` echo.

    Returns ``""`` for tool-result echoes (whose ``content`` is a list of
    ToolResultBlock entries with no TextBlock).
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)
    return ""


class StreamingExecutor:
    """Low-level executor wrapping ClaudeSDKClient.

    One :meth:`execute` call opens one or more ClaudeSDKClient instances
    sequentially — one per logical user turn (initial prompt + /send_message
    injections). Each instance contributes ONE ``agent_summary`` event;
    NodeStats sums them via the existing dispatcher logic.
    """

    def __init__(self, sdk_options: ClaudeAgentOptions):
        self.sdk_options = sdk_options
        # Pending injection slot — set by /send_message handler via inject().
        # Latest write wins; stored as (prompt, prompt_source).
        self._pending_injection: tuple[str, str] | None = None
        # Set by inject() so the receive loop can short-circuit cleanup
        # without treating the partial ResultMessage as an outer-retry-worthy
        # failure.
        self._deliberate_interrupt: bool = False
        # Set by stop_gracefully() — short-circuits the resume loop after
        # the current turn drains so agent_summary still emits but no new
        # ClaudeSDKClient opens.
        self._stop_requested: bool = False
        # Owning loop for cross-thread interrupt scheduling.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Currently-active client (set during execute, used by inject()).
        self._client: ClaudeSDKClient | None = None

    def inject(self, prompt: str, *, prompt_source: str = "human") -> None:
        """Queue a user message for the next ClaudeSDKClient instance.

        Thread-safe. Triggers ``client.interrupt()`` so the current instance
        wraps up promptly; the executor's main loop catches the
        interrupt-induced ResultMessage and opens a new instance with the
        injected prompt as its first user message.

        Latest call wins — earlier pending injections are overwritten if
        the user injects multiple times before the first interrupt fires.
        """
        self._pending_injection = (prompt, prompt_source)
        self._deliberate_interrupt = True
        client = self._client
        loop = self._loop
        if client is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(client.interrupt(), loop)

    def stop_gracefully(self) -> None:
        """Interrupt the current SDK instance and exit the loop without resuming.

        Like :meth:`inject`, but with no follow-up prompt. Triggers
        ``client.interrupt()`` so the SDK drains its current turn —
        the resulting ``ResultMessage`` still flows through
        ``streaming.py`` and emits the per-instance ``agent_summary``
        Run event. After the drain the executor's main loop sees
        ``_stop_requested`` and returns instead of opening a new
        client.

        Used by the ``/stop`` source's soft path: graceful agent
        teardown with a per-task summary, while the source schedules
        a hard SIGINT a few seconds later as a fallback.
        """
        self._pending_injection = None
        self._stop_requested = True
        self._deliberate_interrupt = True
        client = self._client
        loop = self._loop
        if client is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(client.interrupt(), loop)

    async def execute(self, prompt: str, *, task_id: str = "") -> AsyncGenerator:
        """Run a single user prompt across one or more SDK instances.

        Args:
            prompt: The initial prompt text to send.
            task_id: Run-domain ``Task.node_id``. When non-empty, registers
                this executor in :mod:`active_sessions` so /send_message can
                interrupt + inject a follow-up prompt.

        Yields:
            Messages from the SDK — typed AssistantMessage/UserMessage/
            SystemMessage/ResultMessage/StreamEvent/RateLimitEvent.

        Raises:
            SubscriptionAccessError: auth/access issue detected from the
                ProcessError text or captured stderr.
            AgentProcessError: subprocess terminated unexpectedly for
                non-auth reasons (caller's outer retry loop can retry).
        """
        from aii_lib.agent_backend.claude_agent_sdk import active_sessions
        from aii_lib.run import emit

        self._loop = asyncio.get_running_loop()
        self._pending_injection = None
        self._deliberate_interrupt = False

        current_prompt = prompt
        current_prompt_source = "pipeline"
        current_session_id: str | None = None

        while True:
            stderr_lines: list[str] = []

            # B023: ``_stderr_callback`` captures ``stderr_lines`` from this
            # iteration. The SDK call below consumes the callback and returns
            # before we loop again, so the closure can't outlive its bound list.
            def _stderr_callback(line: str) -> None:
                stderr_lines.append(line)  # noqa: B023

            opts = self.sdk_options
            if opts.stderr is None:
                opts = replace(opts, stderr=_stderr_callback)
            if current_session_id:
                opts = replace(opts, resume=current_session_id, fork_session=False)

            last_result: ResultMessage | None = None

            try:
                async with ClaudeSDKClient(options=opts) as client:
                    self._client = client
                    if task_id:
                        active_sessions.register(
                            task_id,
                            executor=self,
                            client=client,
                            loop=self._loop,
                        )

                    try:
                        await client.query(current_prompt)
                        async for message in client.receive_response():
                            if isinstance(message, ResultMessage):
                                last_result = message
                            elif isinstance(message, UserMessage):
                                # SDK echo of a user-side message (with
                                # replay-user-messages enabled). Real prompts
                                # arrive as TextBlock content; tool results
                                # carry ToolResultBlock and resolve to "".
                                text = _extract_text_from_user_message(message)
                                if text and task_id:
                                    emit.agent_user_prompt(
                                        task_id=task_id,
                                        text=text,
                                        prompt_source=current_prompt_source,
                                    )
                            yield message
                    finally:
                        if task_id:
                            active_sessions.deregister(task_id)
                        self._client = None

            except ProcessError as e:
                if _is_auth_crash(e) or _check_stderr_for_auth(stderr_lines):
                    detail = str(e)
                    if stderr_lines:
                        detail += f" | stderr: {' '.join(stderr_lines[-3:])}"
                    raise SubscriptionAccessError(f"Failed to authenticate. {detail}") from e
                stderr_tail = (
                    f" | stderr: {' '.join(line.strip() for line in stderr_lines[-5:] if line.strip())}"
                    if stderr_lines
                    else ""
                )
                raise AgentProcessError(
                    f"Agent subprocess terminated unexpectedly: {e}{stderr_tail}"
                ) from e
            except GeneratorExit:
                # Generator was closed (e.g., due to timeout or cancellation)
                return

            # No explicit emit here — streaming.py's message_callback path
            # converts each ResultMessage into an ``agent_summary`` Run event
            # automatically (single source of truth). NodeStats's
            # ``apply_leaf_summary`` adds each per-instance summary.
            if last_result is not None and last_result.session_id:
                current_session_id = last_result.session_id

            # Stop requested via stop_gracefully() — the current turn
            # drained + emitted agent_summary; bail without resuming.
            if self._stop_requested:
                return
            # Continue with the next pending injection (if any). Otherwise
            # the original prompt is done — return.
            if self._pending_injection is not None:
                current_prompt, current_prompt_source = self._pending_injection
                self._pending_injection = None
                self._deliberate_interrupt = False
                continue
            return
