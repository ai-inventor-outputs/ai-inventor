"""OpenRouter Client — async access to 300+ models. Docs: https://openrouter.ai/models."""

import json
from datetime import UTC, datetime
from typing import Any

import aiohttp
from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

from ..schema_utils import (
    add_additional_properties_false,
    calculate_tool_costs,
    make_all_fields_required,
)

# Retry configuration. Per-instance overridable via the ``max_retries`` ctor
# arg — summarize endpoints opt down to 1 (no retry) so a slow tier doesn't
# block the next fallback in the chain.
MAX_RETRIES = 8  # Increased from 4 for better reliability


class OpenRouterError(Exception):
    """Raised on non-200 OpenRouter API responses or in-body error fields."""


def _log_retry_error(retry_state: Any) -> None:
    """Retry callback that includes the model name from the OpenRouterClient instance."""
    attempt = retry_state.attempt_number
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    try:
        exc = retry_state.outcome.exception()
    except Exception:
        exc = None
    err = str(exc).split("\n")[0] if exc else "Unknown error"
    if not err:
        err = type(exc).__name__  # e.g. "TimeoutError" when str(exc) is empty

    # Extract model + per-instance retry cap from self (first positional arg).
    # AsyncRetrying loses the args binding, so model/cap fall through as "?"
    # in that path — only the decorator-style call exposes them.
    model = "?"
    cap: int | str = MAX_RETRIES
    if retry_state.args:
        client = retry_state.args[0]
        model = getattr(client, "model", "?")
        cap = getattr(client, "_max_retries", MAX_RETRIES)

    # `attempt` is the upcoming attempt number, so the last retry log fires
    # at attempt cap with no further attempts after this one.
    msg = f"OpenRouter retry {model} ({attempt}/{cap}) in {wait:.0f}s: {err}"
    logger.warning(msg)
    from aii_lib.run import emit, get_current_run

    run = get_current_run()
    if run is not None:
        emit.status_public_warning(msg)


from ._stats import ConversationStats
from .or_to_json import extract_json_from_text, extract_output, extract_usage


class OpenRouterClient:
    """Async OpenRouter client for accessing 300+ models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        config_path: str | None = None,
        max_retries: int = MAX_RETRIES,
    ):
        # Load config defaults first
        from ..config import get_openrouter_config, load_config

        if config_path:
            load_config(config_path)

        config = get_openrouter_config()

        # Config defaults, overridden by explicit parameters
        self.model = model or config.get("default_model", "anthropic/claude-sonnet-4")

        # Store API key and timeout for HTTP requests
        # Default timeout is 10 min (600s), but can be overridden
        self._api_key = api_key or config.get("api_key")
        if not self._api_key:
            logger.warning("No OpenRouter API key provided; requests will likely fail")
        self._timeout_ms = int((timeout if timeout is not None else 600.0) * 1000)  # Convert to ms
        self._session: aiohttp.ClientSession | None = None
        # ``max_retries=1`` means a single attempt, no exponential backoff.
        # Used by summarize endpoints so a slow Groq doesn't burn 6+ minutes
        # before the chain's next tier (Cerebras gpt-oss-120b) gets a turn.
        self._max_retries = max(1, int(max_retries))

    @staticmethod
    def _dict_to_obj(data: Any) -> Any:
        """Convert nested dict to object with attribute access."""
        from types import SimpleNamespace

        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = OpenRouterClient._dict_to_obj(value)
            return SimpleNamespace(**data)
        if isinstance(data, list):
            return [OpenRouterClient._dict_to_obj(item) for item in data]
        return data

    async def _send_once(self, payload: dict) -> Any:
        """Single HTTP attempt — no retry. Used as the inner step of :meth:`_send_with_retry`."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Always include usage accounting
        payload["usage"] = {"include": True}

        if self._session is None or self._session.closed:
            # ClientTimeout(total=...) alone does NOT cap DNS/connect/TLS phases.
            # Hot-path callers (summarize chain) set timeout=3s and need that
            # budget honored end-to-end; without per-phase caps a stuck connect
            # can blow 5-10s per tier and bust asyncio.wait_for cancellation.
            t = self._timeout_ms / 1000
            timeout = aiohttp.ClientTimeout(
                total=t,
                connect=t,
                sock_connect=t,
                sock_read=t,
            )
            # Use aiodns + force IPv4. aiohttp's default ThreadedResolver
            # uses loop.run_in_executor(socket.getaddrinfo) and wraps the
            # whole thing in asyncio.shield() — meaning a hung getaddrinfo
            # can't be cancelled by per-request timeouts (observed: tier 1
            # of the summary chain hangs ~20s on the FIRST call from a
            # fresh worker-thread asyncio loop, busting the buffer's
            # wait_for budget). aiodns uses raw UDP sockets directly on
            # the loop — no executor, fully cancellable.
            import socket as _sock

            try:
                resolver = aiohttp.AsyncResolver()
            except Exception:
                resolver = None
            if resolver is not None:
                connector = aiohttp.TCPConnector(
                    family=_sock.AF_INET,
                    resolver=resolver,
                )
            else:
                connector = aiohttp.TCPConnector(family=_sock.AF_INET)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        async with self._session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise OpenRouterError(f"OpenRouter API error {resp.status}: {error_text[:500]}")

            data = await resp.json()

            # Check for error in response body (OpenRouter sometimes returns 200 with error)
            if "error" in data:
                error_info = data.get("error", {})
                error_msg = (
                    error_info.get("message", "Unknown error")
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
                error_code = (
                    error_info.get("code", "unknown") if isinstance(error_info, dict) else "unknown"
                )
                raise OpenRouterError(f"OpenRouter API error (in body): {error_code}: {error_msg}")

            # Convert to object with attribute access (like SDK response)
            return self._dict_to_obj(data)

    async def _send_with_retry(self, payload: dict) -> Any:
        """Send chat request via raw HTTP with usage accounting enabled.

        Honors ``self._max_retries``: ``1`` short-circuits the retry loop
        (single attempt, no backoff) for low-latency callers like the
        summarize chain. Otherwise uses tenacity with exponential jitter
        (2s, 4s, 8s, ... up to 120s, capped at ``max_retries`` attempts).

        Uses aiohttp instead of OpenRouter SDK to support usage: {include: true}.
        See: https://openrouter.ai/docs/guides/usage-accounting
        """
        if self._max_retries <= 1:
            return await self._send_once(payload)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=2, max=120, jitter=5),
            before_sleep=_log_retry_error,
            reraise=True,
        ):
            with attempt:
                return await self._send_once(payload)
        # Unreachable: ``reraise=True`` raises on exhaustion. ruff can't infer.
        raise RuntimeError("AsyncRetrying loop exited without raising or returning")

    async def call(
        self,
        prompt: str | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type | dict | None = None,
        provider: dict | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        emit_summary: bool = True,
        emit_system: bool = True,
        conversation_stats: ConversationStats | None = None,
        *,
        task_id: str = "",
        task_name: str = "",
    ) -> Any:
        """Send a chat message (async, no streaming).

        Each emit lands on the active Run bus via
        ``current_run()._on(typed_msg)``; pass ``task_id`` / ``task_name``
        so messages carry their identity.

        Args:
            prompt: The user prompt (ignored if messages is provided)
            system: System prompt (ignored if messages is provided)
            model: Override default model (e.g., "anthropic/claude-4.5-sonnet")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Pydantic model class or JSON schema dict for structured output
            provider: Provider preferences (e.g., {"sort": "price"})
            reasoning_effort: Reasoning effort level (low/medium/high)
            tools: List of tool definitions in OpenAI format
            messages: Full message history (for multi-turn conversations)
            emit_summary: Whether to emit summary message at end
            emit_system: Whether to emit config/system message on first turn
            conversation_stats: Stats object to aggregate across turns
            task_id: Task identity stamped onto every emitted Run-bus message.
            task_name: Display name stamped alongside task_id.

        Returns:
            The API response object
        """
        from aii_lib.run import get_current_run

        from .openrouter_llm_tel_adapter import adapt as _adapt

        def _emit(raw: dict) -> None:
            run = get_current_run()
            if run is None:
                return
            run._on(_adapt(raw, task_id, task_name))

        resolved_model = model or self.model

        # Build messages - use provided messages or construct from prompt/system
        if messages is not None:
            chat_messages = messages
        else:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            chat_messages = []
            if system:
                chat_messages.append({"role": "system", "content": system})
            chat_messages.append({"role": "user", "content": prompt})

            # Log system prompt and user prompt (only when not using raw messages)
            if system:
                _emit({"type": "s_prompt", "text": system, "backend": "openrouter"})
            _emit({"type": "prompt", "text": prompt, "backend": "openrouter"})

        # Emit config message (only on first turn for multi-turn conversations)
        if emit_system:
            system_text = f"{resolved_model}"
            if reasoning_effort:
                system_text += f" | Reasoning: {reasoning_effort}"
            tool_names = []
            if tools:
                tool_names = [t.get("function", {}).get("name") for t in tools]
                system_text += f" | Tools: {', '.join(tool_names)}"
            if response_format:
                system_text += " | Structured output: enabled"

            _emit(
                {
                    "type": "system",
                    "text": system_text,
                    "model": resolved_model,
                    "backend": "openrouter",
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": getattr(response_format, "__name__", None)
                    if response_format
                    else None,
                    "tools": tool_names,
                }
            )

        # Build payload for raw HTTP request
        kwargs = {
            "model": resolved_model,
            "messages": chat_messages,
            "stream": False,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if provider:
            kwargs["provider"] = provider

        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # Add tools (tool_choice defaults to "auto")
        if tools:
            # Filter out internal fields like _original_name that might cause API errors
            clean_tools = []
            for tool in tools:
                clean_tool = {k: v for k, v in tool.items() if not k.startswith("_")}
                clean_tools.append(clean_tool)
            kwargs["tools"] = clean_tools

        # Add structured output format
        if response_format:
            if hasattr(response_format, "model_json_schema"):
                # Pydantic model - convert to JSON schema
                schema = response_format.model_json_schema()
                # OpenAI strict mode requires: additionalProperties=false, all fields required
                schema = add_additional_properties_false(schema)
                schema = make_all_fields_required(schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }
            elif isinstance(response_format, dict):
                # Already a dict schema
                kwargs["response_format"] = response_format

        # Track start time for runtime calculation
        start_time = datetime.now(UTC)

        # Make async request with connection retry (pass payload dict directly)
        response = await self._send_with_retry(kwargs)

        # Calculate runtime
        runtime_minutes = (datetime.now(UTC) - start_time).total_seconds() / 60.0

        # Log response content
        self._log_response(response, _emit)

        # Extract usage, update stats, emit summary
        return self._process_response(
            response,
            resolved_model,
            runtime_minutes,
            _emit,
            emit_summary,
            conversation_stats,
        )

    def _log_response(self, response: Any, _emit: Any) -> None:
        """Log response content (reasoning, content, tool calls) via _emit."""
        if not hasattr(response, "choices") or not response.choices:
            return
        for choice in response.choices:
            if not hasattr(choice, "message") or not choice.message:
                continue
            msg = choice.message

            # Reasoning (GPT-5 style direct string)
            reasoning = getattr(msg, "reasoning", None)
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                _emit(
                    {
                        "type": "or_reasoning",
                        "text": reasoning.strip(),
                        "backend": "openrouter",
                        "extras": {"reasoning_field": True},
                    }
                )

            # Reasoning details (array format)
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details:
                text = "".join(
                    (
                        getattr(d, "content", "") or d.get("content", "")
                        if isinstance(d, dict)
                        else ""
                    )
                    + "\n"
                    for d in reasoning_details
                ).strip()
                if text:
                    _emit(
                        {
                            "type": "or_reasoning",
                            "text": text,
                            "backend": "openrouter",
                            "extras": {"reasoning_details": True},
                        }
                    )

            # Refusal
            refusal = getattr(msg, "refusal", None)
            if refusal:
                _emit({"type": "or_refusal", "text": refusal, "backend": "openrouter"})

            # Content
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            has_other_output = bool(reasoning) or bool(tool_calls)

            if content and content.strip():
                display = content
                if content.strip().startswith("{"):
                    try:
                        display = json.dumps(json.loads(content), indent=2)
                    except json.JSONDecodeError:
                        pass
                _emit(
                    {
                        "type": "or_msg",
                        "text": display,
                        "backend": "openrouter",
                        "extras": {
                            "raw_api_response": choice.model_dump()
                            if hasattr(choice, "model_dump")
                            else None
                        },
                    }
                )
            elif not has_other_output:
                _emit(
                    {
                        "type": "or_msg",
                        "text": "(empty response)",
                        "backend": "openrouter",
                        "extras": {
                            "raw_api_response": choice.model_dump()
                            if hasattr(choice, "model_dump")
                            else None
                        },
                    }
                )

            # Tool calls
            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name if hasattr(tc, "function") else "unknown"
                    args = tc.function.arguments if hasattr(tc, "function") else "{}"
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    _emit(
                        {
                            "type": "or_tool_in",
                            "text": f"Tool: {name}\nArgs: {json.dumps(args, indent=2) if isinstance(args, dict) else args}",
                            "backend": "openrouter",
                            "tool": name,
                            "tool_id": getattr(tc, "id", None) or "",
                            "input": args,
                            "extras": {
                                "raw_api_response": tc.model_dump()
                                if hasattr(tc, "model_dump")
                                else None
                            },
                        }
                    )

    def _process_response(
        self,
        response: Any,
        resolved_model: str,
        runtime_minutes: float,
        _emit: Any,
        emit_summary: bool,
        conversation_stats: ConversationStats | None,
    ) -> Any:
        """Extract usage, update stats, emit summary. Returns response."""
        usage = extract_usage(response)
        total_cost = usage.get("cost", 0.0)

        finish_reason = "unknown"
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown") or "unknown"

        actual_model = getattr(response, "model", resolved_model) or resolved_model
        turn_tool_calls = self.extract_tool_calls(response) if self.has_tool_calls(response) else []

        if conversation_stats is not None:
            conversation_stats.add_turn(usage, total_cost, turn_tool_calls)
            conversation_stats.last_response = response
            conversation_stats.model = actual_model
            conversation_stats.finish_reason = finish_reason

        if emit_summary:
            if conversation_stats is not None:
                stats = conversation_stats
                s_input = stats.prompt_tokens
                s_output = stats.completion_tokens
                s_reasoning = stats.reasoning_tokens or 0
                s_cache_read = stats.cached_tokens or 0
                s_cost = stats.total_cost
                s_runtime = stats.get_runtime_minutes() * 60
                s_turns = stats.num_turns
                s_tool_calls = stats.tool_calls
            else:
                s_input = usage.get("prompt_tokens", 0)
                s_output = usage.get("completion_tokens", 0)
                s_reasoning = usage.get("reasoning_tokens", 0)
                s_cache_read = usage.get("cached_tokens", 0)
                s_cost = total_cost
                s_runtime = runtime_minutes * 60
                s_turns = 1
                s_tool_calls = {}
                for tc in turn_tool_calls:
                    n = tc.get("name", "unknown")
                    s_tool_calls[n] = s_tool_calls.get(n, 0) + 1

            tool_costs, tool_cost_total = calculate_tool_costs(s_tool_calls)
            _emit(
                {
                    "type": "summary",
                    "total_cost": s_cost + tool_cost_total,
                    "token_cost": s_cost,
                    "tool_cost": tool_cost_total,
                    "model": actual_model,
                    "finish_reason": finish_reason,
                    "num_calls": s_turns,
                    "runtime_seconds": s_runtime,
                    "llm_time_seconds": s_runtime,
                    "input_tokens": s_input,
                    "output_tokens": s_output,
                    "reasoning_tokens": s_reasoning,
                    "cache_write_tokens": 0,
                    "cache_read_tokens": s_cache_read,
                    "tool_calls": s_tool_calls,
                    "tool_costs": tool_costs,
                    "backend": "openrouter",
                }
            )

        return response

    def extract_text_from_response(self, response: Any) -> str:
        """Extract output text from response."""
        return extract_output(response)

    def extract_tool_calls(self, response: Any) -> list[dict]:
        """Extract tool calls from response with keys: id, name, arguments."""
        tool_calls = []
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments if hasattr(tc, "function") else "{}"
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", None),
                            "name": tc.function.name if hasattr(tc, "function") else "unknown",
                            "arguments": args,
                        }
                    )
        return tool_calls

    def has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls."""
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            return bool(getattr(msg, "tool_calls", None))
        return False

    def get_finish_reason(self, response: Any) -> str:
        """Get finish reason from response."""
        if hasattr(response, "choices") and response.choices:
            return getattr(response.choices[0], "finish_reason", "unknown") or "unknown"
        return "unknown"

    def extract_json_from_response(self, response: Any) -> str:
        """Extract JSON from response, handling markdown code blocks.

        Some models (e.g., haiku) wrap JSON in ```json ... ``` blocks.
        This method extracts the raw JSON for parsing.
        """
        text = extract_output(response)
        return extract_json_from_text(text)

    @staticmethod
    def resolve_model(model: str, suffix: str = "") -> str:
        """Resolve model name with optional OpenRouter suffix.

        OpenRouter supports routing suffixes like :nitro (fast) and :floor (cheapest).
        See: https://openrouter.ai/docs/model-routing

        Args:
            model: Base model name (e.g., "openai/gpt-5-mini")
            suffix: Optional routing suffix (e.g., "nitro", "floor")

        Returns:
            Model string with suffix if provided (e.g., "openai/gpt-5-mini:nitro")
        """
        return f"{model}:{suffix}" if suffix else model

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        timeout_per_call: int = 180,
        *,
        task_id: str = "",
        task_name: str = "",
    ) -> bytes | None:
        """Generate an image using OpenRouter. Delegates to _image module."""
        from ._image import generate_image as _generate_image

        return await _generate_image(
            api_key=self._api_key,
            model=model or self.model,
            prompt=prompt,
            system=system,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            timeout_per_call=timeout_per_call,
            task_id=task_id,
            task_name=task_name,
        )

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
