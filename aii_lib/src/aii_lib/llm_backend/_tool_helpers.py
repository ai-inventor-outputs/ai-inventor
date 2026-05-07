"""Tool loop helpers — tool execution, abbreviations, schema validation, summary emission."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..abilities.ability_server.ability_client import call_server
from .schema_utils import calculate_tool_costs


async def execute_tool_calls(tool_calls: list[dict], **kwargs) -> list[dict]:
    """Execute tool calls by calling abilities via the ability server."""
    loop = asyncio.get_running_loop()

    async def _call_one(tc: dict) -> dict:
        tool_call_id = tc.get("id")
        name = tc.get("name")
        raw_args = tc.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = raw_args or {}

        try:
            result = await loop.run_in_executor(None, lambda: call_server(name, arguments))
            if result is None:
                return {
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "original_name": name,
                    "result": None,
                    "error": "ability server returned no response",
                    "cache_hit": False,
                }
            return {
                "tool_call_id": tool_call_id,
                "name": name,
                "original_name": name,
                "result": result,
                "error": result.get("error") if isinstance(result, dict) else None,
                "cache_hit": False,
            }
        except Exception as e:
            # Ability-server failures surface to the LLM as a tool error string.
            # Also capture the traceback so operators can investigate via logs.
            from ..telemetry import logger

            logger.exception(f"ability call failed: {name}({arguments}) -> {e}")
            return {
                "tool_call_id": tool_call_id,
                "name": name,
                "original_name": name,
                "result": None,
                "error": str(e),
                "cache_hit": False,
            }

    results = await asyncio.gather(*[_call_one(tc) for tc in tool_calls])
    return list(results)


def _get_tool_abbrev(tool_name: str, suffix: str) -> str:
    """Get abbreviated tool name for display (max 8 chars including suffix)."""
    from ..abilities.endpoint_names import (
        AII_HF_DOWNLOAD,
        AII_HF_PREVIEW,
        AII_HF_SEARCH,
        AII_JSON_FORMAT,
        AII_JSON_VALIDATE,
        AII_LEAN_RUN,
        AII_LEAN_SUGGEST,
        AII_MATHLIB_SEARCH,
        AII_OPENROUTER_CALL,
        AII_OPENROUTER_GET_PARAMS,
        AII_OPENROUTER_SEARCH,
        AII_OWID_DOWNLOAD,
        AII_OWID_SEARCH,
        AII_SEMSCHOLAR_BIB_FETCH,
        AII_WEB_FETCH,
        AII_WEB_FETCH_GREP,
        AII_WEB_SEARCH,
        AII_WEB_VERIFY_QUOTES,
    )

    abbrev_map = {
        AII_WEB_SEARCH: "SRCH",
        AII_WEB_FETCH: "FTCH",
        AII_WEB_FETCH_GREP: "GREP",
        AII_WEB_VERIFY_QUOTES: "VRFY",
        AII_SEMSCHOLAR_BIB_FETCH: "S2BIB",
        AII_HF_SEARCH: "HF_S",
        AII_HF_PREVIEW: "HF_P",
        AII_HF_DOWNLOAD: "HF_D",
        AII_OWID_SEARCH: "OWID",
        AII_OWID_DOWNLOAD: "OWID",
        AII_OPENROUTER_SEARCH: "OR_S",
        AII_OPENROUTER_CALL: "OR_C",
        AII_OPENROUTER_GET_PARAMS: "OR_P",
        AII_LEAN_RUN: "LEAN",
        AII_LEAN_SUGGEST: "LEAN",
        AII_MATHLIB_SEARCH: "MLIB",
        AII_JSON_VALIDATE: "JSON",
        AII_JSON_FORMAT: "JSON",
    }
    if tool_name in abbrev_map:
        return f"{abbrev_map[tool_name]}{suffix}"
    return f"{tool_name[:4].upper()}{suffix}"


def _validate_response_schema(response: Any, schema: type, client: Any) -> tuple[bool, str]:
    """Validate response JSON against Pydantic schema."""
    try:
        finish_reason = (
            client.get_finish_reason(response)
            if hasattr(client, "get_finish_reason")
            else "unknown"
        )
        if finish_reason == "length":
            return False, "Response truncated (hit max_tokens)"

        if hasattr(client, "extract_json_from_response"):
            text = client.extract_json_from_response(response)
        else:
            text = client.extract_text_from_response(response)

        if not text or not text.strip():
            has_reasoning = False
            if hasattr(response, "choices") and response.choices:
                msg = (
                    response.choices[0].message if hasattr(response.choices[0], "message") else None
                )
                if msg:
                    reasoning = getattr(msg, "reasoning", None) or getattr(
                        msg, "reasoning_details", None
                    )
                    has_reasoning = bool(reasoning)
            if has_reasoning:
                return False, "Model produced reasoning but no JSON output"
            return False, "Empty response - no JSON content"

        schema.model_validate_json(text)
        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        error_str = str(e)
        if len(error_str) > 500:
            error_str = error_str[:500] + "..."
        return False, error_str


def _emit_summary(conv_stats: Any, client: Any, *, task_id: str = "", task_name: str = "") -> None:
    """Emit a typed :class:`LlmSummaryMessage` from conversation stats."""
    if not conv_stats:
        return

    from aii_lib.run import get_current_run
    from aii_lib.run.messages import LlmSummaryMessage

    run = get_current_run()
    if run is None:
        return

    tool_costs, tool_cost_total = calculate_tool_costs(conv_stats.tool_calls)
    actual_model = conv_stats.model or getattr(client, "model", "unknown")
    backend = getattr(client, "provider_name", None)

    run._on(
        LlmSummaryMessage(
            task_id=task_id,
            parent_id=task_id,
            backend=backend,
            model=actual_model,
            total_cost=conv_stats.total_cost + tool_cost_total,
            input_tokens=conv_stats.prompt_tokens or 0,
            output_tokens=conv_stats.completion_tokens or 0,
            extras={
                "token_cost": conv_stats.total_cost,
                "tool_cost": tool_cost_total,
                "reasoning_tokens": conv_stats.reasoning_tokens or 0,
                "cache_read_tokens": conv_stats.cached_tokens or 0,
                "num_calls": conv_stats.num_turns or 0,
                "runtime_seconds": conv_stats.get_runtime_minutes() * 60,
                "llm_time_seconds": conv_stats.get_runtime_minutes() * 60,
                "tool_calls": dict(conv_stats.tool_calls or {}),
                "tool_costs": dict(tool_costs or {}),
                "finish_reason": conv_stats.finish_reason,
            },
        )
    )
