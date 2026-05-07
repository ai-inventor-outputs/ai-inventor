#!/usr/bin/env python
"""
OpenRouter LLM Call - Make API calls to LLMs via OpenRouter.

Usage:
    python openrouter_call.py --model "anthropic/claude-haiku-4.5" --input "What is 2+2?"
    python openrouter_call.py --model "openai/o1" --input "Solve this" --reasoning high
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from aii_lib.abilities.aii_ability import aii_ability

API_URL = "https://openrouter.ai/api/v1/responses"
SERVER_NAME = "aii_openrouter_llms__call"
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50

VALID_REASONING_EFFORTS = ["minimal", "low", "medium", "high"]

# OpenRouter routing directives — top-level request keys that aren't part of
# any model's supported_parameters but ARE valid request fields. Don't filter
# these out via the per-model allowlist.
OPENROUTER_ROUTING_KEYS = frozenset({"provider", "route", "transforms", "models"})

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Session pooling for connection reuse
_session = None


def init_openrouter_call():
    """Initialize OpenRouter call environment and warmup."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update(
        {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    )

    # Warmup - fetch models list to establish connection
    try:
        _session.get(MODELS_URL, timeout=SESSION_TIMEOUT)
    except Exception:
        pass


@aii_ability(
    name="aii_openrouter_llms__call",
    description="Call an LLM model via OpenRouter API with reasoning and temperature control.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_openrouter_call",
    check_env="check_env.sh",
)
def core_openrouter_call(
    model: str = "",
    input_text: str | None = None,
    input_json: str | None = None,
    max_tokens: int = 9000,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    instructions: str | None = None,
    web_search_max_results: int | None = None,
    extra_params: dict | None = None,
) -> dict:
    """
    Make an API call to an OpenRouter LLM model.

    Args:
        model: API model name (e.g., 'anthropic/claude-sonnet-4')
        input_text: Simple string prompt
        input_json: Full conversation JSON for multi-turn
        max_tokens: Maximum output tokens
        reasoning_effort: Reasoning level (minimal, low, medium, high)
        temperature: Randomness (0.0-2.0)
        top_p: Nucleus sampling (0.0-1.0)
        instructions: System instructions
        web_search_max_results: Enable web search with max results
        extra_params: JSON string or dict of additional model-specific parameters

    Returns:
        Dict with success, model, response, tokens, and formatted output
    """
    global _session

    api_key = OPENROUTER_API_KEY
    if not api_key:
        return {"success": False, "error": "OPENROUTER_API_KEY not set"}

    if not model:
        return {"success": False, "error": "Model is required"}

    if not input_text and not input_json:
        return {
            "success": False,
            "error": "Either input or input_json must be provided",
        }

    if input_text and input_json:
        return {"success": False, "error": "Cannot use both input and input_json"}

    try:
        payload = {
            "model": model,
            "max_output_tokens": max_tokens,
        }

        if input_json:
            try:
                input_data = json.loads(input_json)
                if instructions:
                    has_system = any(
                        msg.get("role") == "system" for msg in input_data if isinstance(msg, dict)
                    )
                    if not has_system:
                        input_data.insert(
                            0,
                            {
                                "type": "message",
                                "role": "system",
                                "content": [{"type": "input_text", "text": instructions}],
                            },
                        )
                payload["input"] = input_data
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid input JSON: {e!s}"}
        else:
            if instructions:
                payload["input"] = [
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": instructions}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": input_text}],
                    },
                ]
            else:
                payload["input"] = input_text

        if reasoning_effort:
            if reasoning_effort not in VALID_REASONING_EFFORTS:
                return {
                    "success": False,
                    "error": f"Invalid reasoning_effort. Valid: {VALID_REASONING_EFFORTS}",
                }
            payload["reasoning"] = {"effort": reasoning_effort}

        if temperature is not None:
            payload["temperature"] = temperature

        if top_p is not None:
            payload["top_p"] = top_p

        if web_search_max_results is not None:
            payload["plugins"] = [{"id": "web", "max_results": web_search_max_results}]

        # Merge extra_params into payload (for model-specific parameters)
        ignored_params = []
        if extra_params:
            if isinstance(extra_params, str):
                try:
                    extra_params = json.loads(extra_params)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Invalid extra_params JSON: {e!s}",
                    }
            if isinstance(extra_params, dict):
                # Fetch supported params for this model
                supported_params = set()
                try:
                    models_resp = _session.get(MODELS_URL, timeout=10)
                    if models_resp.status_code == 200:
                        for m in models_resp.json().get("data", []):
                            if m.get("id", "").lower() == model.lower():
                                supported_params = set(m.get("supported_parameters", []))
                                break
                except Exception:
                    pass  # If we can't fetch, allow all params through

                for key, value in extra_params.items():
                    if value is not None:
                        if (
                            supported_params
                            and key not in supported_params
                            and key not in OPENROUTER_ROUTING_KEYS
                        ):
                            ignored_params.append(key)
                        else:
                            payload[key] = value

        response = _session.post(API_URL, json=payload, timeout=SESSION_TIMEOUT)

        if response.status_code != 200:
            error_text = response.text[:500]
            # Strip sensitive fields from error messages
            for sensitive in (
                "user_id",
                "api_key",
                "authorization",
                "bearer",
                "key",
                "token",
            ):
                error_text = re.sub(
                    rf'"{sensitive}"\s*:\s*"[^"]*"',
                    f'"{sensitive}":"[REDACTED]"',
                    error_text,
                    flags=re.IGNORECASE,
                )
            return {
                "success": False,
                "error": f"API returned status {response.status_code}: {error_text}",
            }

        result = response.json()

        output_text = ""
        reasoning_text = ""

        # Check top-level output_text first
        if result.get("output_text"):
            output_text = result["output_text"]

        if result.get("output"):
            for item in result["output"]:
                item_type = item.get("type", "")

                # Handle reasoning output (OpenRouter returns summary for reasoning models)
                if item_type == "reasoning":
                    # Check summary array (primary source for reasoning summary)
                    if isinstance(item.get("summary"), list) and item["summary"]:
                        reasoning_text = item["summary"][0].get("text", "")

                # Handle message output
                elif item_type == "message" and "content" in item:
                    if isinstance(item["content"], list) and item["content"]:
                        first_content = item["content"][0]
                        if isinstance(first_content, dict) and "text" in first_content:
                            output_text = first_content["text"]
                    elif isinstance(item["content"], str):
                        output_text = item["content"]

        # Use reasoning as output if no message output
        if not output_text and reasoning_text:
            output_text = reasoning_text

        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Build human-readable output
        lines = []
        if ignored_params:
            lines.append(f"Warning: Ignored unsupported params: {', '.join(ignored_params)}\n")
        lines.append(f"Model: {model}\n")
        if reasoning_text:
            lines.append(f"Reasoning:\n{reasoning_text}\n")
        if not output_text:
            output_text = "No output generated"
        lines.append(f"Response:\n{output_text}\n")
        lines.append(f"Tokens: {input_tokens} in, {output_tokens} out")

        return {
            "success": True,
            "model": model,
            "response": output_text,
            "reasoning": reasoning_text if reasoning_text else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ignored_params": ignored_params if ignored_params else None,
            "output": "\n".join(lines),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Call an LLM via OpenRouter")
    parser.add_argument("--model", "-m", required=True, help="Model API name")
    parser.add_argument("--input", "-i", dest="input_text", help="Input prompt")
    parser.add_argument("--input-json", help="Multi-turn conversation JSON")
    parser.add_argument("--max-tokens", type=int, default=9000, help="Max output tokens")
    parser.add_argument("--reasoning", dest="reasoning_effort", help="Reasoning effort")
    parser.add_argument("--temperature", "-t", type=float, help="Temperature (0.0-2.0)")
    parser.add_argument("--top-p", type=float, help="Top-p sampling")
    parser.add_argument("--instructions", help="System instructions")
    parser.add_argument(
        "--web-search",
        type=int,
        dest="web_search_max_results",
        help="Enable web search",
    )
    parser.add_argument(
        "--params",
        "-p",
        dest="extra_params",
        help='Extra model params as JSON (e.g., \'{"top_k": 50, "seed": 42}\')',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    if not args.input_text and not args.input_json:
        print("Error: Either --input or --input-json is required", file=sys.stderr)
        sys.exit(1)

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "model": args.model,
            "input_text": args.input_text,
            "input_json": args.input_json,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "instructions": args.instructions,
            "web_search_max_results": args.web_search_max_results,
            "extra_params": args.extra_params,
        },
        timeout=args.timeout,
    )

    if result is None:
        print("Error: Ability service not available.", file=sys.stderr)
        sys.exit(1)

    if result.get("success"):
        print(result.get("output", ""))
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
