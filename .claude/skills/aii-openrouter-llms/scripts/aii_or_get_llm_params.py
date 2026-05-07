#!/usr/bin/env python
"""
OpenRouter Get LLM Params - Get supported parameters for a specific model.

Usage:
    python openrouter_get_llm_params.py "anthropic/claude-haiku-4.5"
    python openrouter_get_llm_params.py "openai/o1"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from aii_lib.abilities.aii_ability import aii_ability

MODELS_URL = "https://openrouter.ai/api/v1/models"
SERVER_NAME = "aii_openrouter_llms__get_params"
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

# Session pooling for connection reuse
_session = None


def init_openrouter_get_params():
    """Initialize OpenRouter params lookup."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({"Authorization": f"Bearer {OPENROUTER_API_KEY}"})

    # Warmup
    try:
        _session.get(MODELS_URL, timeout=SESSION_TIMEOUT)
    except Exception:
        pass


@aii_ability(
    name="aii_openrouter_llms__get_params",
    description="Get supported parameters for a specific OpenRouter model.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_openrouter_get_params",
    check_env="check_env.sh",
)
def core_openrouter_get_params(model: str = "") -> dict:
    """
    Get supported parameters for a specific OpenRouter model.

    Args:
        model: Model API name (e.g., 'anthropic/claude-haiku-4.5')

    Returns:
        Dict with success, model info, and formatted output
    """
    global _session

    if not model:
        return {"success": False, "error": "model parameter is required"}

    try:
        response = _session.get(MODELS_URL, timeout=SESSION_TIMEOUT)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}",
            }

        data = response.json()
        models = data.get("data", [])

        # Find the model (exact match or partial match)
        model_lower = model.lower()
        found_model = None

        for m in models:
            model_id = m.get("id", "").lower()
            if model_id == model_lower:
                found_model = m
                break
            # Partial match fallback
            if model_lower in model_id and found_model is None:
                found_model = m

        if not found_model:
            return {"success": False, "error": f"Model '{model}' not found"}

        # Extract all info
        name = found_model.get("name", "Unknown")
        model_id = found_model.get("id", "")
        context_len = found_model.get("context_length", 0)
        pricing = found_model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0)) * 1000000
        completion_price = float(pricing.get("completion", 0)) * 1000000

        supported_params = found_model.get("supported_parameters", [])
        default_params = found_model.get("default_parameters", {})

        architecture = found_model.get("architecture", {})
        modality = architecture.get("modality", "")
        input_modalities = architecture.get("input_modalities", [])
        output_modalities = architecture.get("output_modalities", [])

        top_provider = found_model.get("top_provider", {})
        max_completion = top_provider.get("max_completion_tokens", 0)
        is_moderated = top_provider.get("is_moderated", False)

        # Format output
        lines = [
            f"Model: {name}",
            f"API: {model_id}",
            "",
            "=== Capabilities ===",
            f"Context Length: {context_len:,} tokens",
            f"Max Output: {max_completion:,} tokens"
            if max_completion
            else "Max Output: (not specified)",
            f"Modality: {modality}" if modality else "",
            f"Input: {', '.join(input_modalities)}" if input_modalities else "",
            f"Output: {', '.join(output_modalities)}" if output_modalities else "",
            f"Moderated: {'Yes' if is_moderated else 'No'}",
            "",
            "=== Pricing ===",
            f"Input: ${prompt_price:.4f}/M tokens",
            f"Output: ${completion_price:.4f}/M tokens",
            "",
            "=== Supported Parameters ===",
        ]

        if supported_params:
            for param in sorted(supported_params):
                default_val = default_params.get(param)
                if default_val is not None:
                    lines.append(f"  - {param} (default: {default_val})")
                else:
                    lines.append(f"  - {param}")
        else:
            lines.append("  (none listed)")

        # Filter empty lines
        lines = [ln for ln in lines if ln or ln == ""]

        return {
            "success": True,
            "model": model_id,
            "name": name,
            "context_length": context_len,
            "max_completion_tokens": max_completion,
            "supported_parameters": supported_params,
            "default_parameters": default_params,
            "output": "\n".join(lines),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Get supported parameters for an OpenRouter model")
    parser.add_argument("model", help="Model API name (e.g., anthropic/claude-haiku-4.5)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "model": args.model,
        },
        timeout=args.timeout,
    )

    if result is None:
        # Fall back to direct call if server not available
        result = core_openrouter_get_params(model=args.model)

    if result.get("success"):
        print(result.get("output", ""))
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
