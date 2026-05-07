#!/usr/bin/env python
"""
OpenRouter Model Search - Search for LLMs in OpenRouter's catalog.

Usage:
    python openrouter_search.py "claude" --limit 5
    python openrouter_search.py "gpt" --series GPT --limit 10
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from aii_lib.abilities.aii_ability import aii_ability

MODELS_URL = "https://openrouter.ai/api/v1/models"
SERVER_NAME = "aii_openrouter_llms__search"
DEFAULT_LIMIT = 10
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


def init_openrouter_search():
    """Initialize OpenRouter search environment and warmup."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({"Authorization": f"Bearer {OPENROUTER_API_KEY}"})

    # Warmup - fetch models list to establish connection
    try:
        _session.get(MODELS_URL, timeout=SESSION_TIMEOUT)
    except Exception:
        pass


@aii_ability(
    name="aii_openrouter_llms__search",
    description="Search for AI models on OpenRouter by name, family, or capability.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_openrouter_search",
    check_env="check_env.sh",
)
def core_openrouter_search(query: str = "", limit: int = 10, series: str = "") -> dict:
    """
    Search for AI models on OpenRouter.

    Args:
        query: Search query to filter models (e.g., 'claude', 'gpt')
        limit: Maximum number of results
        series: Filter by model family (GPT, Claude, Gemini, etc.)

    Returns:
        Dict with success, query, count, results, and formatted output
    """
    global _session

    if not OPENROUTER_API_KEY:
        return {"success": False, "error": "OPENROUTER_API_KEY not set"}

    try:
        response = _session.get(MODELS_URL, timeout=SESSION_TIMEOUT)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}",
            }

        data = response.json()
        models = data.get("data", [])

        # Filter models - support multi-word OR queries
        query_terms = [t.strip().lower() for t in query.split() if t.strip()]
        series_lower = series.lower() if series else ""

        filtered_models = []
        for model in models:
            model_id = model.get("id", "").lower()
            model_name = model.get("name", "").lower()
            searchable = model_id + " " + model_name

            # Match if ANY query term is found (OR logic)
            if query_terms and not any(term in searchable for term in query_terms):
                continue
            if series_lower and series_lower not in searchable:
                continue

            filtered_models.append(model)

        # Sort by created date (newest first) and limit
        filtered_models.sort(key=lambda x: x.get("created", 0), reverse=True)
        filtered_models = filtered_models[:limit]

        # Format as human-readable string
        query_display = query or "(all models)"
        lines = [f"Found {len(filtered_models)} models for query: {query_display}\n"]

        for i, model in enumerate(filtered_models, 1):
            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 0)) * 1000000
            completion_price = float(pricing.get("completion", 0)) * 1000000
            context_len = model.get("context_length", 0)
            desc = model.get("description", "")[:150]
            supported_params = model.get("supported_parameters", [])

            lines.append(f"[{i}] {model.get('name', 'Unknown')}")
            lines.append(f"    API: {model.get('id', '')}")
            lines.append(f"    Context: {context_len:,} tokens")
            lines.append(f"    Price: ${prompt_price:.2f}/M in, ${completion_price:.2f}/M out")
            if supported_params:
                lines.append(f"    Params: {', '.join(supported_params)}")
            if desc:
                lines.append(f"    {desc}...")
            lines.append("")

        return {
            "success": True,
            "query": query,
            "series": series,
            "count": len(filtered_models),
            "results": filtered_models,
            "output": "\n".join(lines),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Search for models on OpenRouter")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--limit", "-n", type=int, default=DEFAULT_LIMIT, help="Max results")
    parser.add_argument("--series", "-s", help="Filter by model family")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "query": args.query,
            "limit": args.limit,
            "series": args.series or "",
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
