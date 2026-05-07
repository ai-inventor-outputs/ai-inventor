#!/usr/bin/env python
"""
Fast Web Search Tool

Search the web using Serper.dev (Google API).

Usage:
    python aii_fast_web_search.py --query "machine learning papers 2024"
    python aii_fast_web_search.py --query "AI news" --max-results 5 --type news
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_web_tools__search"
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# Session pooling for connection reuse
_session = None


def init_web_search():
    """Initialize web search environment with warmup request."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"})

    # Warmup
    try:
        _session.post(
            "https://google.serper.dev/search",
            json={"q": "warmup", "num": 1},
            timeout=10,
        )
    except Exception:
        pass


@aii_ability(
    name="aii_web_tools__search",
    description="Search the web using Serper.dev (Google API).",
    check_env="check_env.sh",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_web_search",
)
def core_web_search(query: str = "", max_results: int = 10) -> dict:
    """
    Search the web using Serper.dev API.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10, max: 100)

    Returns:
        Dict with success, query, count, and results list
    """
    global _session

    if not query or not query.strip():
        return {
            "success": False,
            "error": "Search query is required (got empty or missing 'query' parameter)",
        }

    if len(query) > 2000:
        return {
            "success": False,
            "error": f"Query too long ({len(query)} chars, max 2000)",
        }

    max_results = min(max(1, max_results), 100)

    if not SERPER_API_KEY:
        return {"success": False, "error": "SERPER_API_KEY not set"}

    try:
        resp = _session.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            timeout=SESSION_TIMEOUT,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}"}
        data = resp.json()
        organic = data.get("organic", [])
        results = [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in organic[:max_results]
        ]
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Search the web using Serper.dev")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max-results", "-n", type=int, default=10)
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "query": args.query,
            "max_results": args.max_results,
        },
        timeout=DEFAULT_TIMEOUT,
    )

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.get("success"):
        print(f"Search: {result['query']}")
        print(f"Found: {result['count']} results\n")
        for i, r in enumerate(result.get("results", []), 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['link']}")
            if r.get("snippet"):
                print(f"   {r['snippet'][:200]}...")
            print()
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
