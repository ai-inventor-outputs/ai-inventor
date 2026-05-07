#!/usr/bin/env python
"""
Mathlib Pattern Search Tool

Search Mathlib by type signature and patterns via Loogle API.

Usage:
    python aii_mathlib_pattern_search.py "List.map"
    python aii_mathlib_pattern_search.py '"prime"' --limit 10
    python aii_mathlib_pattern_search.py "List _ → List _"
"""

import argparse
import sys

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_lean__mathlib_pattern_search"
DEFAULT_LIMIT = 10
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50

API_URL = "https://loogle.lean-lang.org/json"

# Session pooling for connection reuse
_session = None


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_mathlib_pattern_search():
    """Initialize and warmup connection."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)

    # Warmup
    try:
        _session.get(API_URL, params={"q": "Nat.add"}, timeout=10)
    except Exception:
        pass


@aii_ability(
    name="aii_lean__mathlib_pattern_search",
    description="Search Mathlib by type signature and patterns via Loogle API.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_mathlib_pattern_search",
    check_env="check_env.sh",
)
def core_mathlib_pattern_search(
    query: str = "",
    limit: int = DEFAULT_LIMIT,
    max_results: int | None = None,
    timeout: int = SESSION_TIMEOUT,
) -> dict:
    """
    Search Mathlib by type signature and patterns via Loogle API.

    Args:
        query: Search query (name, type pattern, or substring in quotes)
        limit: Maximum number of results (default: 10)

    Returns:
        Dict with success status and result string
    """
    global _session

    if max_results is not None:
        limit = max_results

    if not query:
        return {"success": False, "error": "Query is required"}

    try:
        response = _session.get(API_URL, params={"q": query}, timeout=timeout)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}",
            }

        data = response.json()

        if "error" in data:
            return {"success": False, "error": f"Loogle error: {data['error']}"}

        count = data.get("count", 0)
        hits = data.get("hits", [])[:limit]

        if not hits:
            return {"success": True, "result": f"No results found for: {query}"}

        lines = [f"Found {count} results for: {query}\n"]

        for i, hit in enumerate(hits, 1):
            name = hit.get("name", "Unknown")
            type_sig = hit.get("type", "").strip()
            module = hit.get("module", "")
            doc = hit.get("doc", "")

            lines.append(f"[{i}] {name}")
            if module:
                lines.append(f"    Module: {module}")
            if type_sig:
                if len(type_sig) > 120:
                    type_sig = type_sig[:120] + "..."
                lines.append(f"    Type: {type_sig}")
            if doc:
                doc = doc.strip().replace("\n", " ")[:100]
                lines.append(f"    Doc: {doc}...")
            lines.append("")

        return {"success": True, "result": "\n".join(lines)}

    except Exception as e:
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            return {
                "success": False,
                "error": "Timeout - query may be too broad. Try adding a constant name.",
            }
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Search Mathlib with type patterns (Loogle)")
    parser.add_argument("query", help="Search query (name, type pattern, or quoted substring)")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of results (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=30.0,
        help="Request timeout (default: 30)",
    )
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "query": args.query,
            "limit": args.limit,
            "timeout": args.timeout,
        },
        timeout=60.0,
    )

    if result is None:
        print("Error: Ability service not available.", file=sys.stderr)
        sys.exit(1)

    if isinstance(result, dict):
        if result.get("success"):
            print(result.get("result", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
    else:
        print(result)


if __name__ == "__main__":
    main()
