#!/usr/bin/env python
"""
OWID Table Search Tool

Search for tables in Our World in Data catalog using BM25.

Usage:
    python aii_owid_search_datasets.py "renewable energy" --limit 5
"""

import argparse
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_owid_datasets__search_datasets"
DEFAULT_LIMIT = 3


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_owid_search():
    """Initialize OWID search environment and warmup BM25 index."""
    import os

    os.environ["TQDM_DISABLE"] = "1"

    import bm25s

    # Pre-load index and metadata into memory
    script_dir = Path(__file__).parent
    index_dir = script_dir.parent / "_index"
    metadata_path = index_dir / "table_metadata_filtered.json"

    if index_dir.exists() and metadata_path.exists():
        try:
            # Load BM25 index
            retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
            # Warmup query
            query_tokens = bm25s.tokenize(["test"], stemmer=None, show_progress=False)
            retriever.retrieve(query_tokens, k=1, show_progress=False)
        except Exception:
            pass


@aii_ability(
    name="aii_owid_datasets__search_datasets",
    description="Search Our World in Data catalog for tables using BM25.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_owid_search",
)
def core_owid_search(query: str = "", limit: int = 3) -> dict:
    """
    Search for tables in OWID catalog using BM25.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 3)

    Returns:
        Dict with success status and result string
    """
    import json
    import os

    os.environ["TQDM_DISABLE"] = "1"

    import bm25s

    script_dir = Path(__file__).parent
    index_dir = script_dir.parent / "_index"
    metadata_path = index_dir / "table_metadata_filtered.json"

    if not index_dir.exists() or not metadata_path.exists():
        return {"success": False, "error": f"Index not found at {index_dir}"}

    try:
        # Load index and metadata
        retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
        with open(metadata_path) as f:
            metadata_list = json.load(f)

        # Search
        query_tokens = bm25s.tokenize([query], stemmer=None, show_progress=False)
        results, scores = retriever.retrieve(query_tokens, k=limit, show_progress=False)

        # Format results as human-readable string
        lines = [f"Found {limit} OWID tables for '{query}':\n"]

        for i, (doc_idx, score) in enumerate(zip(results[0], scores[0], strict=False)):
            if doc_idx < len(metadata_list) and score > 0:
                meta = metadata_list[doc_idx]
                table_name = (
                    meta.get("table_title")
                    or meta.get("table_short_name")
                    or meta.get("table", "Unknown")
                )
                path = meta.get("path", "")
                desc = meta.get("table_description") or meta.get("dataset_description") or ""
                if len(desc) > 200:
                    desc = desc[:200] + "..."

                # Variables detail
                variables = meta.get("variables", [])

                lines.append(f"[{i + 1}] {table_name}")
                lines.append(f"    Path: {path}")
                if desc:
                    lines.append(f"    Description: {desc}")
                lines.append(f"    Variables ({len(variables)} total):")

                for v in variables[:20]:
                    var_name = v.get("title") or v.get("name") or "unnamed"
                    var_desc = (
                        v.get("description")
                        or v.get("description_short")
                        or v.get("description_from_producer")
                        or ""
                    )
                    if len(var_desc) > 150:
                        var_desc = var_desc[:150] + "..."
                    var_unit = v.get("unit") or v.get("short_unit") or ""

                    if var_desc and var_unit:
                        lines.append(f"      - {var_name} ({var_unit}): {var_desc}")
                    elif var_unit:
                        lines.append(f"      - {var_name} ({var_unit})")
                    elif var_desc:
                        lines.append(f"      - {var_name}: {var_desc}")
                    else:
                        lines.append(f"      - {var_name}")

                lines.append("")

        return {"success": True, "result": "\n".join(lines)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Search OWID tables using BM25")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of results (default: 3)",
    )
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "query": args.query,
            "limit": args.limit,
        },
        timeout=60.0,
    )

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
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
