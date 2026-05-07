#!/usr/bin/env python
"""
HuggingFace Dataset Search Tool

Search and discover datasets on HuggingFace Hub with metadata.

Usage:
    python aii_hf_search_datasets.py --query "machine learning" --limit 5
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_hf_datasets__search_datasets"
CONNECTION_TIMEOUT = 180  # seconds

# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global HfApi instance for session reuse
_hf_api = None


def init_search_datasets():
    """Initialize HuggingFace environment for search."""
    global _hf_api
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)

    # Pre-import to cache
    from huggingface_hub import DatasetCard, HfApi

    # Create global HfApi instance for session reuse
    _hf_api = HfApi()

    # Warmup API connection
    try:
        datasets = list(_hf_api.list_datasets(search="test", limit=1))
        if datasets:
            DatasetCard.load(datasets[0].id)
    except Exception:
        pass


def _load_card_text(dataset_id: str) -> str:
    """Load a dataset card's text body (capped at 500 chars). Returns empty
    string if the card is missing or fails to load."""
    from huggingface_hub import DatasetCard

    try:
        card = DatasetCard.load(dataset_id)
        return card.text[:500] if card and card.text else ""
    except Exception:
        return ""


def _check_loadable(api, dataset_id: str) -> dict:
    """Determine whether the agent can actually load this dataset.

    Returns ``{"loadable": bool, "has_loader_script": bool}``:
      * ``has_loader_script`` — repo ships a top-level ``<reponame>.py``
        loader. ``datasets>=3`` refuses to execute these.
      * ``loadable`` — True if the dataset is reachable via *any* path:
        either there's no loader script (native parquet-backed dataset),
        or HF Datasets Server has auto-converted the script output to
        parquet (so ``aii_hf_datasets__download_datasets`` can still
        fetch it via the parquet API).

    Failures (auth, transient) default to ``loadable=True`` to stay
    permissive — the agent will discover the real failure on download.
    """
    has_script = False
    try:
        files = api.list_repo_files(dataset_id, repo_type="dataset")
        # Loader convention: <last-segment-of-id>.py at repo root.
        repo_name = dataset_id.split("/")[-1]
        has_script = f"{repo_name}.py" in files
    except Exception:
        pass

    if not has_script:
        return {"loadable": True, "has_loader_script": False}

    # Has a loader script — check whether HF auto-converted it to parquet.
    import httpx

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = httpx.get(
            "https://datasets-server.huggingface.co/is-valid",
            params={"dataset": dataset_id},
            headers=headers,
            timeout=10.0,
        )
        if r.status_code == 200:
            data = r.json()
            # ``preview`` and ``viewer`` both indicate parquet conversion landed.
            loadable = bool(data.get("preview") or data.get("viewer"))
            return {"loadable": loadable, "has_loader_script": True}
    except Exception:
        pass
    # Couldn't determine — default to permissive so we don't hide datasets
    # the agent could otherwise still fetch via fallbacks.
    return {"loadable": True, "has_loader_script": True}


@aii_ability(
    name="aii_hf_datasets__search_datasets",
    description="Search and discover datasets on HuggingFace Hub with metadata.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_search_datasets",
    check_env="check_env.sh",
)
def core_search_datasets(
    query: str = "", limit: int = 5, tags: str = "", sort: str = "downloads"
) -> dict:
    """
    Search HuggingFace datasets.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 5)
        tags: Comma-separated tags to filter by
        sort: Sort by 'downloads' or 'likes' (default: downloads)

    Returns:
        Dict with success, query, count, and results list
    """
    limit = min(max(1, limit), 100)

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    global _hf_api
    api = _hf_api  # Reuse global session
    try:
        # tags are passed via filter param (tags= was deprecated in huggingface_hub)
        tag_filters = tags.split(",") if tags else None
        datasets = list(
            api.list_datasets(
                search=query,
                sort=sort,
                limit=limit,
                filter=tag_filters,
            )
        )

        # Concurrently check each candidate's loader-script status. Datasets
        # that ship a ``<repo>.py`` loader work only with ``datasets<3`` and
        # break the modern stack — flag them so the agent can deprioritise.
        # We keep them in results (don't drop) because some still work via
        # the Datasets Server parquet API; the flag plus loadable=False
        # warns the caller.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(datasets)))) as ex:
            future_card = {ex.submit(_load_card_text, ds.id): ds for ds in datasets}
            future_loadable = {ex.submit(_check_loadable, api, ds.id): ds for ds in datasets}
            card_results: dict[str, str] = {}
            loadable_results: dict[str, dict] = {}
            for fut in as_completed(future_card):
                ds = future_card[fut]
                try:
                    card_results[ds.id] = fut.result()
                except Exception:
                    card_results[ds.id] = ""
            for fut in as_completed(future_loadable):
                ds = future_loadable[fut]
                try:
                    loadable_results[ds.id] = fut.result()
                except Exception:
                    loadable_results[ds.id] = {"loadable": True, "has_loader_script": False}

        for ds in datasets:
            ld = loadable_results.get(ds.id, {"loadable": True, "has_loader_script": False})
            results.append(
                {
                    "id": ds.id,
                    "downloads": ds.downloads,
                    "likes": ds.likes,
                    "tags": ds.tags[:10] if ds.tags else [],
                    "description": card_results.get(ds.id, ""),
                    # ``has_loader_script``: ships a *.py loader. Won't run
                    # under datasets>=3 directly, but the parquet API may
                    # still serve it — see ``loadable``.
                    "has_loader_script": ld["has_loader_script"],
                    # ``loadable``: True if the dataset can be fetched via
                    # *some* path (parquet API or native datasets-3). Agent
                    # should prefer loadable=True candidates.
                    "loadable": ld["loadable"],
                }
            )

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
    parser = argparse.ArgumentParser(description="Search datasets on HuggingFace Hub")
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--tags", default="", help="Filter by tags (comma-separated)")
    parser.add_argument("--sort", choices=["downloads", "likes"], default="downloads")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "query": args.query,
            "limit": args.limit,
            "tags": args.tags,
            "sort": args.sort,
        },
    )

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.get("success"):
        print(f"Found {result['count']} dataset(s) for query='{result['query']}'")
        for i, ds in enumerate(result.get("results", []), 1):
            print(f"\n{'=' * 60}")
            print(f"Dataset {i}: {ds['id']}")
            print(f"Downloads: {ds.get('downloads', 0):,} | Likes: {ds.get('likes', 0)}")
            if ds.get("description"):
                print(f"Description: {ds['description'][:200]}...")
            if ds.get("tags"):
                print(f"Tags: {', '.join(ds['tags'][:5])}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
