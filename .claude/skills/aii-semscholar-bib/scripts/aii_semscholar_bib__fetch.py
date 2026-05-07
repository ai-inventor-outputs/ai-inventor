#!/usr/bin/env python
"""
Semantic Scholar Bibliography Builder

Batch-build a .bib file from a list of references using the Semantic Scholar API.
Each reference can have: doi, arxiv, title, author, year.

Phase 1: refs with DOI/ArXiv → POST /paper/batch (single call, up to 500)
Phase 2: title-only refs → GET /paper/search/match (1s delay between)
Post-process: fix entry type, fix citation key, inject DOI

Usage:
    python aii_semscholar_bib__fetch.py --refs '[{"doi": "10.xxx"}, {"title": "Attention", "author": "Vaswani", "year": 2017}]'
"""

import argparse
import json
import re
import sys
import time

import requests
from aii_lib.abilities.aii_ability import aii_ability
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

SERVER_NAME = "aii_semscholar_bib__fetch"
DEFAULT_TIMEOUT = 300.0
SESSION_TIMEOUT = 30
POOL_CONNECTIONS = 10
POOL_MAXSIZE = 10

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_URL = f"{S2_API_BASE}/paper/batch"
S2_MATCH_URL = f"{S2_API_BASE}/paper/search/match"
S2_FIELDS = "citationStyles,externalIds,title,authors,year,venue,publicationTypes,journal"
S2_MATCH_DELAY = 1.0  # seconds between match requests (rate limit)

# Session pooling
_session: requests.Session | None = None


class _S2RateLimitError(Exception):
    """Raised when Semantic Scholar returns 429."""


def _s2_request_with_retry(
    method: str,
    url: str,
    max_retries: int = 5,
    **kwargs,
) -> requests.Response:
    """Make an S2 HTTP request with retry on 429."""
    global _session
    if _session is None:
        init_semscholar_bib()

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        retry=retry_if_exception_type(_S2RateLimitError),
        reraise=True,
    )
    def _request():
        if method == "GET":
            response = _session.get(url, timeout=SESSION_TIMEOUT, **kwargs)
        else:
            response = _session.post(url, timeout=SESSION_TIMEOUT, **kwargs)
        if response.status_code == 429:
            raise _S2RateLimitError("S2 rate limited")
        return response

    try:
        return _request()
    except _S2RateLimitError:
        # Return last response on exhaustion (caller handles 429)
        if method == "GET":
            return _session.get(url, timeout=SESSION_TIMEOUT, **kwargs)
        return _session.post(url, timeout=SESSION_TIMEOUT, **kwargs)


# =============================================================================
# BibTeX post-processing
# =============================================================================


def _fix_entry_type(bibtex: str) -> str:
    """Fix entry type: CoRR/arXiv → @article, has booktitle → @inproceedings."""
    if not bibtex:
        return bibtex

    # Check venue from BibTeX content
    venue_match = re.search(r"(?:journal|booktitle)\s*=\s*\{([^}]*)\}", bibtex, re.IGNORECASE)
    venue = venue_match.group(1).lower() if venue_match else ""

    if "corr" in venue or "arxiv" in venue:
        bibtex = re.sub(r"@\w+\{", "@article{", bibtex, count=1)
    elif re.search(r"booktitle\s*=", bibtex, re.IGNORECASE):
        bibtex = re.sub(r"@\w+\{", "@inproceedings{", bibtex, count=1)

    return bibtex


def _fix_citation_key(bibtex: str, author: str = "", year: int | None = None) -> str:
    """Replace citation key with AuthorYYYY format."""
    if not bibtex:
        return bibtex

    # Try to extract author from BibTeX if not provided
    if not author:
        bib_author = re.search(r"author\s*=\s*\{([^}]+)\}", bibtex)
        if bib_author:
            first_author = bib_author.group(1).split(" and ")[0].strip()
            if "," in first_author:
                author = first_author.split(",")[0].strip()
            else:
                author = first_author.split()[-1].strip()

    # Try to extract year from BibTeX if not provided
    if not year:
        yr_match = re.search(r"year\s*=\s*\{?(\d{4})\}?", bibtex)
        if yr_match:
            year = int(yr_match.group(1))

    if author and year:
        # Clean author name: keep only alpha chars
        clean_author = re.sub(r"[^A-Za-z]", "", author)
        if clean_author:
            new_key = f"{clean_author}{year}"
            bibtex = re.sub(r"@(\w+)\{([^,]+),", rf"@\1{{{new_key},", bibtex, count=1)

    return bibtex


def _inject_doi(bibtex: str, doi: str) -> str:
    """Add DOI field if not already present."""
    if not bibtex or not doi:
        return bibtex

    if re.search(r"doi\s*=", bibtex, re.IGNORECASE):
        return bibtex

    # Insert DOI before closing brace
    bibtex = bibtex.rstrip()
    if bibtex.endswith("}"):
        bibtex = bibtex[:-1].rstrip()
    bibtex += f",\n  doi = {{{doi}}}\n}}"
    return bibtex


def _process_paper(paper: dict, ref: dict) -> dict | None:
    """Process a single S2 paper result into a bib entry.

    Args:
        paper: S2 API paper result dict.
        ref: Original reference dict from input.

    Returns:
        Dict with citation_key, bibtex, title, doi, or ``None`` when the
        paper has no BibTeX representation.
    """
    bibtex = (paper.get("citationStyles") or {}).get("bibtex", "")
    if not bibtex:
        return None

    ext_ids = paper.get("externalIds") or {}
    doi = ext_ids.get("DOI", "") or ref.get("doi", "")
    arxiv_id = ext_ids.get("ArXiv", "") or ref.get("arxiv", "")

    # Get author name for citation key
    authors = paper.get("authors") or []
    first_author = authors[0].get("name", "").split()[-1] if authors else ref.get("author", "")
    year = paper.get("year") or ref.get("year")

    # Post-process
    bibtex = _fix_entry_type(bibtex)
    bibtex = _fix_citation_key(bibtex, author=first_author, year=year)
    bibtex = _inject_doi(bibtex, doi)

    # Extract final citation key
    key_match = re.search(r"@\w+\{([^,]+),", bibtex)
    citation_key = key_match.group(1) if key_match else ""

    return {
        "citation_key": citation_key,
        "bibtex": bibtex.strip(),
        "title": paper.get("title", ""),
        "doi": doi,
        "arxiv": arxiv_id,
    }


# =============================================================================
# Core Logic
# =============================================================================


def init_semscholar_bib() -> None:
    """Initialize Semantic Scholar session with connection pooling."""
    global _session
    from requests.adapters import HTTPAdapter

    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update(
        {
            "User-Agent": "aii-semscholar-tool/1.0",
            "Accept": "application/json",
        }
    )

    # Warmup
    try:
        _session.get(
            S2_MATCH_URL,
            params={"query": "warmup", "fields": "title"},
            timeout=10,
        )
    except Exception:
        pass

    logger.info("Semantic Scholar tools initialized")


@aii_ability(
    name="aii_semscholar_bib__fetch",
    description="Batch-build a .bib file from a list of references using Semantic Scholar API.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_semscholar_bib",
    max_workers=1,
)
def core_semscholar_bib_fetch(references: list | None = None) -> dict:
    """
    Batch-build a .bib file from a list of references.

    Args:
        references: List of dicts, each with optional keys:
            - doi: DOI string (e.g. "10.1234/...")
            - arxiv: ArXiv ID (e.g. "2305.14325")
            - title: Paper title
            - author: Author name (for citation key)
            - year: Publication year (int)

    Returns:
        Dict with: success, bib_text, total, found, failed_count, entries, failed
    """

    if references is None:
        references = []
    if isinstance(references, str):
        try:
            references = json.loads(references)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "references must be a JSON list of objects",
            }

    if not references:
        return {"success": False, "error": "No references provided"}

    # Separate refs into batch-able (have DOI/ArXiv) and match-only (title only)
    batch_refs = []  # (index, s2_id, ref)
    match_refs = []  # (index, ref)

    for i, ref in enumerate(references):
        doi = ref.get("doi", "").strip()
        arxiv = ref.get("arxiv", "").strip()

        # Auto-convert ArXiv DOIs (10.48550/arXiv.XXXX.XXXXX) to ArXiv IDs
        if doi and not arxiv and doi.startswith("10.48550/arXiv."):
            arxiv = doi.replace("10.48550/arXiv.", "")

        if arxiv:
            batch_refs.append((i, f"ArXiv:{arxiv}", ref))
        elif doi:
            batch_refs.append((i, f"DOI:{doi}", ref))
        elif ref.get("title"):
            match_refs.append((i, ref))
        else:
            logger.warning(f"Ref {i}: no doi, arxiv, or title — skipping")

    entries = [None] * len(references)
    failed = []

    # Phase 1: batch lookup for DOI/ArXiv refs (chunked at 500 — S2 API limit)
    S2_BATCH_LIMIT = 500
    if batch_refs:
        logger.info(f"Phase 1: batch lookup for {len(batch_refs)} refs with DOI/ArXiv")

        for chunk_start in range(0, len(batch_refs), S2_BATCH_LIMIT):
            chunk = batch_refs[chunk_start : chunk_start + S2_BATCH_LIMIT]
            s2_ids = [s2_id for _, s2_id, _ in chunk]

            try:
                response = _s2_request_with_retry(
                    "POST",
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": s2_ids},
                )

                if response.status_code == 200:
                    results = response.json()
                    for (idx, s2_id, ref), paper in zip(chunk, results, strict=False):
                        if paper is None:
                            logger.warning(f"Ref {idx}: S2 returned null for {s2_id}")
                            failed.append(
                                {
                                    "index": idx,
                                    "ref": ref,
                                    "reason": f"Not found: {s2_id}",
                                }
                            )
                            continue
                        entry = _process_paper(paper, ref)
                        if entry:
                            entries[idx] = entry
                        else:
                            failed.append(
                                {
                                    "index": idx,
                                    "ref": ref,
                                    "reason": "No BibTeX in response",
                                }
                            )
                else:
                    logger.error(
                        f"S2 batch API returned {response.status_code}: {response.text[:200]}"
                    )
                    # Fall back to individual match for this chunk
                    for idx, _s2_id, ref in chunk:
                        match_refs.append((idx, ref))
            except Exception as e:
                logger.error(f"S2 batch API error: {e}")
                for idx, _s2_id, ref in chunk:
                    match_refs.append((idx, ref))

    # Phase 2: individual match for title-only refs
    if match_refs:
        logger.info(f"Phase 2: title match for {len(match_refs)} refs")

        for i, (idx, ref) in enumerate(match_refs):
            if entries[idx] is not None:
                continue  # already resolved in phase 1

            title = ref.get("title", "")
            author = ref.get("author", "")
            query = title
            if author:
                query = f"{author} {query}"

            try:
                if i > 0:
                    time.sleep(S2_MATCH_DELAY)

                response = _s2_request_with_retry(
                    "GET",
                    S2_MATCH_URL,
                    params={"query": query, "fields": S2_FIELDS},
                )

                if response.status_code == 200:
                    data = response.json()
                    # search/match returns {"data": [...]} with the best match first
                    papers = data.get("data", [])
                    if not papers:
                        failed.append(
                            {
                                "index": idx,
                                "ref": ref,
                                "reason": f"No match for: {query}",
                            }
                        )
                        continue

                    entry = _process_paper(papers[0], ref)
                    if entry:
                        entries[idx] = entry
                    else:
                        failed.append(
                            {
                                "index": idx,
                                "ref": ref,
                                "reason": "No BibTeX in match response",
                            }
                        )
                elif response.status_code == 404:
                    failed.append({"index": idx, "ref": ref, "reason": f"No match for: {query}"})
                else:
                    failed.append(
                        {
                            "index": idx,
                            "ref": ref,
                            "reason": f"S2 HTTP {response.status_code}",
                        }
                    )
            except requests.exceptions.Timeout:
                failed.append({"index": idx, "ref": ref, "reason": "Timeout"})
            except Exception as e:
                failed.append({"index": idx, "ref": ref, "reason": str(e)})

    # Build combined .bib text
    valid_entries = [e for e in entries if e is not None]
    bib_parts = [e["bibtex"] for e in valid_entries]
    bib_text = "\n\n".join(bib_parts)

    return {
        "success": True,
        "bib_text": bib_text,
        "total": len(references),
        "found": len(valid_entries),
        "failed_count": len(failed),
        "entries": valid_entries,
        "failed": failed,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Build .bib from references via Semantic Scholar")
    parser.add_argument("--refs", "-r", required=True, help="JSON array of reference objects")
    parser.add_argument(
        "--json", "-j", action="store_true", help="Output raw JSON instead of .bib text"
    )
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    try:
        references = json.loads(args.refs)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    result = call_server(SERVER_NAME, {"references": references}, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("success"):
        print(f"Found {result.get('found', 0)}/{result.get('total', 0)} references\n")
        if result.get("bib_text"):
            print(result["bib_text"])
        if result.get("failed"):
            print(f"\n% Failed ({result['failed_count']}):", file=sys.stderr)
            for f in result["failed"]:
                print(f"%   [{f['index']}] {f['reason']}", file=sys.stderr)
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
