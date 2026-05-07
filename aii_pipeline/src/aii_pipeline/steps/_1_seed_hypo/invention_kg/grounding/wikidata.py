#!/usr/bin/env python3
"""
Wikidata enrichment for triples.

Two-phase enrichment:
1. Wikipedia REST API -> Get QID and description (fast)
2. Wikidata Entity API -> Get ALL properties (claims)

Stores everything under a single `wikidata` key so we can later
analyze which properties are common/useful and discard others.

No API key needed - both APIs are free and public.
"""

import asyncio
import time
from typing import Any
from urllib.parse import quote

import aiohttp
import requests
from aii_lib.run import emit
from aii_lib.utils.retry import make_retry_log
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._parser import (
    extract_title_from_wikipedia_url,
    parse_wikidata_entity_full,
    resolve_qids_in_claims,
)


async def get_wikipedia_qid_async(
    session: aiohttp.ClientSession, title: str, semaphore: asyncio.Semaphore
) -> dict[str, Any] | None:
    """Phase 1: Get QID from Wikipedia REST API."""
    async with semaphore:
        encoded_title = quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

        last_exc = None
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 404:
                        return None
                    if response.status in (429, 502, 503, 504):
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                        )
                    response.raise_for_status()
                    data = await response.json()

                    qid = data.get("wikibase_item")
                    if qid:
                        return {"id": qid, "description": data.get("description", "")}
                    return None
            except (TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("All retry attempts failed for Wikipedia QID lookup")


async def get_wikidata_entity_full_async(
    session: aiohttp.ClientSession, qid: str, semaphore: asyncio.Semaphore
) -> dict[str, Any] | None:
    """Phase 2: Get full Wikidata entity with ALL claims."""
    async with semaphore:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

        last_exc = None
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 404:
                        return None
                    if response.status in (429, 502, 503, 504):
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                        )
                    response.raise_for_status()
                    data = await response.json()

                    entity = data.get("entities", {}).get(qid)
                    if entity:
                        return parse_wikidata_entity_full(entity)
                    return None
            except (TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("All retry attempts failed for Wikidata entity fetch")


async def resolve_qids_batch(
    session: aiohttp.ClientSession, qids: set[str], semaphore: asyncio.Semaphore
) -> dict[str, str]:
    """Resolve Q-IDs to labels in batch (max 50 at a time)."""
    if not qids:
        return {}

    qid_labels = {}
    qid_list = list(qids)

    # Process in batches of 50 (Wikidata API limit)
    from itertools import batched

    for batch in batched(qid_list, 50):
        async with semaphore:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
            last_exc = None
            for attempt in range(3):
                try:
                    async with session.get(
                        url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status in (429, 502, 503, 504):
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                            )
                        if response.status == 200:
                            data = await response.json()
                            for qid, ent in data.get("entities", {}).items():
                                label = ent.get("labels", {}).get("en", {}).get("value")
                                if label:
                                    qid_labels[qid] = label
                        break
                except (TimeoutError, aiohttp.ClientError) as e:
                    last_exc = e
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)
            else:
                if last_exc:
                    raise last_exc

    return qid_labels


async def enrich_triples_async(
    triples: list[dict[str, Any]], max_concurrent: int = 10, resolve_labels: bool = True
) -> list[dict[str, Any]]:
    """
    Enrich all triples with full Wikidata data.

    Phase 1: Wikipedia REST API -> QID
    Phase 2: Wikidata Entity API -> ALL claims
    Phase 3: Resolve Q-IDs to labels (optional)

    Stores everything under triple["wikidata"] = {...}
    """
    # Filter triples that need enrichment
    to_enrich = [
        (i, t) for i, t in enumerate(triples) if not t.get("wikidata") and t.get("wikipedia_url")
    ]

    if not to_enrich:
        already_done = sum(1 for t in triples if t.get("wikidata"))
        emit.status_public_info(f"Wikidata: {already_done}/{len(triples)} already enriched")
        return triples

    emit.status_public_info(f"Phase 1: Getting QIDs for {len(to_enrich)} triples...")

    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {"User-Agent": "InventionKG/1.0 (research project)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        # Phase 1: Get QIDs
        phase1_tasks = []
        for idx, triple in to_enrich:
            title = extract_title_from_wikipedia_url(triple.get("wikipedia_url", ""))
            if title:
                phase1_tasks.append(
                    (idx, title, get_wikipedia_qid_async(session, title, semaphore))
                )

        phase1_results = await asyncio.gather(*[t[2] for t in phase1_tasks], return_exceptions=True)

        # Collect QIDs for Phase 2
        qids_to_fetch = []
        for (idx, _title, _), result in zip(phase1_tasks, phase1_results, strict=False):
            if isinstance(result, BaseException) or not result:
                continue
            qids_to_fetch.append((idx, result["id"], result.get("description", "")))

        emit.status_public_info(f"Phase 1: {len(qids_to_fetch)}/{len(to_enrich)} QIDs found")

        if not qids_to_fetch:
            return triples

        # Phase 2: Get full entity data
        emit.status_public_info("Phase 2: Fetching full entity data...")

        phase2_tasks = []
        for idx, qid, desc in qids_to_fetch:
            phase2_tasks.append(
                (
                    idx,
                    qid,
                    desc,
                    get_wikidata_entity_full_async(session, qid, semaphore),
                )
            )

        phase2_results = await asyncio.gather(*[t[3] for t in phase2_tasks], return_exceptions=True)

        # Collect all Q-IDs that need label resolution
        all_qids_to_resolve: set[str] = set()
        entities_fetched = []

        for (idx, qid, desc, _), result in zip(phase2_tasks, phase2_results, strict=False):
            if isinstance(result, Exception) or not result:
                # Still store basic info even if full fetch failed
                triples[idx]["wikidata"] = {"id": qid, "description": desc}
                continue

            entities_fetched.append((idx, result))

            # Collect Q-IDs from claims for resolution
            if resolve_labels:
                for value in result.get("claims", {}).values():
                    if isinstance(value, str) and value.startswith("Q"):
                        all_qids_to_resolve.add(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and v.startswith("Q"):
                                all_qids_to_resolve.add(v)

        emit.status_public_info(f"Phase 2: {len(entities_fetched)} entities fetched")

        # Phase 3: Resolve Q-ID labels
        qid_labels = {}
        if resolve_labels and all_qids_to_resolve:
            emit.status_public_info(f"Phase 3: Resolving {len(all_qids_to_resolve)} Q-ID labels...")
            qid_labels = await resolve_qids_batch(session, all_qids_to_resolve, semaphore)
            emit.status_public_info(f"Phase 3: {len(qid_labels)} labels resolved")

        # Store results in triples
        for idx, entity_data in entities_fetched:
            # Resolve Q-IDs to labels in claims
            if qid_labels and entity_data.get("claims"):
                entity_data["claims"] = resolve_qids_in_claims(entity_data["claims"], qid_labels)

            triples[idx]["wikidata"] = entity_data

    enriched = sum(1 for t in triples if t.get("wikidata"))
    emit.status_public_success(f"Done: {enriched}/{len(triples)} triples enriched")

    return triples


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=make_retry_log(label="Wikidata query"),
    reraise=True,
)
def _sync_get(session, url, **kwargs):
    """requests.get with retry on transient errors."""
    resp = session.get(url, **kwargs)
    if resp.status_code in (429, 502, 503, 504):
        raise requests.exceptions.ConnectionError(f"Transient HTTP {resp.status_code}")
    return resp


def enrich_triples_sync(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synchronous fallback (basic enrichment only)."""
    headers = {"User-Agent": "InventionKG/1.0 (research project)"}
    session = requests.Session()
    session.headers.update(headers)
    enriched = 0

    for triple in triples:
        if triple.get("wikidata"):
            enriched += 1
            continue

        url = triple.get("wikipedia_url")
        if not url:
            continue

        title = extract_title_from_wikipedia_url(url)
        if not title:
            continue

        # Get QID
        encoded = quote(title.replace(" ", "_"))
        resp = _sync_get(
            session,
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
            timeout=10,
        )
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        data = resp.json()

        qid = data.get("wikibase_item")
        if not qid:
            continue

        # Get full entity
        time.sleep(0.1)
        resp2 = _sync_get(
            session,
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
            timeout=15,
        )
        if resp2.status_code == 200:
            entity = resp2.json().get("entities", {}).get(qid)
            if entity:
                triple["wikidata"] = parse_wikidata_entity_full(entity)
                enriched += 1

        time.sleep(0.1)

    emit.status_public_info(f"Wikidata: {enriched}/{len(triples)} enriched")
    return triples


# ============================================================================
# Main enrichment function (used by step 5)
# ============================================================================


def enrich_triples_with_wikidata(
    triples: list[dict[str, Any]], use_async: bool = True, resolve_labels: bool = True
) -> list[dict[str, Any]]:
    """
    Enrich triples with full Wikidata data.

    Stores ALL Wikidata info under triple["wikidata"] = {
        "id": "Q2539",
        "label": "machine learning",
        "description": "...",
        "aliases": ["ML", ...],
        "claims": {
            "instance_of": [{"id": "Q11862829", "label": "academic discipline"}, ...],
            "subclass_of": [...],
            "openalex_id": "C2982736386",
            ...all other properties...
        }
    }
    """
    if not triples:
        return triples

    if use_async:
        return asyncio.run(enrich_triples_async(triples, resolve_labels=resolve_labels))
    return enrich_triples_sync(triples)


# Convenience function
def get_wikidata_from_wikipedia_url(url: str) -> dict[str, Any] | None:
    """Get full Wikidata info from Wikipedia URL."""
    title = extract_title_from_wikipedia_url(url)
    if not title:
        return None

    session = requests.Session()
    session.headers.update({"User-Agent": "InventionKG/1.0"})

    encoded = quote(title.replace(" ", "_"))
    resp = _sync_get(
        session,
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    qid = resp.json().get("wikibase_item")

    if qid:
        resp2 = _sync_get(
            session,
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
            timeout=15,
        )
        if resp2.status_code == 200:
            entity = resp2.json().get("entities", {}).get(qid)
            if entity:
                return parse_wikidata_entity_full(entity)
    return None
