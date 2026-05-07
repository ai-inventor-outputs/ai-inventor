"""Wikipedia URL verification for knowledge graph triples.

Verifies that Wikipedia URLs in triples actually exist using HTTP requests.
"""

import concurrent.futures
from collections.abc import Callable

from ...abilities.ability_server.ability_client import call_server
from ...abilities.endpoint_names import AII_WEB_FETCH


def _verify_single_url(url: str) -> dict:
    """Verify a single Wikipedia URL exists.

    Wikipedia returns HTTP 200 even for non-existent pages (shows "page does
    not exist" template). We must check the content for the non-existence
    message.

    Args:
        url: Wikipedia URL to verify

    Returns:
        Dict with keys: url, status ('valid' or 'invalid'), reason (if invalid)
    """
    if not url.startswith("https://en.wikipedia.org/wiki/"):
        return {
            "url": url,
            "status": "invalid",
            "reason": "Invalid format: must start with https://en.wikipedia.org/wiki/",
        }

    try:
        result = call_server(AII_WEB_FETCH, {"url": url, "max_chars": 5000})
        if result is None:
            return {
                "url": url,
                "status": "invalid",
                "reason": "Ability server not responding",
            }

        if not result.get("success"):
            status_code = result.get("status_code", 0)
            return {
                "url": url,
                "status": "invalid",
                "reason": result.get(
                    "error", f"HTTP {status_code}" if status_code else "Fetch failed"
                ),
            }

        # Wikipedia uses redlink=1 parameter for non-existent pages.
        content = result.get("content", "")
        if "action=edit&redlink=1" in content:
            return {
                "url": url,
                "status": "invalid",
                "reason": "Wikipedia article does not exist",
            }

        return {"url": url, "status": "valid"}

    except Exception as e:
        return {"url": url, "status": "invalid", "reason": f"Error: {e!s}"}


def _verify_urls_parallel(urls: list[str], max_workers: int = 10) -> list[dict]:
    """Verify multiple URLs in parallel via a thread pool.

    Replaces the previous async/asyncio.gather pattern: ``call_server`` is
    sync, so the async layer was just wrapping it in ``run_in_executor``.
    Pure threads are simpler and don't pay the per-call asyncio loop
    cold-start that bit other call sites (see _SummaryBuffer history).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_verify_single_url, urls))


def verify_wikipedia_urls(
    triples: list[dict],
    callback: Callable[[str], None] | None = None,
) -> dict:
    """Verify all Wikipedia URLs in triples list.

    Args:
        triples: List of triple dicts, each with 'wikipedia_url' key
        callback: Optional callback for status messages

    Returns:
        Dict with keys:
            - valid: bool (True if all URLs valid)
            - total: int (total URLs checked)
            - verified: int (number of valid URLs)
            - failed: int (number of invalid URLs)
            - results: list[dict] (per-URL results with 'url', 'status', 'reason')
            - failed_triples: list[dict] (triples with invalid URLs)
    """
    if not triples:
        if callback:
            callback("[!] No triples to verify")
        return {
            "valid": True,
            "total": 0,
            "verified": 0,
            "failed": 0,
            "results": [],
            "failed_triples": [],
        }

    # Extract URLs from triples
    urls = [t.get("wikipedia_url", "") for t in triples if t.get("wikipedia_url")]

    if not urls:
        if callback:
            callback("[!] No Wikipedia URLs found in triples")
        return {
            "valid": False,
            "total": 0,
            "verified": 0,
            "failed": len(triples),
            "results": [],
            "failed_triples": triples,
        }

    results = _verify_urls_parallel(urls)

    # Build URL -> result mapping
    url_results = {r["url"]: r for r in results}

    # Find failed triples
    failed_triples = []
    for triple in triples:
        url = triple.get("wikipedia_url", "")
        if url in url_results and url_results[url]["status"] == "invalid":
            failed_triples.append(
                {
                    "triple": triple,
                    "error": url_results[url].get("reason", "Unknown error"),
                }
            )

    # Log results via callback
    for r in results:
        if callback:
            if r["status"] == "valid":
                callback(f"[✓] Valid: {r['url']}")
            else:
                callback(f"[✗] Invalid: {r['url']}\n    Reason: {r.get('reason', 'Unknown')}")

    verified = sum(1 for r in results if r["status"] == "valid")
    failed = len(results) - verified

    return {
        "valid": failed == 0 and results,
        "total": len(results),
        "verified": verified,
        "failed": failed,
        "results": results,
        "failed_triples": failed_triples,
    }


__all__ = ["verify_wikipedia_urls"]
