#!/usr/bin/env python
"""
Citation Verification Tool

Verify quoted citations against source URLs.

Usage:
    python aii_verify_quotes.py --text-file paper.txt

Citation format: ["quote"](https://example.com)
"""

import argparse
import re
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_web_tools__verify_quotes"
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

# Session pooling for connection reuse
_session = None


def init_verify_quotes():
    """Initialize verify quotes environment with warmup."""
    global _session
    import requests
    from requests.adapters import HTTPAdapter

    # Create session with connection pooling (pool_maxsize=50 for parallel requests)
    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Warmup
    try:
        _session.get("https://example.com", timeout=10)
    except Exception:
        pass


@aii_ability(
    name="aii_web_tools__verify_quotes",
    description="Verify quoted citations against source URLs.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_verify_quotes",
)
def core_verify_quotes(text: str = "") -> dict:
    """
    Verify citations in text against source URLs.

    Args:
        text: Text containing citations in format ["quote"](url)

    Returns:
        Dict with success, counts, and citation verification results
    """
    global _session
    import fitz
    import html2text

    # Extract citations
    pattern = r'\["([^"]+)"\]\((https?://[^\)]+)\)'
    matches = re.findall(pattern, text)

    if not matches:
        return {
            "success": False,
            "error": 'No citations found. Format: ["quote"](https://example.com)',
        }

    # Deduplicate
    seen = set()
    citations = []
    for quote, url in matches:
        key = (quote.strip(), url.strip())
        if key not in seen:
            seen.add(key)
            citations.append({"quote": quote.strip(), "url": url.strip()})

    def fetch_content(url: str) -> str | None:
        try:
            resp = _session.get(url, allow_redirects=True, timeout=SESSION_TIMEOUT)
            content_type = resp.headers.get("content-type", "").lower()
            is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
            if is_pdf:
                doc = fitz.open(stream=resp.content, filetype="pdf")
                content = "\n".join(page.get_text() for page in doc)
                doc.close()
                return content
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            return h.handle(resp.text)
        except Exception:
            return None

    def find_match(quote: str, content: str) -> str | None:
        quote_norm = " ".join(quote.split()).lower()
        content_norm = " ".join(content.split()).lower()
        idx = content_norm.find(quote_norm)
        if idx == -1:
            return None
        start = max(0, idx - 100)
        end = min(len(content_norm), idx + len(quote_norm) + 100)
        ctx = content_norm[start:end]
        if start > 0:
            ctx = "..." + ctx
        if end < len(content_norm):
            ctx = ctx + "..."
        return ctx

    # Verify each citation
    results = []
    valid_count = invalid_count = error_count = 0

    for cit in citations:
        content = fetch_content(cit["url"])
        if content is None:
            results.append(
                {
                    "quote": cit["quote"],
                    "source_url": cit["url"],
                    "status": "error",
                    "match_type": "none",
                    "error_message": "Failed to fetch URL",
                }
            )
            error_count += 1
        else:
            context = find_match(cit["quote"], content)
            if context:
                results.append(
                    {
                        "quote": cit["quote"],
                        "source_url": cit["url"],
                        "status": "valid",
                        "match_type": "exact",
                        "context": context,
                    }
                )
                valid_count += 1
            else:
                results.append(
                    {
                        "quote": cit["quote"],
                        "source_url": cit["url"],
                        "status": "invalid",
                        "match_type": "none",
                    }
                )
                invalid_count += 1

    return {
        "success": True,
        "total_citations": len(citations),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "error_count": error_count,
        "citations": results,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Verify citations against source URLs")
    parser.add_argument("--text-file", required=True, help="Path to text file with citations")
    args = parser.parse_args()

    try:
        text = Path(args.text_file).read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    from aii_lib.abilities.ability_server import call_server

    result = call_server(SERVER_NAME, {"text": text}, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.get("success"):
        for idx, r in enumerate(result.get("citations", []), 1):
            if r.get("status") == "error":
                verdict = f"ERROR - {r.get('error_message', 'Unknown')}"
            elif r.get("status") == "valid":
                verdict = "VALID"
            else:
                verdict = "INVALID"
            print(f"Citation {idx}/{result['total_citations']}: {r['quote'][:60]}...")
            print(f"URL: {r['source_url']}")
            print(f"Verdict: {verdict}\n")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
