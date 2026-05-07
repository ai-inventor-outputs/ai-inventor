#!/usr/bin/env python
"""
Fast Web Fetch Tool

Fetch web pages as markdown (HTML and PDF supported).
Supports optional grep-style pattern matching to search through full documents.

Usage:
    python aii_fast_web_fetch.py --url "https://example.com"
    python aii_fast_web_fetch.py --url "https://arxiv.org/pdf/2301.00001.pdf" --max-chars 5000
    python aii_fast_web_fetch.py --url "https://arxiv.org/pdf/2301.00001.pdf" --pattern "attention mechanism" --context 5
"""

import argparse
import re
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_web_tools__fetch"
DEFAULT_TIMEOUT = 120.0
SESSION_TIMEOUT = 120
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

# Session pooling for connection reuse
_session = None


def init_web_fetch():
    """Initialize web fetch environment with warmup."""
    global _session
    import fitz  # pymupdf
    import requests
    from requests.adapters import HTTPAdapter

    # Suppress MuPDF warnings (color space errors, etc.)
    fitz.TOOLS.mupdf_display_errors(False)

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


def _grep_content(
    content: str,
    pattern: str,
    max_matches: int = 20,
    chars_before: int = 200,
    chars_after: int = 200,
    case_insensitive: bool = False,
) -> dict:
    """
    Search content for pattern matches, returning char-based context around each match.

    Uses character windows instead of line-based context to handle web pages where
    a single "line" can be an entire paragraph or even the whole page.

    Args:
        content: Full text to search through
        pattern: Regex pattern to search for
        max_matches: Maximum number of matches to return (default: 20)
        chars_before: Characters of context before each match (default: 200)
        chars_after: Characters of context after each match (default: 200)
        case_insensitive: Whether to ignore case

    Returns:
        Dict with match_count (total found) and formatted content (up to max_matches)
    """
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return {"match_count": 0, "content": f"Invalid regex pattern: {e}"}

    matches = list(compiled.finditer(content))

    if not matches:
        return {"match_count": 0, "content": f"No matches found for pattern: {pattern}"}

    total_matches = len(matches)
    matches = matches[:max_matches]

    # Build context windows, merging overlapping ones
    windows = []
    for m in matches:
        win_start = max(0, m.start() - chars_before)
        win_end = min(len(content), m.end() + chars_after)
        # Merge with previous window if overlapping
        if windows and win_start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], win_end, windows[-1][2] + [m])
        else:
            windows.append((win_start, win_end, [m]))

    # Format output (ripgrep-style: "char_pos:context_with_match")
    output_parts = []
    for win_start, win_end, win_matches in windows:
        snippet = content[win_start:win_end]
        prefix = "..." if win_start > 0 else ""
        suffix = "..." if win_end < len(content) else ""
        output_parts.append(f"{win_matches[0].start()}:{prefix}{snippet}{suffix}")

    result_content = "\n--\n".join(output_parts)
    if total_matches > max_matches:
        result_content += f"\n--\n[{total_matches - max_matches} more matches not shown]"

    return {
        "match_count": total_matches,
        "content": result_content,
    }


def _fetch_url(url: str) -> dict:
    """
    Fetch a URL and return raw content + metadata.

    Returns dict with: success, url, status_code, content (str), is_pdf, original_length.
    On failure: success=False, error message.
    """
    global _session
    import fitz
    import html2text

    if _session is None:
        init_web_fetch()

    if not url or not url.startswith(("http://", "https://")):
        return {"success": False, "error": "Invalid URL"}

    try:
        resp = _session.get(url, allow_redirects=True, timeout=SESSION_TIMEOUT)
        status_code = resp.status_code

        if status_code != 200:
            return {
                "success": False,
                "url": url,
                "status_code": status_code,
                "error": f"HTTP {status_code}",
            }

        content_type = resp.headers.get("content-type", "").lower()
        is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")

        if is_pdf:
            doc = fitz.open(stream=resp.content, filetype="pdf")
            content = "\n".join(page.get_text() for page in doc)
            doc.close()
        else:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            content = h.handle(resp.text)

        return {
            "success": True,
            "url": url,
            "status_code": status_code,
            "content": content,
            "original_length": len(content),
            "is_pdf": is_pdf,
        }
    except Exception as e:
        return {"success": False, "url": url, "status_code": 0, "error": str(e)}


@aii_ability(
    name="aii_web_tools__fetch",
    description="Fetch a URL and return a slice of its text content (HTML or PDF).",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_web_fetch",
)
def core_web_fetch(url: str = "", max_chars: int = 10000, char_offset: int = 0) -> dict:
    """
    Fetch a URL and return a slice of its text content.

    Args:
        url: URL to fetch (HTML or PDF)
        max_chars: Maximum characters to return (default: 10000, max: 50000)
        char_offset: Character offset to start from (default: 0). Use with max_chars to paginate.

    Returns:
        Dict with success, url, content, original_length, truncated, is_pdf, char_offset
    """
    max_chars = min(max_chars, 50000)
    char_offset = max(char_offset, 0)

    result = _fetch_url(url)
    if not result["success"]:
        return result

    content = result["content"]
    original_length = result["original_length"]

    # Apply offset + truncation
    content = content[char_offset:]
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return {
        "success": True,
        "url": result["url"],
        "status_code": result["status_code"],
        "content": content,
        "original_length": original_length,
        "truncated": truncated,
        "is_pdf": result["is_pdf"],
        "char_offset": char_offset,
    }


@aii_ability(
    name="aii_web_tools__fetch_grep",
    description="Fetch a URL and grep through its full content for a regex pattern.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_web_fetch",
)
def core_web_grep(
    url: str = "",
    pattern: str = "",
    max_matches: int = 20,
    context_chars: int = 200,
    chars_before: str | None = None,
    chars_after: str | None = None,
    case_insensitive: bool = False,
) -> dict:
    """
    Fetch a URL and grep through its full content for a regex pattern.

    Always fetches the ENTIRE document (HTML or PDF) and searches for matching
    content, returning each match with a character-based context window.

    Args:
        url: URL to fetch (HTML or PDF)
        pattern: Regex pattern to search for (required)
        max_matches: Maximum matches to return (default: 20). Total count still reported.
        context_chars: Characters of context before AND after each match (default: 200).
        chars_before: Characters before each match. Overrides context_chars for before.
        chars_after: Characters after each match. Overrides context_chars for after.
        case_insensitive: Case-insensitive matching (default: false).

    Returns:
        Dict with success, url, content (formatted grep output), match_count, original_length, is_pdf
    """

    if not pattern:
        return {"success": False, "error": "pattern is required"}

    result = _fetch_url(url)
    if not result["success"]:
        return result

    # Resolve before/after from context_chars
    ctx_before = chars_before if chars_before is not None else context_chars
    ctx_after = chars_after if chars_after is not None else context_chars

    grep_result = _grep_content(
        content=result["content"],
        pattern=pattern,
        max_matches=max_matches,
        chars_before=ctx_before,
        chars_after=ctx_after,
        case_insensitive=case_insensitive,
    )

    return {
        "success": True,
        "url": result["url"],
        "status_code": result["status_code"],
        "content": grep_result["content"],
        "original_length": result["original_length"],
        "match_count": grep_result["match_count"],
        "pattern": pattern,
        "is_pdf": result["is_pdf"],
    }


# =============================================================================
# CLI
# =============================================================================


def _print_result(result: dict):
    """Print fetch/grep result to stdout."""
    if result.get("success"):
        print(f"URL: {result['url']}")
        print(f"Type: {'PDF' if result.get('is_pdf') else 'HTML'}")
        if result.get("match_count") is not None:
            print(
                f"Pattern: {result['pattern']} ({result['match_count']} matches in {result['original_length']} chars)"
            )
        else:
            offset_info = (
                f" (offset: {result['char_offset']})" if result.get("char_offset", 0) > 0 else ""
            )
            print(
                f"Length: {result['original_length']} chars"
                + (" (truncated)" if result.get("truncated") else "")
                + offset_info
            )
        print("\n--- Content ---\n")
        print(result["content"])
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fetch web pages or grep through them")
    sub = parser.add_subparsers(dest="command", help="Command")

    # Fetch subcommand
    fetch = sub.add_parser("fetch", help="Fetch a web page as text")
    fetch.add_argument("--url", "-u", required=True, help="URL to fetch")
    fetch.add_argument("--max-chars", "-n", type=int, default=10000)
    fetch.add_argument("--char-offset", type=int, default=0, help="Character offset to start from")
    fetch.add_argument("--output", "-o", help="Save to file")

    # Grep subcommand
    grep = sub.add_parser("grep", help="Grep through a web page or PDF")
    grep.add_argument("--url", "-u", required=True, help="URL to fetch")
    grep.add_argument("--pattern", "-p", required=True, help="Regex pattern")
    grep.add_argument(
        "-m",
        "--max-matches",
        type=int,
        default=20,
        help="Max matches to return (default: 20)",
    )
    grep.add_argument(
        "-C",
        "--context-chars",
        type=int,
        default=200,
        help="Chars of context around matches",
    )
    grep.add_argument("-B", "--chars-before", type=int, default=None, help="Chars before matches")
    grep.add_argument("-A", "--chars-after", type=int, default=None, help="Chars after matches")
    grep.add_argument("-i", "--ignore-case", action="store_true", help="Case-insensitive matching")
    grep.add_argument("--output", "-o", help="Save to file")

    args = parser.parse_args()

    # Default to fetch if no subcommand
    if not args.command:
        parser.print_help()
        sys.exit(1)

    from aii_lib.abilities.ability_server import call_server

    if args.command == "fetch":
        result = call_server(
            SERVER_NAME,
            {
                "url": args.url,
                "max_chars": args.max_chars,
                "char_offset": args.char_offset,
            },
            timeout=DEFAULT_TIMEOUT,
        )
    else:  # grep
        params = {
            "url": args.url,
            "pattern": args.pattern,
            "max_matches": args.max_matches,
            "context_chars": args.context_chars,
            "case_insensitive": args.ignore_case,
        }
        if args.chars_before is not None:
            params["chars_before"] = args.chars_before
        if args.chars_after is not None:
            params["chars_after"] = args.chars_after
        result = call_server("aii_web_tools__fetch_grep", params, timeout=DEFAULT_TIMEOUT)

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(result.get("content", ""))
        print(f"Saved to: {args.output}")
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
