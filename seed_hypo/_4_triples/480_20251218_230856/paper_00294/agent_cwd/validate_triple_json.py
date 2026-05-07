#!/usr/bin/env python3
"""
Validation script for triples_output.json

Checks:
1. Valid JSON format
2. Required fields present
3. Enum values correct (paper_type, entity_type, relation)
4. Wikipedia URL format valid
5. Wikipedia pages actually exist (HTTP HEAD request)
6. Triples structure correct
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse
import urllib.request as urlreq
from urllib.error import HTTPError, URLError

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
END = "\033[0m"

# Valid enum values
VALID_PAPER_TYPES = ["contribution", "survey"]
VALID_ENTITY_TYPES = ["task", "method", "data", "artifact", "tool", "concept", "other"]
VALID_RELATIONS = ["uses", "proposes"]

# Cache for Wikipedia URL verification
_url_cache = {}


def verify_wikipedia_url(url: str) -> tuple[bool, str]:
    """
    Verify a Wikipedia URL exists using Wikipedia REST API.

    Returns:
        (is_valid, error_message)
    """
    # Check cache first
    if url in _url_cache:
        return _url_cache[url]

    # Extract page title from URL
    # URL format: https://en.wikipedia.org/wiki/Page_Title
    if not url.startswith("https://en.wikipedia.org/wiki/"):
        msg = "Invalid Wikipedia URL format"
        _url_cache[url] = (False, msg)
        return False, msg

    # Get page title from URL (everything after /wiki/)
    page_title = url.replace("https://en.wikipedia.org/wiki/", "")

    # Use Wikipedia REST API to check if page exists
    # API endpoint: https://en.wikipedia.org/w/rest.php/v1/page/{title}
    api_url = f"https://en.wikipedia.org/w/rest.php/v1/page/{page_title}"

    try:
        req = urlreq.Request(api_url, headers={'User-Agent': 'Mozilla/5.0 (validate_triple_json validator)'})
        with urlreq.urlopen(req, timeout=5) as response:
            if response.status == 200:
                _url_cache[url] = (True, "")
                return True, ""
            else:
                msg = f"Wikipedia API returned status {response.status}"
                _url_cache[url] = (False, msg)
                return False, msg
    except HTTPError as e:
        msg = f"HTTP {e.code}: Page not found or does not exist on Wikipedia"
        _url_cache[url] = (False, msg)
        return False, msg
    except URLError as e:
        msg = f"Network error: {e.reason}"
        _url_cache[url] = (False, msg)
        return False, msg
    except Exception as e:
        msg = f"Verification failed: {str(e)}"
        _url_cache[url] = (False, msg)
        return False, msg


def validate_analysis(file_path: Path, verify_urls: bool = True) -> tuple[bool, list[str]]:
    """
    Validate the analysis output file.

    Args:
        file_path: Path to triples_output.json
        verify_urls: Whether to verify Wikipedia URLs exist (HTTP requests)

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check file exists
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    # Load JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON syntax: {e}\n  → Fix: Ensure valid JSON formatting"]

    # Check required top-level fields
    required_fields = ["paper_type", "triples"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'\n  → Fix: Add '{field}' to top-level JSON")

    # Validate paper_type enum
    if "paper_type" in data:
        if data["paper_type"] not in VALID_PAPER_TYPES:
            errors.append(f"Invalid paper_type: '{data['paper_type']}'\n  → Fix: Must be one of: {VALID_PAPER_TYPES}")

    # Validate triples structure
    if "triples" in data:
        if not isinstance(data["triples"], list):
            errors.append("'triples' must be a list\n  → Fix: Use array syntax: \"triples\": [...]")
        else:
            for idx, block in enumerate(data["triples"]):
                # Check it's a dict
                if not isinstance(block, dict):
                    errors.append(f"triples[{idx}] must be an object/dict")
                    continue

                # Check required fields
                required_block_fields = ["name", "entity_type", "relation", "relevance", "wikipedia_url"]
                for field in required_block_fields:
                    if field not in block:
                        errors.append(f"triples[{idx}] missing '{field}' field\n  → Fix: Add \"{field}\" to this triple")

                # Validate name is not empty
                if "name" in block:
                    if not isinstance(block["name"], str) or not block["name"].strip():
                        errors.append(f"triples[{idx}] 'name' must be a non-empty string")

                # Validate entity_type enum
                if "entity_type" in block:
                    if block["entity_type"] not in VALID_ENTITY_TYPES:
                        errors.append(f"triples[{idx}] invalid entity_type: '{block['entity_type']}'\n  → Fix: Must be one of: {VALID_ENTITY_TYPES}")

                # Validate relation enum
                if "relation" in block:
                    if block["relation"] not in VALID_RELATIONS:
                        errors.append(f"triples[{idx}] invalid relation: '{block['relation']}'\n  → Fix: Must be one of: {VALID_RELATIONS}")

                # Validate relevance is not empty
                if "relevance" in block:
                    relevance = block["relevance"]
                    if not isinstance(relevance, str):
                        errors.append(f"triples[{idx}] 'relevance' must be a string")
                    elif not relevance.strip():
                        errors.append(f"triples[{idx}] 'relevance' cannot be empty\n  → Fix: Provide a 1-sentence explanation")

                # Validate Wikipedia URL format
                if "wikipedia_url" in block:
                    url = block["wikipedia_url"]
                    if not isinstance(url, str):
                        errors.append(f"triples[{idx}] 'wikipedia_url' must be a string")
                    elif not url.startswith("https://en.wikipedia.org/wiki/"):
                        errors.append(f"triples[{idx}] invalid Wikipedia URL format: {url}\n  → Fix: Must start with 'https://en.wikipedia.org/wiki/'")
                    else:
                        # Verify URL actually exists
                        if verify_urls:
                            is_valid, error_msg = verify_wikipedia_url(url)
                            if not is_valid:
                                errors.append(f"triples[{idx}] Wikipedia page does not exist: {url}\n  → {error_msg}\n  → Fix: Use WikiSearch to find the correct Wikipedia article")

                # Check for extra fields (warning only)
                allowed_fields = {"name", "entity_type", "relation", "relevance", "wikipedia_url"}
                extra_fields = set(block.keys()) - allowed_fields
                if extra_fields:
                    errors.append(f"triples[{idx}] has unexpected fields: {extra_fields}\n  → Fix: Remove these fields")

    # Check for extra top-level fields (warning only)
    allowed_top_level = {"paper_type", "triples"}
    extra_top_level = set(data.keys()) - allowed_top_level
    if extra_top_level:
        errors.append(f"Unexpected top-level fields: {extra_top_level}\n  → Fix: Remove these fields (only {allowed_top_level} allowed)")

    # Validate relation requirements based on paper_type
    if "triples" in data and isinstance(data["triples"], list) and "paper_type" in data:
        relation_counts = {"uses": 0, "proposes": 0}
        for triple in data["triples"]:
            if isinstance(triple, dict) and "relation" in triple:
                rel = triple["relation"]
                if rel in relation_counts:
                    relation_counts[rel] += 1

        # Both contribution and survey must have at least 1 "uses"
        if relation_counts["uses"] == 0:
            errors.append(
                f"Paper must have at least 1 'uses' relation (found 0)\n"
                f"  → Fix: Every paper should reference existing work it uses"
            )

        # Contribution papers must have at least 1 "proposes"
        if data["paper_type"] == "contribution" and relation_counts["proposes"] == 0:
            errors.append(
                f"Contribution papers must have at least 1 'proposes' relation (found 0)\n"
                f"  → Fix: Identify what new entity the paper introduces/proposes"
            )

    return len(errors) == 0, errors


def main():
    """Run validation and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate triples_output.json")
    parser.add_argument("--no-verify-urls", action="store_true", help="Skip Wikipedia URL verification (faster)")
    args = parser.parse_args()

    file_path = Path(__file__).parent / "triples_output.json"

    print(f"{BLUE}╔══════════════════════════════════════════════════════════════╗{END}")
    print(f"{BLUE}║  Validating Analysis Output                                  ║{END}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════════════╝{END}\n")
    print(f"{CYAN}File: {file_path}{END}")

    if args.no_verify_urls:
        print(f"{YELLOW}Note: Wikipedia URL verification disabled{END}\n")
    else:
        print(f"{CYAN}Note: Verifying Wikipedia URLs (this may take a moment)...{END}\n")

    is_valid, errors = validate_analysis(file_path, verify_urls=not args.no_verify_urls)

    if is_valid:
        print(f"\n{GREEN}╔══════════════════════════════════════════════════════════════╗{END}")
        print(f"{GREEN}║  ✅ VALIDATION PASSED                                        ║{END}")
        print(f"{GREEN}╚══════════════════════════════════════════════════════════════╝{END}")
        print(f"{GREEN}All checks passed successfully!{END}")

        # Show summary
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"\n{CYAN}Summary:{END}")
        print(f"  Paper Type: {data.get('paper_type', 'N/A')}")
        print(f"  Triples: {len(data.get('triples', []))}")

        # Show entity type breakdown
        if data.get('triples'):
            type_counts = {}
            for block in data['triples']:
                et = block.get('entity_type', 'unknown')
                type_counts[et] = type_counts.get(et, 0) + 1
            print(f"  Entity Types: {type_counts}")

        return 0
    else:
        print(f"\n{RED}╔══════════════════════════════════════════════════════════════╗{END}")
        print(f"{RED}║  ❌ VALIDATION FAILED                                        ║{END}")
        print(f"{RED}╚══════════════════════════════════════════════════════════════╝{END}\n")
        print(f"{RED}Found {len(errors)} error(s):{END}\n")
        for idx, error in enumerate(errors, 1):
            # Split multi-line errors for better formatting
            lines = error.split('\n')
            print(f"{YELLOW}{idx}. {lines[0]}{END}")
            for line in lines[1:]:
                print(f"   {CYAN}{line}{END}")
        print(f"\n{RED}Please fix the errors above and run validation again.{END}")
        print(f"{CYAN}Tip: Each error includes a '→ Fix:' suggestion{END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
