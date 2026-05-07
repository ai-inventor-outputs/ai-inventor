"""
Ability server client — HTTP client for calling ability endpoints.

The server itself is now part of aii_server (Django).
This package provides the client and convenience functions.
"""

from typing import Any

from ..endpoint_names import (
    AII_HF_DOWNLOAD,
    AII_HF_PREVIEW,
    AII_HF_SEARCH,
    AII_JSON_VALIDATE,
    AII_LEAN_RUN,
    AII_LEAN_SUGGEST,
    AII_OPENROUTER_CALL,
    AII_OPENROUTER_SEARCH,
    AII_OWID_DOWNLOAD,
    AII_OWID_SEARCH,
    AII_SEMSCHOLAR_BIB_FETCH,
    AII_WEB_FETCH,
    AII_WEB_FETCH_GREP,
    AII_WEB_SEARCH,
    AII_WEB_VERIFY_QUOTES,
)
from .ability_client import (
    AbilityTransientError,
    async_call_server,
    call_server,
    get_ability_service_url,
    internal_headers,
    server_available,
)

# =============================================================================
# Convenience functions - call endpoints via HTTP
# =============================================================================


def call_ability(name: str, **kwargs) -> dict[str, Any]:
    """Call an ability by name with keyword arguments."""
    result = call_server(name, kwargs)
    if result is None:
        return {"success": False, "error": "Ability service not available"}
    return result


def hf_search(
    query: str, limit: int = 5, tags: str = "", sort: str = "downloads"
) -> dict[str, Any]:
    """Search HuggingFace datasets."""
    return call_ability(AII_HF_SEARCH, query=query, limit=limit, tags=tags, sort=sort)


def hf_preview(
    dataset_id: str, config: str | None = None, split: str = "train", num_rows: int = 5
) -> dict[str, Any]:
    """Preview a HuggingFace dataset."""
    return call_ability(
        AII_HF_PREVIEW,
        dataset_id=dataset_id,
        config=config,
        split=split,
        num_rows=num_rows,
    )


def hf_download(
    dataset_id: str, config: str | None = None, split: str | None = None
) -> dict[str, Any]:
    """Download a HuggingFace dataset."""
    return call_ability(AII_HF_DOWNLOAD, dataset_id=dataset_id, config=config, split=split)


def aii_web_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search the web."""
    return call_ability(AII_WEB_SEARCH, query=query, max_results=max_results)


def aii_web_fetch(url: str, max_chars: int = 10000, char_offset: int = 0) -> dict[str, Any]:
    """Fetch content from a URL."""
    return call_ability(AII_WEB_FETCH, url=url, max_chars=max_chars, char_offset=char_offset)


def aii_web_fetch_grep(
    url: str,
    pattern: str,
    max_matches: int = 20,
    context_chars: int = 200,
    chars_before: int | None = None,
    chars_after: int | None = None,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Fetch and grep content from a URL."""
    kwargs = {
        "url": url,
        "pattern": pattern,
        "max_matches": max_matches,
        "context_chars": context_chars,
        "case_insensitive": case_insensitive,
    }
    if chars_before is not None:
        kwargs["chars_before"] = chars_before
    if chars_after is not None:
        kwargs["chars_after"] = chars_after
    return call_ability(AII_WEB_FETCH_GREP, **kwargs)


def semscholar_bib_fetch(references: list[dict]) -> dict[str, Any]:
    """Fetch bibliographic data from Semantic Scholar."""
    return call_ability(AII_SEMSCHOLAR_BIB_FETCH, references=references)


def verify_quotes(text: str) -> dict[str, Any]:
    """Verify quotes from text."""
    return call_ability(AII_WEB_VERIFY_QUOTES, text=text)


def lean_run(code: str) -> dict[str, Any]:
    """Run Lean 4 code."""
    return call_ability(AII_LEAN_RUN, code=code)


def lean_suggest(
    code: str,
    tactics: str = "exact?,apply?,simp?,rw?,simp,aesop,omega,decide,ring,linarith,nlinarith,norm_num,field_simp,positivity",
) -> dict[str, Any]:
    """Suggest Lean 4 tactics."""
    return call_ability(AII_LEAN_SUGGEST, code=code, tactics=tactics)


def owid_search(query: str, limit: int = 3) -> dict[str, Any]:
    """Search Our World in Data."""
    return call_ability(AII_OWID_SEARCH, query=query, limit=limit)


def owid_download(path: str, output_dir: str | None = None) -> dict[str, Any]:
    """Download dataset from Our World in Data."""
    kwargs = {"path": path}
    if output_dir:
        kwargs["output_dir"] = output_dir
    return call_ability(AII_OWID_DOWNLOAD, **kwargs)


def json_validate(format_type: str, file_path: str, strict: bool = False) -> dict[str, Any]:
    """Validate JSON file format."""
    return call_ability(
        AII_JSON_VALIDATE, format_type=format_type, file_path=file_path, strict=strict
    )


def openrouter_search(query: str = "", limit: int = 10, series: str = "") -> dict[str, Any]:
    """Search OpenRouter models."""
    return call_ability(AII_OPENROUTER_SEARCH, query=query, limit=limit, series=series)


def openrouter_call(model: str, input_text: str | None = None, **kwargs) -> dict[str, Any]:
    """Call an OpenRouter model."""
    return call_ability(AII_OPENROUTER_CALL, model=model, input_text=input_text, **kwargs)


__all__ = [
    "AbilityTransientError",
    "aii_web_fetch",
    "aii_web_fetch_grep",
    "aii_web_search",
    "async_call_server",
    "call_ability",
    "call_server",
    "get_ability_service_url",
    "hf_download",
    "hf_preview",
    "hf_search",
    "internal_headers",
    "json_validate",
    "lean_run",
    "lean_suggest",
    "openrouter_call",
    "openrouter_search",
    "owid_download",
    "owid_search",
    "semscholar_bib_fetch",
    "server_available",
    "verify_quotes",
]
