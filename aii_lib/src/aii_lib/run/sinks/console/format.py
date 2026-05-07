"""JSON formatting helper for ConsoleRunSink output.

Lifted from the now-deleted ``aii_lib.telemetry.utils.json_formatter``
so the sink owns its own pretty-printer + MCP-content unwrapper.
"""

import json
import re


def _clean_string_value(s: str) -> str:
    r"""Replace \\n\\t whitespace artifacts with single spaces and trim."""
    s = re.sub(r"[\n\t]+", " ", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


def _clean_json_strings(obj: object) -> object:
    """Recursively clean string values in a JSON object for display."""
    if isinstance(obj, str):
        return _clean_string_value(obj)
    if isinstance(obj, dict):
        return {k: _clean_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_strings(item) for item in obj]
    return obj


def _unwrap_mcp_content(parsed: object) -> tuple[object, bool]:
    """Unwrap MCP ``[{"type":"text","text":"..."}]`` content blocks.

    Returns ``(unwrapped, was_unwrapped)``. The text field gets parsed
    as JSON when possible so we display the inner payload pretty-printed
    instead of as an escaped string.
    """
    if isinstance(parsed, dict) and parsed.get("type") == "text" and "text" in parsed:
        text_content = parsed["text"]
        if isinstance(text_content, str):
            try:
                nested = json.loads(text_content)
                return _clean_json_strings(nested), True
            except json.JSONDecodeError:
                return _clean_string_value(text_content), True
        return parsed, False

    if isinstance(parsed, list) and len(parsed) >= 1:
        unwrapped_items = []
        any_unwrapped = False
        for item in parsed:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_content = item["text"]
                if isinstance(text_content, str):
                    try:
                        nested = json.loads(text_content)
                        unwrapped_items.append(_clean_json_strings(nested))
                        any_unwrapped = True
                        continue
                    except json.JSONDecodeError:
                        unwrapped_items.append(_clean_string_value(text_content))
                        any_unwrapped = True
                        continue
            unwrapped_items.append(item)
        if any_unwrapped:
            if len(unwrapped_items) == 1:
                return unwrapped_items[0], True
            return unwrapped_items, True

    return parsed, False


def format_json_output(text: str, indent: int = 2) -> str:
    """Pretty-print JSON in *text* (with MCP content-block unwrapping)."""
    try:
        parsed = json.loads(text)
        parsed, _ = _unwrap_mcp_content(parsed)
        formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
        lines = formatted.split("\n")
        if len(lines) > 1:
            formatted = "\n         ".join(lines)
        return formatted
    except json.JSONDecodeError:
        pass

    # Skip regex on large strings to avoid catastrophic backtracking
    if len(text) > 10000:
        return text

    # Try to find JSON objects/arrays within the text
    json_pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])"

    def format_match(match: object) -> str:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            lines = formatted.split("\n")
            if len(lines) > 1:
                formatted = "\n         ".join(lines)
            return formatted
        except json.JSONDecodeError:
            return json_str

    return re.sub(json_pattern, format_match, text)
