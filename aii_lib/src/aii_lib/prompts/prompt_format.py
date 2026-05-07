"""Low-level YAML formatting for LLM prompts.

YAML is the recommended format for structured input to LLMs:
- 30-55% fewer tokens than pretty-printed JSON
- Better comprehension accuracy than JSON/XML across models
- Block scalars (> and |) handle long strings cleanly

References:
- https://www.improvingagents.com/blog/best-nested-data-format/
- https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df
"""

from __future__ import annotations

from enum import Enum

import yaml

# =============================================================================
# Custom YAML Dumper
# =============================================================================


class _PromptDumper(yaml.SafeDumper):
    """YAML dumper optimized for LLM prompt readability.

    - Long strings (>threshold chars) use folded block scalar (>)
    - Strings with newlines use literal block scalar (|)
    - No YAML document markers (---)
    - No trailing ...
    - Clean 2-space indentation
    """


def _str_representer(dumper: _PromptDumper, data: str) -> yaml.ScalarNode:
    """Represent strings with block scalars when appropriate."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    if len(data) > 80:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _none_representer(dumper: _PromptDumper, _data: None) -> yaml.ScalarNode:
    """Represent None as empty string instead of 'null'."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


def _enum_representer(dumper: _PromptDumper, data: Enum) -> yaml.ScalarNode:
    """Represent Enum members by their value."""
    return dumper.represent_data(data.value)


_PromptDumper.add_representer(str, _str_representer)
_PromptDumper.add_representer(type(None), _none_representer)
_PromptDumper.add_multi_representer(Enum, _enum_representer)


# =============================================================================
# Public API
# =============================================================================


def to_prompt_yaml(data: dict | list, *, strip_nulls: bool = False) -> str:
    """Convert a dict or list to clean YAML for embedding in LLM prompts.

    Args:
        data: Dict or list to convert (typically from Pydantic model_dump()).
        strip_nulls: If True, recursively remove keys with None/empty values.

    Returns:
        Clean YAML string (no document markers, no trailing newline).
    """
    if strip_nulls:
        data = _strip_nulls(data)

    result = yaml.dump(
        data,
        Dumper=_PromptDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    return result.rstrip("\n")


def to_prompt_yaml_list(
    items: list[dict],
    *,
    separator: str = "---",
    label: str = "Item",
    strip_nulls: bool = False,
) -> str:
    """Convert a list of dicts to labeled YAML blocks.

    Args:
        items: List of dicts to convert.
        separator: Line prefix for each block header.
        label: Label for each block (e.g., "Strategy", "Artifact").
        strip_nulls: If True, recursively remove keys with None/empty values.

    Returns:
        Labeled YAML blocks separated by blank lines.

    Example:
        >>> items = [{"title": "A"}, {"title": "B"}]
        >>> print(to_prompt_yaml_list(items, label="Strategy"))
        --- Strategy 1 ---
        title: A

        --- Strategy 2 ---
        title: B
    """
    blocks = []
    for i, item in enumerate(items, 1):
        header = f"{separator} {label} {i} {separator}"
        body = to_prompt_yaml(item, strip_nulls=strip_nulls)
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


# =============================================================================
# Helpers
# =============================================================================


def _strip_nulls(data: dict | list) -> dict | list:
    """Recursively remove keys with None or empty values."""
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            v = _strip_nulls(v) if isinstance(v, (dict, list)) else v
            if v is not None and v != "" and v != [] and v != {}:
                cleaned[k] = v
        return cleaned
    if isinstance(data, list):
        return [
            _strip_nulls(item) if isinstance(item, (dict, list)) else item
            for item in data
            if item is not None and item != ""
        ]
    return data
