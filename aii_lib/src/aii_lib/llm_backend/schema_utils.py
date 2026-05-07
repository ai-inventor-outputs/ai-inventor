"""Shared utilities for LLM backends.

- Schema helpers: prepare Pydantic schemas for strict structured output mode.
- Tool cost calculation: compute per-tool costs for summary metrics.
"""

from __future__ import annotations

from ..abilities.endpoint_names import AII_WEB_SEARCH


def add_additional_properties_false(schema: dict) -> dict:
    """Add additionalProperties: false to all objects in schema."""
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    if result.get("type") == "object":
        result["additionalProperties"] = False

    if "properties" in result:
        new_props = {}
        for key, value in result["properties"].items():
            new_props[key] = add_additional_properties_false(value)
        result["properties"] = new_props

    if "items" in result:
        result["items"] = add_additional_properties_false(result["items"])

    if "$defs" in result:
        new_defs = {}
        for key, value in result["$defs"].items():
            new_defs[key] = add_additional_properties_false(value)
        result["$defs"] = new_defs

    return result


def make_all_fields_required(schema: dict) -> dict:
    """Make all properties required in schema (strict mode requirement).

    Structured output requires ALL fields in 'properties' to also be in
    'required'. This recursively fixes Pydantic schemas where fields with
    defaults are marked as optional.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # If this object has properties, make them ALL required
    if result.get("type") == "object" and "properties" in result:
        result["required"] = list(result["properties"].keys())
        # Recursively process nested properties
        new_props = {}
        for key, value in result["properties"].items():
            new_props[key] = make_all_fields_required(value)
        result["properties"] = new_props

    # Process array items
    if "items" in result:
        result["items"] = make_all_fields_required(result["items"])

    # Process $defs (nested type definitions)
    if "$defs" in result:
        new_defs = {}
        for key, value in result["$defs"].items():
            new_defs[key] = make_all_fields_required(value)
        result["$defs"] = new_defs

    return result


# =========================================================================
# Tool cost calculation
# =========================================================================

# Per-call pricing for tools that have a cost
TOOL_PRICING: dict[str, float] = {
    AII_WEB_SEARCH: 0.001,  # $0.001/call
}


def calculate_tool_costs(tool_calls: dict[str, int]) -> tuple[dict, float]:
    """Calculate per-tool costs from a tool_name -> call_count mapping.

    Returns:
        (tool_costs_dict, total_tool_cost)
        where tool_costs_dict maps tool_name -> {"count", "unit", "total"}
        for tools with non-zero cost.
    """
    tool_costs: dict[str, dict] = {}
    total = 0.0
    for tool_name, count in tool_calls.items():
        unit_cost = TOOL_PRICING.get(tool_name, 0.0)
        tool_total = count * unit_cost
        total += tool_total
        if unit_cost > 0:
            tool_costs[tool_name] = {
                "count": count,
                "unit": unit_cost,
                "total": tool_total,
            }
    return tool_costs, total
