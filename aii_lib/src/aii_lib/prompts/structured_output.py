"""LLMStructOutModel — Pydantic BaseModel for LLM structured output schemas.

Inherit from LLMStructOutModel instead of BaseModel for any schema that the LLM
returns as JSON structured output. Mark fields with LLMStructOut to control which
appear in the JSON schema:

    class Strategy(LLMStructOutModel):
        title: Annotated[str, LLMStructOut] = Field(...)    # in schema
        objective: Annotated[str, LLMStructOut] = Field(...) # in schema
        id: str = Field(...)                             # excluded from schema

Every LLMStructOutModel must annotate its fields — only marked fields are
included. Unmarked fields are excluded.

Nested models in $defs are also filtered — unmarked fields are stripped
recursively so that e.g. a code-assigned ``id`` on a nested model never
leaks into the JSON schema sent to the LLM.

Usage:
    output_format=Strategies.to_struct_output()
    output_format=Strategy.to_struct_output(include={"title"})
"""

from __future__ import annotations

from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .annotations import LLMStructOut
from .prompt_serializable import LLMPromptModel, _get_marked_fields


class LLMStructOutModel(BaseModel):
    """BaseModel for schemas used as LLM structured output.

    Mark fields with Annotated[type, LLMStructOut] to include them in the schema.
    Unmarked fields are excluded.
    """

    @classmethod
    def to_struct_output(
        cls,
        *,
        include: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build the output_format dict for Claude Agent SDK.

        Args:
            include: Only include these fields. Overrides LLMStructOut markers.

        Returns:
            {"type": "json_schema", "schema": <json_schema_dict>}
            Ready to pass to AgentOptions(output_format=...).
        """
        schema = cls.model_json_schema()
        fields = include if include is not None else _get_marked_fields(cls, LLMStructOut)
        nested_filters = _collect_nested_filters(cls)
        schema = _filter_schema(schema, fields, nested_filters=nested_filters)
        return {"type": "json_schema", "schema": schema}


class BaseExpectedFiles(LLMPromptModel, LLMStructOutModel):
    """Base class for per-type expected file specifications.

    All fields must resolve to file path strings:
    - ``str``: single file path
    - ``list[str]``: multiple file paths
    - ``BaseExpectedFiles`` subclass: nested file group
    - ``list[BaseExpectedFiles]``: multiple nested file groups

    Subclasses should be pure data — no custom methods.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, field_info in cls.model_fields.items():
            if not _is_valid_expected_file_type(field_info.annotation):
                raise TypeError(
                    f"{cls.__name__}.{name}: expected file path type "
                    f"(str, list[str], or BaseExpectedFiles subclass), got {field_info.annotation}"
                )


def _is_valid_expected_file_type(ann: Any) -> bool:
    """Check if a type annotation resolves to file paths.

    Valid types: str, list[str], BaseExpectedFiles subclass, list[BaseExpectedFiles].
    Annotated wrappers are stripped automatically.
    """
    # Unwrap Annotated[T, ...]
    if hasattr(ann, "__metadata__"):
        ann = get_args(ann)[0]

    if ann is str:
        return True

    origin = get_origin(ann)
    if origin is list:
        args = get_args(ann)
        return bool(args) and _is_valid_expected_file_type(args[0])

    return bool(isinstance(ann, type) and issubclass(ann, BaseExpectedFiles))


# ---------------------------------------------------------------------------
# Nested model discovery
# ---------------------------------------------------------------------------


def _collect_nested_filters(cls: type[BaseModel]) -> dict[str, frozenset[str]]:
    """Walk the model type tree and collect LLMStructOut-marked fields for every nested model.

    Returns a mapping of ``{ClassName: frozenset_of_marked_field_names}`` used
    by ``_filter_schema`` to strip unmarked fields from ``$defs``.
    """
    filters: dict[str, frozenset[str]] = {}
    seen: set[type] = set()

    def _extract_models(annotation: Any) -> list[type[BaseModel]]:
        """Extract BaseModel subclasses from a type annotation."""
        models: list[type[BaseModel]] = []
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            models.append(annotation)
            return models
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                models.append(arg)
            elif get_args(arg):
                models.extend(_extract_models(arg))
        return models

    def _walk(model_cls: type[BaseModel]) -> None:
        if model_cls in seen:
            return
        seen.add(model_cls)
        marked = _get_marked_fields(model_cls, LLMStructOut)
        if marked:
            filters[model_cls.__name__] = marked
        for field_info in model_cls.model_fields.values():
            for nested in _extract_models(field_info.annotation):
                _walk(nested)

    _walk(cls)
    return filters


# ---------------------------------------------------------------------------
# Schema filtering
# ---------------------------------------------------------------------------


def _filter_schema(
    schema: dict,
    fields: set[str] | frozenset[str],
    *,
    nested_filters: dict[str, frozenset[str]] | None = None,
) -> dict:
    """Filter a JSON schema to only include specified top-level properties.

    When *nested_filters* is provided, ``$defs`` entries whose name matches
    a key in the mapping also have their properties stripped to only the
    allowed set.  This prevents unmarked fields (e.g. code-assigned ``id``)
    from leaking into the schema for nested models.
    """
    schema = schema.copy()

    if "properties" in schema:
        schema["properties"] = {k: v for k, v in schema["properties"].items() if k in fields}

    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r in fields]

    # Filter nested $defs based on annotation markers
    if "$defs" in schema and nested_filters:
        new_defs = {}
        for def_name, def_schema in schema["$defs"].items():
            if def_name in nested_filters:
                allowed = nested_filters[def_name]
                def_schema = def_schema.copy()
                if "properties" in def_schema:
                    def_schema["properties"] = {
                        k: v for k, v in def_schema["properties"].items() if k in allowed
                    }
                if "required" in def_schema:
                    def_schema["required"] = [r for r in def_schema["required"] if r in allowed]
            new_defs[def_name] = def_schema
        schema["$defs"] = new_defs

    # Clean up $defs — keep only those transitively referenced
    if "$defs" in schema:
        import json

        # Seed: refs from top-level properties
        kept: set[str] = set()
        frontier = json.dumps(schema.get("properties", {}))
        # Iterate until no new $defs are discovered
        while True:
            added = {
                k for k in schema["$defs"] if k not in kept and f'"$ref": "#/$defs/{k}"' in frontier
            }
            if not added:
                break
            kept |= added
            frontier = json.dumps({k: schema["$defs"][k] for k in added})
        schema["$defs"] = {k: v for k, v in schema["$defs"].items() if k in kept}
        if not schema["$defs"]:
            del schema["$defs"]

    return schema
