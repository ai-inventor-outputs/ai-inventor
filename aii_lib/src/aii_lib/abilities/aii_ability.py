"""
@aii_ability decorator for declaring tool functions.

Usage:
    @aii_ability(
        name="aii_lean__run",
        description="Compile Lean 4 code with Mathlib",
        venv="../../../../.aii-venv",
        requirements="../../../../pyproject.toml",
        worker_init="init_run_lean",
        max_workers=4,
        timeout=180.0,
        retries=3,
    )
    def core_run_lean(code: str) -> dict:
        ...
"""

import inspect
from typing import Annotated, get_args, get_origin, get_type_hints

# Single source for the default ability per-request timeout (seconds).
# Imported by ability_client / logging_config to keep their fallbacks aligned.
DEFAULT_ABILITY_TIMEOUT = 180.0

# Global registry — all @aii_ability decorated functions register here
_registry: dict[str, dict] = {}


def get_registry() -> dict[str, dict]:
    """Return all registered tools."""
    return _registry


def aii_ability(
    name: str,
    description: str,
    venv: str | None = None,
    requirements: str | list[str] | None = None,
    worker_init: str | None = None,
    max_workers: int = 10,
    timeout: float = DEFAULT_ABILITY_TIMEOUT,
    retries: int = 3,
    check_env: str | None = None,
) -> object:
    """
    Decorator that registers a function as an AII tool.

    Args:
        name: Unique tool name (becomes HTTP endpoint name)
        description: What this tool does (used in LLM tool schemas)
        venv: Path to venv, relative to the script's location
        requirements: Path to pyproject.toml (relative to script) or list of packages
        worker_init: Name of init function in the SAME file (runs once per worker)
        max_workers: Max concurrent requests for this tool
        timeout: Per-request timeout in seconds
        retries: Number of retries on transient errors (0 = no retries, default 3)
        check_env: Path to a bash script (relative to script) that verifies
            all non-pip prerequisites are available (e.g. elan/lake for Lean,
            system binaries, API keys). Exit 0 = OK, non-zero = missing deps.
            Runs once at bootstrap time. Empty/None = no extra check needed.
    """

    def decorator(func: object) -> object:
        # Store metadata on the function
        func._aii_ability = {
            "name": name,
            "description": description,
            "venv": venv,
            "requirements": requirements,
            "worker_init": worker_init,
            "max_workers": max_workers,
            "timeout": timeout,
            "retries": retries,
            "check_env": check_env,
            # Filled at discovery time:
            "script_path": None,
            "func_name": func.__name__,
            "params": None,
        }

        # Auto-generate parameter schema from type hints
        func._aii_ability["params"] = schema_from_function(func)

        # Register globally (skip duplicates from sibling imports)
        if name in _registry:
            existing = _registry[name]
            if existing["func_name"] == func.__name__:
                return func  # Same function re-imported, skip
            raise ValueError(f"Duplicate @aii_ability name: {name}")
        _registry[name] = func._aii_ability
        _registry[name]["func"] = func

        return func

    return decorator


# =============================================================================
# Schema generation from function signature
# =============================================================================

# Python type -> JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def schema_from_function(func: object) -> dict:
    """
    Generate JSON Schema 'parameters' object from function signature.

    Supports:
    - Type annotations: str, int, float, bool, list, dict
    - Annotated[type, "description"] for per-param descriptions
    - Default values -> parameter becomes optional
    - Docstring Args section -> fallback descriptions
    - **kwargs functions (extracts from docstring only)

    Returns:
        {"type": "object", "properties": {...}, "required": [...]}
    """
    hints = {}
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        pass

    sig = inspect.signature(func)
    docstring_args = _parse_docstring_args(func.__doc__ or "")

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        # For **kwargs functions, build schema from docstring only
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return _schema_from_docstring(func.__doc__ or "")
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        hint = hints.get(param_name, str)  # default to string

        # Handle Annotated[type, "description"]
        param_desc = docstring_args.get(param_name, "")
        json_type = "string"

        if get_origin(hint) is Annotated:
            args = get_args(hint)
            actual_type = args[0]
            # Second arg is the description string
            if len(args) > 1 and isinstance(args[1], str):
                param_desc = args[1]
            json_type = _TYPE_MAP.get(actual_type, "string")
        else:
            json_type = _TYPE_MAP.get(hint, "string")

        prop = {"type": json_type}
        if param_desc:
            prop["description"] = param_desc

        properties[param_name] = prop

        # Required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """Parse 'Args:' section from Google-style docstring."""
    result = {}
    in_args = False
    current_name = None

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            # Check for section end
            if stripped and not stripped.startswith("-") and ":" in stripped:
                if any(
                    stripped.startswith(s)
                    for s in ["Returns", "Raises", "Yields", "Note", "Example"]
                ):
                    break
            # "param_name: description" or "param_name (type): description"
            if ":" in stripped:
                parts = stripped.split(":", 1)
                name_part = parts[0].strip().strip("-").strip()
                # Remove type annotation in parens
                if "(" in name_part:
                    name_part = name_part[: name_part.index("(")].strip()
                desc = parts[1].strip()
                if name_part and " " not in name_part:
                    current_name = name_part
                    result[current_name] = desc
            elif current_name and stripped:
                # Continuation line
                result[current_name] += " " + stripped

    return result


def _schema_from_docstring(docstring: str) -> dict:
    """Build schema entirely from docstring Args section (for **kwargs functions)."""
    args = _parse_docstring_args(docstring)
    properties = {}
    required = []

    for name, desc in args.items():
        properties[name] = {"type": "string", "description": desc}
        required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# =============================================================================
# OpenAI tool schema conversion
# =============================================================================


def ability_to_openai_tool(name: str) -> dict:
    """Convert a registered ability to OpenAI/OpenRouter tool schema format.

    Args:
        name: Registered ability name (e.g. "aii_web_tools__search")

    Returns:
        OpenAI-compatible tool definition:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    if name not in _registry:
        raise KeyError(f"Ability '{name}' not registered")
    entry = _registry[name]
    return {
        "type": "function",
        "function": {
            "name": entry["name"],
            "description": entry["description"],
            "parameters": entry["params"],
        },
    }


def abilities_to_openai_tools(names: list[str]) -> list[dict]:
    """Convert multiple registered abilities to OpenAI/OpenRouter tool schema format.

    Args:
        names: List of registered ability names

    Returns:
        List of OpenAI-compatible tool definitions
    """
    return [ability_to_openai_tool(n) for n in names]
