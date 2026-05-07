"""User uploads tools for OpenRouter/LLM steps.

Provides sandboxed list_user_uploads and read_user_upload tools
for LLM steps that use chat() instead of Claude agents.
"""

from collections.abc import Callable
from pathlib import Path


def _validate_path(user_folder_root: str, relative_path: str) -> Path:
    """Resolve and validate a path stays inside the user uploads folder."""
    root = Path(user_folder_root).resolve()
    target = (root / relative_path).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"Path escapes user uploads folder: {relative_path}")
    return target


def make_user_folder_tools(
    user_folder_path: str,
) -> tuple[list[dict], dict[str, Callable]]:
    """Create user uploads tool definitions and handlers.

    Returns:
        Tuple of (tool_definitions, handler_map) ready for chat().
        tool_definitions: list of OpenAI-format tool dicts
        handler_map: dict mapping tool name to callable
    """

    def list_user_uploads(path: str = "") -> str:
        target = _validate_path(user_folder_path, path)
        if not target.exists():
            return f"Path not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(target.iterdir())
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                size = e.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  {e.name}  ({size_str})")
        return (
            f"Contents of {path or '/'}:\n" + "\n".join(lines)
            if lines
            else f"Empty directory: {path or '/'}"
        )

    def read_user_upload(path: str, offset: int = 0, limit: int = 2000) -> str:
        target = _validate_path(user_folder_path, path)
        if not target.exists():
            return f"File not found: {path}"
        if target.is_dir():
            return f"Cannot read directory: {path}. Use list_user_uploads instead."
        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading {path}: {e}"
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1:>6}\t{line.rstrip()}" for i, line in enumerate(selected)]
        header = f"File: {path} (lines {offset + 1}-{offset + len(selected)} of {len(lines)})"
        return header + "\n" + "\n".join(numbered)

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "list_user_uploads",
                "description": "List files and directories in the user uploads folder. Supports nested paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path within user uploads folder (empty = root)",
                            "default": "",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_user_upload",
                "description": "Read contents of a file in the user uploads folder. Returns line-numbered content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file within user uploads folder",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (0-based)",
                            "default": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read",
                            "default": 2000,
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    handlers: dict[str, Callable] = {
        "list_user_uploads": list_user_uploads,
        "read_user_upload": read_user_upload,
    }

    return tool_defs, handlers
