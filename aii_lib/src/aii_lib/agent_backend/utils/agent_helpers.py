"""Module-level helpers for pre/post agent setup and finalization.

These were previously bundled into ``AgentInitializer`` and ``AgentFinalizer``
classes that held no real state — they only stashed task_id/task_name from
contextvars on ``self`` at construction time. This module exposes the same
behavior as plain functions that resolve telemetry + task at call time, so
context changes between configuration and execution are reflected correctly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aii_lib.run import emit

if TYPE_CHECKING:
    from collections.abc import Callable

    from aii_lib.agent_backend import AgentOptions

# Default file size limit (GitHub's 100MB limit)
MAX_FILE_SIZE_MB = 100

# Directories excluded from GitHub deployment.
# Used by both the oversized-file check and the deploy-to-repo step.
# Keep this as the single source of truth — import from here, don't duplicate.
EXCLUDED_WORKSPACE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".cache",
    "node_modules",
    "temp",
    "tmp",
    "dependencies",
    ".claude",
}


# === Internal telemetry helper ===


def _emit(message: str, level: str = "INFO") -> None:
    """Route a status message into the active workflow's journal.

    Routes the message to the matching ``emit.status_*`` emitter based
    on ``level``. Helpers in this module pass static level strings, so
    the dispatch is one-shot per call. Outside a DBOS workflow context
    the underlying ``journal_event_step`` is a no-op.
    """
    method = {
        "ERROR": emit.status_public_error,
        "WARNING": emit.status_public_warning,
        "WARN": emit.status_public_warning,
        "SUCCESS": emit.status_public_success,
        "INFO": emit.status_private_info,
    }.get(level.upper(), emit.status_private_info)
    method(message)


# === Workspace setup ===


def setup_workspace(
    workspace_dir: Path,
    template_dir: Path | None = None,
) -> Path:
    """Set up the agent workspace directory.

    Idempotent: existing workspaces are reused (common after fork/resume),
    creates directory if needed, copies template if provided.
    """
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if template_dir and Path(template_dir).exists():
        for item in Path(template_dir).iterdir():
            dest = workspace_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    return workspace_dir


# === Dependency copying / prompts ===


def _read_artifact_title(workspace_dir: Path) -> str:
    """Read title from artifact_title.txt."""
    title_path = workspace_dir / "artifact_title.txt"
    if title_path.exists():
        return title_path.read_text(encoding="utf-8").strip()
    return ""


def _read_artifact_summary(workspace_dir: Path) -> str:
    """Read summary from artifact_metadata.json."""
    metadata_path = workspace_dir / "artifact_metadata.json"
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return data.get("summary", "")
        except (OSError, json.JSONDecodeError) as e:
            _emit(f"Failed to read artifact metadata from {metadata_path}: {e}", "ERROR")
            return ""
    return ""


def copy_dependencies(
    dependencies: list[Any],
    workspace_dir: Path,
    get_workspace_path: Callable[[Any], str | None] | None = None,
    get_id: Callable[[Any], str] | None = None,
    get_type: Callable[[Any], str] | None = None,
    get_title: Callable[[Any], str] | None = None,
    get_summary: Callable[[Any], str] | None = None,
) -> list[dict]:
    """Copy dependency artifact workspaces into the current workspace.

    Returns a list of dependency info dicts (id, type, title, summary, local_path).
    """
    deps_dir = Path(workspace_dir) / "dependencies"
    deps_dir.mkdir(parents=True, exist_ok=True)

    copied_deps = []
    for artifact in dependencies:
        if get_id:
            artifact_id = get_id(artifact)
        else:
            artifact_id = getattr(artifact, "id", "unknown")

        if get_workspace_path:
            source_path = get_workspace_path(artifact)
        else:
            source_path = (
                artifact.result.get("workspace_path") if hasattr(artifact, "result") else None
            )

        if not source_path:
            _emit(f"Skipping dependency {artifact_id}: no workspace path", "WARNING")
            continue

        source_workspace = Path(source_path)
        if not source_workspace.exists():
            _emit(f"Skipping dependency: workspace not found at {source_path}", "WARNING")
            continue

        if get_title:
            title = get_title(artifact)
        else:
            title = _read_artifact_title(source_workspace) or artifact_id

        # Use artifact ID for folder name (not title) to avoid long sanitized names
        folder_name = artifact_id

        target_dir = deps_dir / folder_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_workspace, target_dir)

        if get_summary:
            summary = get_summary(artifact)
        else:
            summary = _read_artifact_summary(source_workspace)

        if get_type:
            artifact_type = get_type(artifact)
        else:
            artifact_type = getattr(artifact, "type", None)
            if hasattr(artifact_type, "value"):
                artifact_type = artifact_type.value

        copied_deps.append(
            {
                "id": artifact_id,
                "type": str(artifact_type) if artifact_type else "unknown",
                "title": title,
                "summary": summary,
                "local_path": f"./dependencies/{folder_name}",
                "is_research": str(artifact_type).lower() == "research" if artifact_type else False,
            }
        )

    _emit(f"Copied {len(copied_deps)} dependencies to {deps_dir}")
    return copied_deps


def gen_dependency_prompt(copied_deps: list[dict]) -> str:
    """Generate markdown prompt section from copied dependencies."""
    if not copied_deps:
        return ""

    lines = [
        "## Dependencies\n",
        "Each path below is a **folder** containing the artifact's workspace.",
        "Use `ls <folder>/` to explore contents and find data files.\n",
    ]
    for dep in copied_deps:
        dep_type = dep.get("type", "unknown").upper()
        title = dep.get("title", dep.get("id", "unknown"))
        lines.append(f"### [{dep.get('id', 'unknown')}] {dep_type}: {title}")
        lines.append(f"**Folder:** `{dep.get('local_path', 'unknown')}/`")

        if dep.get("is_research") and dep.get("summary"):
            lines.append(f"\n{dep['summary']}")

        lines.append("")  # Blank line between deps

    return "\n".join(lines)


# === Server health checks ===


def ensure_servers(servers: list[dict] | None = None) -> dict[str, bool]:
    """Check if the ability server is reachable.

    Returns:
        Dict of {"ability_server": is_available}
    """
    from aii_lib.utils import server_available

    if not servers:
        return {}

    available = server_available(timeout=3.0)
    if available:
        _emit("Ability server is reachable")
    else:
        _emit("Ability server not reachable", "WARNING")
    return {"ability_server": available}


# === AgentOptions builder ===


def build_options(
    agent_cfg: Any,
    workspace_dir: Path,
    *,
    task_id: str,
    task_name: str | None = None,
    system_prompt: str = "",
    output_format: dict | None = None,
    continue_seq_item: bool = True,
    expected_files_field: str | None = None,
    verify_retries: int = 2,
    **overrides,
) -> AgentOptions:
    """Build AgentOptions from any claude_agent config block.

    Reads all standard fields (model, effort, timeouts, retries, tools)
    from agent_cfg. Caller supplies the per-task identity (``task_id``,
    optional ``task_name``) and the unique parts (system_prompt,
    output_format, etc.).
    """
    from aii_lib.agent_backend import AgentOptions

    opts = {
        "llm_backend": getattr(agent_cfg, "llm_backend", "claude_max"),
        "model": agent_cfg.model,
        "effort": agent_cfg.effort,
        "max_turns": agent_cfg.max_turns,
        "agent_timeout": agent_cfg.agent_timeout,
        "agent_retries": agent_cfg.agent_retries,
        "seq_prompt_timeout": agent_cfg.seq_prompt_timeout,
        "seq_prompt_retries": agent_cfg.seq_prompt_retries,
        "message_timeout": agent_cfg.message_timeout,
        "message_retries": agent_cfg.message_retries,
        "cwd": str(workspace_dir),
        "system_prompt": system_prompt,
        "continue_seq_item": continue_seq_item,
        "allowed_tools": getattr(agent_cfg, "allowed_tools", []),
        "disallowed_tools": getattr(agent_cfg, "disallowed_tools", []),
        "run_id": task_id,
        "agent_context": task_name or task_id,
        "output_format": output_format,
        "setting_sources": ["project"],
    }
    if expected_files_field:
        opts["expected_files_struct_out_field"] = expected_files_field
        opts["max_expected_files_retries"] = verify_retries
    opts.update(overrides)
    return AgentOptions(**opts)


# === Task lifecycle ===


def start_task(name: str, parent_module_id: str) -> str:
    """Register task start with the live Run aggregate.

    ``name`` is the role label (``"gen_full_paper"``,
    ``"gen_art_demo"``); ``parent_module_id`` is the owning
    Module's node_id (caller has it in scope from its preceding
    ``start_*_module`` call).

    Returns the auto-generated task node_id (``f"{name}_{random}"``).
    """
    return emit.start_task(name=name, parent_module_id=parent_module_id)


def end_task(
    task_id: str,
    task_name: str,
    status: str,
    *,
    text: str | None = None,
    **metadata,
) -> None:
    """Signal task completion to the live Run aggregate.

    Args:
        task_id: Task identifier.
        task_name: Display name of the task.
        status: Canonical status — must be ``"done"``, ``"failed"``, or
            ``"stopped"``. The convenience helpers below pass the right
            token; direct callers should do the same.
        text: Optional human-readable description (e.g. ``"Failed: foo"``,
            ``"Timeout (30s)"``). Displayed by the FE; orthogonal to
            ``status``.
        **metadata: Additional fields forwarded to ``Run.end_task``.
    """
    emit.end_task(
        task_id,
        name=task_name,
        status=status,
        text=text,
        **metadata,
    )


def end_task_success(
    task_id: str,
    task_name: str,
    **metadata,
) -> None:
    """Convenience function for successful task completion."""
    end_task(task_id, task_name, "done", text="Success", **metadata)


def end_task_failure(
    task_id: str,
    task_name: str,
    error: str,
    **metadata,
) -> None:
    """Convenience function for failed task completion."""
    short = error[:50] if len(error) > 50 else error
    end_task(task_id, task_name, "failed", text=f"Failed: {short}", **metadata)


def end_task_timeout(
    task_id: str,
    task_name: str,
    timeout_seconds: int | None,
    **metadata,
) -> None:
    """Convenience function for timeout task completion."""
    label = f"Timeout ({timeout_seconds}s)" if timeout_seconds else "Timeout"
    end_task(task_id, task_name, "failed", text=label, **metadata)


def end_task_error(
    task_id: str,
    task_name: str,
    error: str,
    **metadata,
) -> None:
    """Convenience function for error task completion."""
    short = error[:50] if len(error) > 50 else error
    end_task(task_id, task_name, "failed", text=f"Error: {short}", **metadata)


# === File size checks ===


def check_oversized_files(
    workspace_dir: Path,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> list[dict]:
    """Check for files exceeding size limit in workspace.

    Returns list of dicts with oversized file info: [{"path": str, "size_mb": float}, ...]
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    oversized = []

    workspace = Path(workspace_dir)
    if not workspace.exists():
        return []

    for filepath in workspace.rglob("*"):
        try:
            rel_path = filepath.relative_to(workspace)
            if any(part in EXCLUDED_WORKSPACE_DIRS for part in rel_path.parts):
                continue
        except ValueError:
            continue

        if filepath.is_file():
            size = filepath.stat().st_size
            if size > max_size_bytes:
                size_mb = size / (1024 * 1024)
                oversized.append(
                    {
                        "path": str(rel_path),
                        "size_mb": round(size_mb, 2),
                    }
                )

    oversized.sort(key=lambda x: x["size_mb"], reverse=True)
    return oversized


def get_oversized_files_prompt(
    oversized_files: list[dict],
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> str:
    """Generate a prompt to tell the agent to reduce file sizes."""
    files_list = "\n".join(f"  - {f['path']} ({f['size_mb']:.1f} MB)" for f in oversized_files)

    return f"""<CRITICAL_ERROR>
Some files in your workspace exceed the {max_size_mb}MB size limit for GitHub deployment.

OVERSIZED FILES:
{files_list}

You MUST reduce these files to under {max_size_mb}MB each. Use ONE of these strategies:

=== STRATEGY 1: SPLIT FILES (PREFERRED) ===
Split large files into smaller parts and update code to read them sequentially.

For data files (JSON, JSONL, CSV, Parquet):
1. Split the file into parts under {max_size_mb}MB each:
   - data.jsonl -> data_part_001.jsonl, data_part_002.jsonl, ...
2. Update ALL code that reads this file to handle the split parts
3. Delete the original large file after splitting

=== STRATEGY 2: COMPRESSION (FALLBACK) ===
Only use if splitting is not feasible (e.g., binary files, model weights).

1. Compress the file with gzip
2. Update ALL code to decompress before use
3. Delete the original uncompressed file

=== REQUIRED: UPDATE AND TEST CODE ===
After applying your chosen strategy, you MUST:

1. Find ALL code files that reference the modified files (use grep/search)
2. Update each file to work with the new format (split parts or compressed)
3. Run the updated code to verify it still works correctly
4. Fix any errors that occur until the code runs successfully

Do NOT skip testing - the code must actually execute without errors.

Start by listing the oversized files with `ls -lh`, then apply the appropriate strategy.
</CRITICAL_ERROR>"""


# === Metadata reading ===


def read_metadata(workspace_dir: Path) -> dict[str, str]:
    """Read metadata from artifact_metadata.json.

    Returns dict with 'summary' and 'title' keys (empty strings if not found).
    """
    workspace_dir = Path(workspace_dir)
    metadata_path = workspace_dir / "artifact_metadata.json"

    result = {
        "summary": "",
        "title": "",
    }

    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            result["summary"] = data.get("summary", "")
            result["title"] = data.get("title", "")
        except (OSError, json.JSONDecodeError) as e:
            _emit(f"Failed to read metadata: {e}", "ERROR")
            raise RuntimeError(f"Failed to read artifact_metadata.json: {e}") from e

    return result


# === Requirements generation ===


def generate_requirements(
    workspace_dir: Path,
    output_file: str = "requirements.txt",
    venv_path: Path | str | None = None,
) -> Path | None:
    """Generate requirements.txt from venv using uv pip freeze.

    Returns path to generated requirements.txt, or None if failed.
    """
    workspace_dir = Path(workspace_dir)
    venv = Path(venv_path) if venv_path else workspace_dir / ".venv"

    if not venv.exists():
        _emit(f"No venv found at {venv}", "WARNING")
        return None

    output_path = workspace_dir / output_file

    try:
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
            env={
                **subprocess.os.environ,
                "VIRTUAL_ENV": str(venv),
            },
        )

        if result.returncode != 0:
            _emit(f"uv pip freeze failed: {result.stderr}", "ERROR")
            return None

        output_path.write_text(result.stdout, encoding="utf-8")
        _emit(f"Generated {output_file} with {len(result.stdout.splitlines())} packages")
        return output_path

    except FileNotFoundError:
        _emit("uv not found, falling back to pip freeze", "WARNING")
        pip_path = venv / "bin" / "pip"
        if not pip_path.exists():
            _emit(f"pip not found at {pip_path}", "ERROR")
            return None

        result = subprocess.run(
            [str(pip_path), "freeze"],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
        )

        if result.returncode != 0:
            _emit(f"pip freeze failed: {result.stderr}", "ERROR")
            return None

        output_path.write_text(result.stdout, encoding="utf-8")
        _emit(f"Generated {output_file} with {len(result.stdout.splitlines())} packages")
        return output_path

    except Exception as e:
        _emit(f"Failed to generate requirements: {e}", "ERROR")
        raise


# === Validators ===


def chain_validators(*validators: Any) -> Callable:
    """Chain multiple post_validate functions into one.

    Runs each validator in order. Returns the first failure.
    If all pass, returns (True, None).
    """

    def chained(structured_output: Any) -> tuple[bool, str | None]:
        for v in validators:
            if v is None:
                continue
            valid, retry_prompt = v(structured_output)
            if not valid:
                return False, retry_prompt
        return True, None

    return chained


def make_file_size_validator(
    workspace_dir: Path,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> Callable:
    """Create a post_validate closure for file size checks.

    Returns a function compatible with ``AgentOptions.post_validate``.
    Checks workspace for oversized files and returns a retry prompt.
    """
    workspace_dir = Path(workspace_dir)

    def validate(structured_output: Any) -> tuple[bool, str | None]:
        oversized = check_oversized_files(workspace_dir, max_size_mb)
        if not oversized:
            return True, None
        return False, get_oversized_files_prompt(oversized, max_size_mb)

    return validate
