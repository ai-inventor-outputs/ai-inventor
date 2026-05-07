"""Workspace path prompt component.

Provides workspace path and filesystem access rules for agent prompts.
"""


def get_workspace_prompt(workspace_path: str) -> str:
    """Get workspace prompt with the exact absolute path and access rules.

    Args:
        workspace_path: Absolute path to the agent's workspace directory.
    """
    return f"""<workspace>
Your workspace: `{workspace_path}`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `{workspace_path}/`:
GOOD: `{workspace_path}/file.py`, `{workspace_path}/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>"""
