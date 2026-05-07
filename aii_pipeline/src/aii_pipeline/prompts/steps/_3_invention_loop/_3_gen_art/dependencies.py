"""Shared dependency prompt builder for all gen_art artifact types.

Builds the <dependencies> prompt section by filtering the artifact list
by dep IDs and YAML-formatting via ``LLMPromptModel.list_to_prompt_yaml``,
so downstream executor agents see id, type, title, summary, workspace_path,
and out_dependency_files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.prompts import LLMPromptModel
from aii_pipeline.prompts.components.data_files import get_reading_mini_preview_full

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )

# Fields to include in the dependency prompt for each artifact.
# These are the fields the downstream executor agent needs to see.
DEPENDENCY_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_dependency_files",
}


def build_dependencies_prompt(
    artifacts: list[BaseArtifact],
    dependency_ids: list[str],
) -> str:
    """Build the <dependencies> prompt section from the artifact list.

    Args:
        artifacts: The full artifact list (typically ``inv.get_artifacts()``).
        dependency_ids: Artifact IDs to include as dependencies.

    Returns:
        Formatted <dependencies>...</dependencies> block, or empty string if none.
    """
    if not dependency_ids:
        return ""

    dep_set = set(dependency_ids)
    deps = [a for a in artifacts if a.id in dep_set]
    if not deps:
        return ""

    content = LLMPromptModel.list_to_prompt_yaml(
        deps,
        label="Dependency",
        include=DEPENDENCY_FIELDS,
        strip_nulls=True,
    )
    if not content:
        return ""

    return f"""<dependencies>
Read the files in these dependency workspaces to understand what's available, then copy any you need into your working directory.

{content}

{get_reading_mini_preview_full()}
</dependencies>"""
