"""User prompt for plan generation (GEN_PLAN).

Expands a single artifact_direction into a detailed plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.prompts.components.artifact_planning import get_artifact_planning
from aii_pipeline.prompts.components.artifact_scope import get_artifact_scope
from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt
from aii_pipeline.utils import LLMPromptModel, to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._1_gen_strat.out_schema import (
        ArtifactDirection,
    )
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

_PLANNING_ONLY = """YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run Bash commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations."""


def PROMPT(
    hypothesis_text: str,
    artifact_direction_text: str,
    dependencies_text: str,
    artifact_planning_guidance: str,
    artifact_scope_text: str,
    compute_profiles_text: str = "",
) -> str:
    return f"""<hypothesis>
{hypothesis_text}
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

{artifact_direction_text}
</artifact_direction>

{
        f'''<dependencies>
Completed artifacts this artifact can use during execution.

{dependencies_text}
</dependencies>'''
        if dependencies_text
        else ""
    }

<instructions>
{_PLANNING_ONLY}

{artifact_scope_text}

{artifact_planning_guidance}

{
        (
            "<compute_profiles>"
            + chr(10)
            + "Choose the compute profile this artifact needs for execution."
            + chr(10)
            + compute_profiles_text
            + chr(10)
            + "</compute_profiles>"
        )
        if compute_profiles_text
        else ""
    }
GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions>"""


# =============================================================================
# HELPERS
# =============================================================================


def _format_hypothesis(hypo: dict) -> str:
    """Format hypothesis dict as YAML for LLM readability."""
    return to_prompt_yaml(hypo)


def _format_artifact_direction(direction: ArtifactDirection) -> str:
    """Format artifact direction as YAML for LLM readability."""
    return direction.to_prompt_yaml()


_PLAN_DEPENDENCY_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "out_dependency_files",
    "workspace_path",
    "out_expected_files",
}


def _format_dependencies_context(
    direction: ArtifactDirection,
    artifacts: list[BaseArtifact],
) -> str:
    """Format dependency artifacts as YAML for LLM readability."""
    if not direction.depends_on:
        return ""
    dep_set = {d.id for d in direction.depends_on}
    deps = [a for a in artifacts if a.id in dep_set]
    if not deps:
        return ""
    return LLMPromptModel.list_to_prompt_yaml(
        deps,
        label="Dependency",
        include=_PLAN_DEPENDENCY_FIELDS,
        strip_nulls=True,
    )


# =============================================================================
# EXPORTS
# =============================================================================


def _format_compute_profiles(
    compute_profiles: dict,
    artifact_type: str,
    artifact_type_profiles: dict[str, list[str]],
) -> str:
    """Format available compute profiles for the prompt.

    Shows tier name + hardware description so the LLM can make an informed choice.
    """
    allowed = artifact_type_profiles.get(artifact_type, list(compute_profiles.keys()))
    if not allowed:
        return ""

    lines = [f"Available profiles for {artifact_type} artifacts:"]
    for name in allowed:
        profile = compute_profiles.get(name)
        if profile is None:
            continue
        desc = profile.description if hasattr(profile, "description") else str(profile)
        line = f"  - {name}: {desc}"
        fb_desc = getattr(profile, "fallback_description", "")
        if fb_desc:
            line += f" (fallback: {fb_desc})"
        lines.append(line)

    lines.append("")
    lines.append("Set runpod_compute_profile to one of these exact tier names.")
    return "\n".join(lines)


def get(
    hypothesis: dict,
    artifacts: list[BaseArtifact],
    artifact_direction: ArtifactDirection,
    compute_profiles: dict | None = None,
    artifact_type_profiles: dict[str, list[str]] | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for expanding an artifact direction into a plan.

    Args:
        hypothesis: Hypothesis dict.
        artifacts: List of existing artifacts.
        artifact_direction: Direction to expand into a plan.
        compute_profiles: Named compute profiles from config (name -> ComputeProfileConfig).
        artifact_type_profiles: Map of artifact_type -> allowed profile tier names.
        user_folder_path: Absolute path to user data folder (empty = no folder).
    """
    hypo_filtered = {
        k: v
        for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    artifact_type = artifact_direction.type

    profiles_text = ""
    if compute_profiles and artifact_type_profiles:
        profiles_text = _format_compute_profiles(
            compute_profiles, artifact_type, artifact_type_profiles
        )

    prompt = PROMPT(
        hypothesis_text=_format_hypothesis(hypo_filtered),
        artifact_direction_text=_format_artifact_direction(artifact_direction),
        dependencies_text=_format_dependencies_context(artifact_direction, artifacts),
        artifact_planning_guidance=get_artifact_planning([artifact_type]),
        artifact_scope_text=get_artifact_scope([artifact_type]),
        compute_profiles_text=profiles_text,
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()
