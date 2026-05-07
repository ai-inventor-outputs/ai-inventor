"""User prompt for upd_hypo (Step 3.6: UPD_HYPO).

Same walkthrough as gen_paper_text + the newly generated paper text.
Revises the hypothesis based on all evidence gathered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt
from aii_pipeline.utils import to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )


_ALL_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_expected_files",
    "in_dependencies",
}
_NEW_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_expected_files",
    "in_dependencies",
}


def PROMPT(
    hypothesis_text: str,
    all_artifacts_str: str,
    new_artifacts_str: str,
    new_artifact_count: int,
    paper_text: str,
    reviewer_feedback_text: str,
) -> str:
    return f"""<current_hypothesis>
The hypothesis as it stands. Revise it based on the evidence below.

{hypothesis_text}
</current_hypothesis>

<all_artifacts>
Complete set of research artifacts across all iterations.

{all_artifacts_str}
</all_artifacts>

{
        f'''<new_artifacts_this_iteration>
These {new_artifact_count} artifacts were created THIS iteration.

{new_artifacts_str}
</new_artifacts_this_iteration>

'''
        if new_artifacts_str and new_artifact_count > 0
        else ""
    }<current_paper>
The paper draft from this iteration — represents the current state of the research story.

{paper_text}
</current_paper>

{
        f'''<reviewer_feedback>
Feedback from the paper reviewer this iteration.

{reviewer_feedback_text}
</reviewer_feedback>

'''
        if reviewer_feedback_text
        else ""
    }

<task>
IMPORTANT: Your ONLY output is the revised hypothesis text. Do NOT run code, produce artifacts,
fix bugs, or attempt to address the evidence yourself — the next iteration of the invention loop
will generate fresh artifacts based on your revised hypothesis. Reflect and rewrite; nothing else.

Do NOT generate a completely new hypothesis. Take the current hypothesis and REVISE it
to incorporate new evidence. Keep the core idea — refine, narrow, or strengthen it.

1. Does the evidence support the hypothesis? Narrow or broaden scope as needed.
2. Which claims now have strong evidence? Which are still unsupported?
3. Should the hypothesis become more specific based on what we've learned?
4. If reviewer feedback is provided, address the critiques directly.

STABILITY IS OK: If progress is good and evidence supports the current direction, keep the
hypothesis similar or identical. Only make substantive changes when evidence clearly calls for
them — e.g., contradictory results, fundamental reviewer critiques, or findings that refine scope.

You must also classify two kinds of edges in the research trace:

(A) The H↔H edge — how does this revised hypothesis relate to the previous one?
    Set `relation_type` (Moulines's structuralist typology) to one of:
    - "evolution": refining specialised claims, same conceptual frame
    - "embedding": previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian shift)
    Set `relation_rationale` to a brief justification (≤50 chars).

(B) The A↔A edges — for each artifact created THIS iteration, classify each of its
    `in_dependencies` (predecessor → dependent) using MultiCite's citation-function
    typology (Lauscher et al., NAACL 2022) — emit one entry in `artifact_relations`
    per (predecessor, dependent) pair:
    - "background": predecessor is treated as background context
    - "motivation": predecessor motivated this artifact's research
    - "uses": this artifact uses the predecessor's data, method, or output
    - "extends": this artifact extends the predecessor
    - "similarities": this artifact's results agree with the predecessor's
    - "differences": this artifact's results disagree with the predecessor's
    Each `relation_rationale` must be ≤50 characters.

Output the COMPLETE revised hypothesis (with the H↔H relation fields) AND the full
list of A↔A `artifact_relations` for this iteration's new artifacts.
</task>"""


def get(
    hypothesis: dict,
    artifacts: list[BaseArtifact],
    current_iter_artifacts: list[BaseArtifact],
    paper_text: str,
    iteration: int,
    reviewer_feedback_text: str | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for hypothesis revision."""
    from aii_lib.prompts import LLMPromptModel

    hypo_display = {
        k: v
        for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    all_arts = (
        LLMPromptModel.list_to_prompt_yaml(
            artifacts,
            label="Item",
            include=_ALL_FIELDS,
            strip_nulls=True,
        )
        or "No artifacts yet."
    )

    if current_iter_artifacts:
        new_arts = "\n\n".join(
            a.to_prompt_yaml(include=_NEW_FIELDS, strip_nulls=True) for a in current_iter_artifacts
        )
    else:
        new_arts = ""

    prompt = PROMPT(
        hypothesis_text=to_prompt_yaml(hypo_display),
        all_artifacts_str=all_arts,
        new_artifacts_str=new_arts,
        new_artifact_count=len(current_iter_artifacts),
        paper_text=paper_text or "No paper text available.",
        reviewer_feedback_text=reviewer_feedback_text or "",
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()
