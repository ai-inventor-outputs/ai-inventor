"""User prompt for review_paper (Step 3.5: REVIEW_PAPER).

Adversarial review of the paper draft by a different LLM.
Mirrors review_hypo style: role in user prompt, web-grounded, scores (a)-(e).
Output: structured feedback with actionable critiques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )


_ART_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_expected_files",
}


def PROMPT(
    paper_text: str,
    artifacts_str: str,
    previous_critiques_text: str,
) -> str:
    return f"""<role>
You are a very experienced and critical conference reviewer specialized in the domain of the work under review.
You have reviewed for top-tier venues in the relevant field. Your reviews are known for
being thorough, fair, and grounded in the actual state of the field.
</role>

<paper>
{paper_text}
</paper>

{
        f'''<supplementary_materials>
The authors' code, data, and experimental artifacts. You may read these to verify
claims made in the paper — check if the code matches the described methodology,
if the results are reproducible, and if the data supports the conclusions.

{artifacts_str}
</supplementary_materials>'''
        if artifacts_str
        else ""
    }

{
        f'''<previous_review>
Your review from the previous iteration. Check which critiques have been addressed
in the revised paper. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

{previous_critiques_text}
</previous_review>'''
        if previous_critiques_text
        else ""
    }

<task>
Review this paper as you would for a top-tier venue submission.

STEP 1 — READ THE PAPER: Read it carefully. Note claims, methodology, and results.

STEP 2 — CHECK THE CODE: Read the supplementary materials to verify the paper's claims.
Do the experiments match what's described? Are there discrepancies between code and paper?

STEP 3 — SEARCH THE LITERATURE: Ground your review in evidence.
- Search for the closest existing work — is this genuinely novel or incremental?
- Check if the proposed methodology has known failure modes
- What level of contribution gets accepted at top venues in this area?

STEP 4 — WRITE YOUR REVIEW:
For each critique:
1. Categorize: methodology, evidence, novelty, clarity, scope, or rigor
2. Rate severity: major (would cause rejection) or minor (polish)
3. Describe the issue clearly
4. Suggest a concrete action to address it

Focus on the most impactful issues. Provide your review via structured output.
</task>"""


def get(
    paper_text: str,
    artifacts: list[BaseArtifact],
    previous_critiques_text: str | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for adversarial paper review."""
    from aii_lib.prompts import LLMPromptModel

    prompt = PROMPT(
        paper_text=paper_text or "No paper text available.",
        artifacts_str=LLMPromptModel.list_to_prompt_yaml(
            artifacts,
            label="Item",
            include=_ART_FIELDS,
            strip_nulls=True,
        )
        or "",
        previous_critiques_text=previous_critiques_text or "",
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()
