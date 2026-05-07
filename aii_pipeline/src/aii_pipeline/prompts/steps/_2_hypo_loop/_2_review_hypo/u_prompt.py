"""User prompts for review_hypo — hypothesis review before invention loop.

Review only (no revision). Feedback goes back to gen_hypo at the loop level.
Mirrors the structure of review_paper (Step 3.5) but operates on hypotheses, not papers.
"""

from __future__ import annotations

from aii_pipeline.prompts.components.user_folder import get_user_folder_prompt
from aii_pipeline.prompts.components.user_request import get_user_request_prompt
from aii_pipeline.utils import to_prompt_yaml


def PROMPT_REVIEW(
    hypothesis_text: str,
    previous_feedback_text: str,
    previous_hypothesis_text: str,
) -> str:
    return f"""<role>
You are a very experienced and critical conference reviewer specialized in the domain of the work under review.
You have reviewed for top-tier venues in the relevant field. Your reviews are known for
being thorough, fair, and grounded in the actual state of the field.
</role>

<hypothesis>
{hypothesis_text}
</hypothesis>

<review_context>
No experiments have been run yet — evaluate the hypothesis purely on its merits.
</review_context>

{
        f'''<previous_hypothesis>
The hypothesis from the PREVIOUS iteration (before the revision under review).
Use this to classify how the current hypothesis relates to it (see the H↔H
edge instructions in the task).

{previous_hypothesis_text}
</previous_hypothesis>'''
        if previous_hypothesis_text
        else ""
    }

{
        f'''<previous_review>
Critiques from the previous review. Check which ones have been addressed
in the revised hypothesis. Do NOT re-raise critiques that have been adequately fixed.
Only re-raise if the fix is insufficient.

{previous_feedback_text}
</previous_review>'''
        if previous_feedback_text
        else ""
    }

<task>
Provide a thorough peer review of this research hypothesis.

STEP 1 — GROUND YOUR REVIEW IN EVIDENCE:
Before writing critiques, search for relevant context to make your review authoritative:
- Search for accepted papers at top venues in this area — what level of
  contribution gets accepted? How does this hypothesis compare?
- Search for the closest existing work — is this genuinely novel or incremental?
- Check if the proposed methodology has known failure modes in the literature

STEP 2 — WRITE YOUR REVIEW:
For each critique:
1. Categorize: methodology, evidence, novelty, clarity, scope, or rigor
2. Rate severity: major (would waste compute if not fixed) or minor (polish)
3. Describe the issue clearly
5. Suggest a concrete action to address it

Focus on the most impactful issues. Flag fatal flaws that would waste compute if not fixed first.

STABILITY IS OK: If the hypothesis is on track and just needs more iterations to prove itself,
keep your feedback similar to the previous round. Don't manufacture new critiques — only escalate
when the revision introduced new issues or failed to address prior ones.

{
        (
            '''STEP 3 — H↔H EDGE (only if a <previous_hypothesis> block is present):
Classify how the current hypothesis relates to the previous iteration's hypothesis
using Moulines's structuralist typology. Set ``relation_type`` to one of:
    - "evolution": refining specialised claims while keeping the same conceptual frame
    - "embedding": the previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian, incommensurable shift)
Set ``relation_rationale`` to a brief justification (≤50 chars).

If no <previous_hypothesis> is present (this is iteration 1), leave both fields
null/empty.'''
        )
        if previous_hypothesis_text
        else (
            '''STEP 3 — H↔H EDGE:
This is the first iteration — there is no previous hypothesis. Leave
``relation_type`` null and ``relation_rationale`` empty.'''
        )
    }

Provide your review via structured output.
</task>"""


def get_review(
    hypothesis: dict,
    previous_feedback_text: str | None = None,
    previous_hypothesis: dict | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for hypothesis review."""
    hypo_display = {
        k: v
        for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    # Mirror the same display-key filtering for the previous hypothesis.
    previous_hypo_text = ""
    if previous_hypothesis:
        prev_display = {
            k: v
            for k, v in previous_hypothesis.items()
            if k not in ["hypothesis_id", "is_seeded", "model"]
            and not (k == "seeds" and not previous_hypothesis.get("is_seeded"))
        }
        previous_hypo_text = to_prompt_yaml(prev_display)

    prompt = PROMPT_REVIEW(
        hypothesis_text=to_prompt_yaml(hypo_display),
        previous_feedback_text=previous_feedback_text or "",
        previous_hypothesis_text=previous_hypo_text,
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()
