"""User prompt for gen_paper_text (Step 3.4: GEN_PAPER_TEXT).

Walks the LLM through what the pipeline did this iteration so it understands
WHY each artifact exists and how it addresses previous feedback.

Prompt structure (chronological pipeline walkthrough):
1. Previous paper text (what we had last iteration)
2. Reviewer feedback (what was wrong with it)
3. Updated hypothesis (how we refined our thinking)
4. All artifacts (complete research so far)
5. New artifacts this iteration (what we just did to address feedback)
6. Task: write/revise the paper
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_pipeline.utils import to_prompt_yaml

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._3_gen_art.out_schema import (
        BaseArtifact,
    )

from ....components.data_files import get_reading_mini_preview_full
from ....components.read_skills import get_read_skills
from ....components.todo import get_todo_header
from ....components.user_folder import get_user_folder_prompt
from ....components.user_request import get_user_request_prompt

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

_ALL_FIELDS: set[str] = {
    "id",
    "type",
    "title",
    "summary",
    "workspace_path",
    "out_expected_files",
}
_NEW_FIELDS: set[str] = {"id", "type", "title", "summary"}


def PROMPT(
    hypothesis_text: str,
    all_artifacts_str: str,
    all_artifact_count: int,
    new_artifacts_str: str,
    new_artifact_count: int,
    previous_paper_text: str,
    reviewer_feedback_text: str,
) -> str:
    is_first = not previous_paper_text

    if is_first:
        task_text = "This is the FIRST paper draft. Write a complete research paper from scratch based on the hypothesis and all available artifacts."
    else:
        task_text = """YOUR TURN (gen_paper_text): Revise the paper.

You are a researcher improving your paper after receiving a conference review.
Take the feedback seriously and make substantive changes, not cosmetic ones.

1. ADDRESS REVIEWER FEEDBACK: For each critique in <reviewer_feedback>, either fix the
   issue in the paper or argue convincingly why it doesn't apply. Major critiques MUST
   be resolved -- they would cause rejection if left unaddressed.
2. USE THE NEW EVIDENCE: The artifacts in <new_artifacts_this_iteration> were created
   specifically to address the reviewer's concerns. Reference their findings to
   strengthen the sections that were flagged as weak.
3. REWRITE, DON'T PATCH: Don't just append new paragraphs. Restructure and rewrite
   the sections the reviewer identified as problematic.
4. MAINTAIN CONSISTENCY: Ensure the paper aligns with the updated hypothesis."""

    return f"""{
        f'''<previous_paper>
STARTING POINT: This is your paper draft from the previous iteration.

{previous_paper_text}
</previous_paper>

'''
        if previous_paper_text
        else ""
    }{
        f'''<reviewer_feedback>
STEP 1 — REVIEW: A reviewer evaluated the previous paper draft above and produced this feedback.

{reviewer_feedback_text}
</reviewer_feedback>

<pipeline_steps>
STEP 2 — STRATEGY: The pipeline's strategy generator (gen_strat) read the reviewer feedback
and designed a new research strategy to address the critiques.

STEP 3 — PLANNING: The planner (gen_plan) turned the strategy into concrete artifact plans —
specific experiments, datasets, or research tasks to execute.

STEP 4 — EXECUTION: The executor (gen_art) ran those plans and produced the new artifacts
shown in <new_artifacts_this_iteration> below.
</pipeline_steps>

'''
        if reviewer_feedback_text
        else ""
    }<hypothesis>
{
        "STEP 5 — HYPOTHESIS UPDATE: The hypothesis was revised based on evidence from previous iterations."
        if not is_first
        else "The research hypothesis."
    }

{hypothesis_text}
</hypothesis>

<all_artifacts>
FULL EVIDENCE BASE: All {all_artifact_count} research artifacts across all iterations.

{all_artifacts_str}
</all_artifacts>

{
        f'''<new_artifacts_this_iteration>
NEW THIS ITERATION: These {new_artifact_count} artifacts were created to address the reviewer
feedback. Their findings should be the primary basis for your revisions.

{new_artifacts_str}
</new_artifacts_this_iteration>

'''
        if new_artifacts_str and new_artifact_count > 0
        else ""
    }<data_files>
{get_reading_mini_preview_full()}
</data_files>

<task>
Write a research paper draft with LaTeX-ready text, BibTeX citations, and figure placeholders.

{task_text}
</task>

<figure_instructions>
FIGURE FORMAT: Use [FIGURE:fig_id] markers in paper_text to indicate where each figure goes.
Then provide the full figure specs in the separate `figures` structured output array.
Each figure in the array must have an `id` matching a marker in the text. Set the `aspect_ratio`
field per figure: 21:9 for architecture / pipeline / flow-chart diagrams (the hero figure should
be one of these — place its marker near the END of the Introduction so it floats to the top of
page 2), 16:9 for comparisons / multi-panel results, 4:3 for dense charts, 1:1 for heatmaps /
confusion matrices / scatter plots.

Example in paper_text:
  "...our method achieves state-of-the-art results as shown below.\\n\\n[FIGURE:fig3]\\n\\nThe results demonstrate..."

Example in figures array (results comparison):
  {{"id": "fig3", "title": "Performance Comparison", "caption": "Comparison of geometric mean query latency across optimizers.", "image_gen_detailed_description": "Grouped bar chart. X-axis: model names. Y-axis: latency (seconds, 0-5). Values: PostgreSQL=4.6s (red), Bao=2.8s (blue), RLQOpt=2.0s (green). Error bars +/-0.3-0.8. Sans-serif font, white background.", "aspect_ratio": "16:9", "summary": "Compares latency across optimizers"}}

Example in figures array (architecture diagram, hero):
  {{"id": "fig1", "title": "System Architecture", "caption": "End-to-end pipeline: encoder feeds latents into the planner, which queries the value head before emitting actions.", "image_gen_detailed_description": "Horizontal flow diagram, left to right. Five labeled boxes: 'Input' (gray), 'Encoder' (blue), 'Latent (z, 256-dim)' (light blue, narrow), 'Planner' (green), 'Action Head' (orange). Arrows labeled with shapes. Value head as separate green box below 'Planner', bidirectional arrow. Sans-serif font, clean white background, no 3D.", "aspect_ratio": "21:9", "summary": "Hero architecture diagram"}}

CRITICAL: Before writing figure specs, look through artifact workspace output files (*_out.json)
and code to find ALL the exact values. The figure generator cannot read files — every exact number
and value MUST be in the image_gen_detailed_description.
</figure_instructions>

{get_todo_header()}
{_format_todos(TODOS)}"""


TODOS = [
    get_read_skills("aii-paper-writing", "aii-semscholar-bib"),
    """LITERATURE REVIEW: Use web search tools to research the landscape — search key terms from
<hypothesis> and <all_artifacts>. Then use aii_semscholar_bib__fetch to batch-fetch real
BibTeX entries. Build a comprehensive Related Work section. Do NOT fabricate entries.""",
    """READ ARTIFACTS: Before writing each section, READ the relevant artifact source code, output
files, and data in the workspace. Extract concrete implementation details, technical innovations,
algorithmic specifics, and quantitative results. Do NOT write surface-level descriptions.

ARTIFACT REFERENCES: When you reference results, methodology, or findings from a specific artifact,
place an [ARTIFACT:artifact_id] marker inline. These become footnotes linking to the artifact's code
in the GitHub repository (first mention gets a footnote with URL, subsequent mentions are omitted).
Use the exact artifact ID from <all_artifacts>. Place the marker right after the claim it supports.
Example:
  "Our evaluation showed a 15% improvement over baselines [ARTIFACT:gen_evaluation_id1_it2__opus]." """,
    """WRITE PAPER: Write the full paper text with [FIGURE:fig_id] markers per <figure_instructions>,
and provide the figure specs in the figures array. Cite with numeric references [1], [2], etc.
At the end of the paper text, include a full bibliography section. Do NOT compile LaTeX or generate
actual image/figure files. Your ONLY output is the structured JSON.""",
]


def _format_todos(todos: list[str]) -> str:
    """Format TODO items into a single <todos> block."""
    lines = ["<todos>"]
    for i, item in enumerate(todos, start=1):
        lines.append(f"TODO {i}. {item}")
    lines.append("</todos>")
    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    hypothesis: dict,
    artifacts: list[BaseArtifact],
    current_iter_artifacts: list[BaseArtifact],
    iteration: int,
    previous_paper_text: str | None = None,
    reviewer_feedback_text: str | None = None,
    user_folder_path: str = "",
) -> str:
    """Build user prompt for paper text generation in the invention loop."""
    from aii_lib.prompts import LLMPromptModel

    hypo_display = {
        k: v
        for k, v in hypothesis.items()
        if k not in ["hypothesis_id", "is_seeded", "model"]
        and not (k == "seeds" and not hypothesis.get("is_seeded"))
    }

    # All artifacts across all iterations
    all_arts = (
        LLMPromptModel.list_to_prompt_yaml(
            artifacts,
            label="Item",
            include=_ALL_FIELDS,
            strip_nulls=True,
        )
        or "No artifacts yet."
    )

    # New artifacts from this iteration only
    if current_iter_artifacts:
        new_arts = "\n\n".join(
            to_prompt_yaml(
                {k: getattr(a, k, None) for k in _NEW_FIELDS if getattr(a, k, None) is not None}
            )
            for a in current_iter_artifacts
        )
    else:
        new_arts = ""

    prompt = PROMPT(
        hypothesis_text=to_prompt_yaml(hypo_display),
        all_artifacts_str=all_arts,
        all_artifact_count=len(artifacts),
        new_artifacts_str=new_arts,
        new_artifact_count=len(current_iter_artifacts),
        previous_paper_text=previous_paper_text or "",
        reviewer_feedback_text=reviewer_feedback_text or "",
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()


# =============================================================================
# RETRY PROMPT (figure verification)
# =============================================================================


def build_figure_retry_prompt(verification: dict) -> str:
    """Build retry prompt for figure marker/array verification failures."""
    lines = [
        "<verification_results>",
        "Your figures have consistency issues that need fixing:",
        "",
    ]

    missing = verification.get("missing_figures", [])
    orphans = verification.get("orphan_figures", [])
    duplicates = verification.get("duplicate_ids", [])
    field_errors = verification.get("field_errors", [])

    if missing:
        lines.append("MISSING FIGURE SPECS (markers in text but no matching figure in array):")
        for fid in missing:
            lines.append(
                f"  - [FIGURE:{fid}] in text but no figure with id='{fid}' in figures array"
            )
        lines.append("")
    if orphans:
        lines.append("ORPHAN FIGURES (in array but no matching marker in text):")
        for fid in orphans:
            lines.append(f"  - Figure id='{fid}' in array but no [FIGURE:{fid}] marker in text")
        lines.append("")
    if duplicates:
        lines.append("DUPLICATE IDS:")
        for fid in duplicates:
            lines.append(f"  - '{fid}' appears multiple times in figures array")
        lines.append("")
    if field_errors:
        lines.append("MISSING FIELDS:")
        for err in field_errors:
            lines.append(f"  - {err}")
        lines.append("")

    marker_ids = verification.get("marker_ids", [])
    figure_ids = verification.get("figure_ids", [])
    lines.append(
        f"Summary: {len(set(marker_ids))} unique markers in text, {len(figure_ids)} figures in array."
    )
    lines.append("</verification_results>")
    lines.append("")
    lines.append("<task>")
    lines.append(
        "Fix ALL issues above. The rule is simple: every [FIGURE:id] marker in paper_text MUST have exactly one matching figure in the figures array, and every figure in the array MUST have a [FIGURE:id] marker in the text."
    )
    lines.append("")
    if missing:
        lines.append(
            f"For missing specs: add a figure entry to the figures array for each of: {', '.join(missing)}. Each needs id, title, caption, image_gen_detailed_description (with ALL data values), and summary."
        )
    if orphans:
        lines.append(
            f"For orphan figures: either add [FIGURE:id] markers in the text for: {', '.join(orphans)}, or remove them from the figures array if they are not needed."
        )
    if duplicates:
        lines.append(
            f"For duplicates: ensure each figure ID is unique. Duplicated: {', '.join(duplicates)}. Rename or merge them."
        )
    if field_errors:
        lines.append(
            "For missing fields: fill in the missing title/caption/image_gen_detailed_description for the affected figures. image_gen_detailed_description must include ALL data values — the image generator cannot read files."
        )
    lines.append("</task>")

    return "\n".join(lines)


__all__ = ["build_figure_retry_prompt", "get"]
