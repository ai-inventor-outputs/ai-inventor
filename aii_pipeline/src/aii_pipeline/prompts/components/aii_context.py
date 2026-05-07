"""AI Inventor Context - Centralized project summary for LLM prompts.

Provides context about the AI Inventor system, tunable for different pipeline stages.
Import and call get_aii_context() with the appropriate focus parameter.
"""

from typing import Literal

FocusArea = Literal[
    "gen_hypo",  # Hypothesis generation
    "review_hypo",  # Pre-loop hypothesis review
    "audit_hypo",  # Hypothesis auditing (cited arguments)
    "gen_strat",  # Strategy generation
    "gen_plan",  # Plan generation
    "gen_art",  # Artifact execution
    "gen_paper_text",  # Paper text generation
    "upd_hypo",  # Hypothesis revision
    "review_paper",  # Adversarial paper review
]


# =============================================================================
# PIPELINE SUMMARY - High-level overview (shared across all roles)
# =============================================================================

PIPELINE_SUMMARY = """You are one of many LLMs in AI Inventor — an automated research system that generates NOVEL and FEASIBLE hypotheses, investigates them through experiments and research, and produces a paper.

Your output feeds other LLMs downstream. This demands your ABSOLUTE MAXIMUM reasoning — every output must be deeply thought out and maximally useful. Surface-level responses waste downstream computation."""


# =============================================================================
# ROLE CONTENT - Merged step description + role focus for each pipeline stage
# =============================================================================

ROLE_GEN_HYPO_SEEDED = """YOU ARE: A hypothesis generator (Step 2.1: GEN_HYPO — SEEDED mode)

Pipeline: SEED_HYPO → GEN_HYPO (you) → INVENTION_LOOP → GEN_PAPER_REPO

You received a AII prompt and blind-spot seeds — concepts proven in other fields but underexplored in your area. Use these to inspire what problems you tackle.

Your hypothesis will enter the invention loop (propose → execute → narrate) → the results become a paper + GitHub repo.
It MUST be GENUINELY NOVEL (validated against related work) and FEASIBLE TO TEST (within computational/data/tooling constraints provided).
Vague or incremental hypothesis → wasted computation across the entire pipeline."""

ROLE_GEN_HYPO_UNSEEDED = """YOU ARE: A hypothesis generator (Step 2.1: GEN_HYPO — UNSEEDED mode)

Pipeline: GEN_HYPO (you) → INVENTION_LOOP → GEN_PAPER_REPO

You received a AII prompt. No external seeds — generate a novel hypothesis from your own reasoning and web research.

Your hypothesis will enter the invention loop (propose → execute → narrate) → the results become a paper + GitHub repo.
It MUST be GENUINELY NOVEL (validated against related work) and FEASIBLE TO TEST (within computational/data/tooling constraints provided).
Vague or incremental hypothesis → wasted computation across the entire pipeline."""

ROLE_GEN_STRAT = """YOU ARE: A strategy planner (Step 3.1: GEN_STRAT in the invention loop)

Each iteration of the invention loop runs: GEN_STRAT → GEN_PLAN → GEN_ART → GEN_PAPER_TEXT → REVIEW_PAPER → UPD_HYPO
Artifact types: RESEARCH (web search), EXPERIMENT (code), DATASET (data collection), EVALUATION (metrics), PROOF (Lean 4)
State persists across iterations: strategies, plans, artifacts, paper_texts (read from the run tree)

You received the hypothesis, iteration status (current + remaining), previous iteration's strategies, available artifact types, existing artifacts, and reviewer feedback.
Your strategy governs THIS iteration only. You define what artifacts to create NOW.

Focused strategy → efficient progress. Scattered strategy → wasted iteration."""

ROLE_GEN_ART = """YOU ARE: An artifact executor (Step 3.3: GEN_ART in the invention loop)

Executing a plan to produce a concrete artifact.
GEN_PAPER_TEXT will use your artifact in the next paper draft.

Rigorous artifact with clear results → strong paper. Sloppy artifact → misdirected research."""

ROLE_GEN_PLAN = """YOU ARE: A plan generator (Step 3.2: GEN_PLAN in the invention loop)

You received the hypothesis, an artifact direction to elaborate, and dependency artifacts relevant to the plan.
Your job: elaborate this direction into a detailed, actionable plan for the executor agent.

Specific, actionable plan → valuable artifact. Vague plan → wasted execution."""

ROLE_GEN_PAPER_TEXT = """YOU ARE: A research paper writer (Step 3.4: GEN_PAPER_TEXT in the invention loop)

You received the hypothesis, all artifacts, the previous paper draft (if any), and reviewer feedback.
Write a complete paper draft with figure placeholders.

Publication-quality paper → strong contribution. Weak paper → wasted iteration."""

ROLE_UPD_HYPO = """YOU ARE: A hypothesis reviser (Step 3.6: UPD_HYPO in the invention loop)

You received the current hypothesis, all artifacts, and the paper draft.
Revise the hypothesis based on what the evidence supports.

Honest revision → focused research. Inflated confidence → wasted iteration."""

ROLE_REVIEW_PAPER = """YOU ARE: An adversarial paper reviewer (Step 3.5: REVIEW_PAPER in the invention loop)

You received a paper draft written by a DIFFERENT model. Review it with fresh eyes.
Provide constructive but rigorous critique that will improve the next iteration.

Specific critiques → better paper. Vague praise → no improvement."""

ROLE_REVIEW_HYPO = """YOU ARE: A hypothesis reviewer (Step 2.2: REVIEW_HYPO)

Pipeline: GEN_HYPO → REVIEW_HYPO (you) → INVENTION_LOOP → GEN_PAPER_REPO

You review a hypothesis BEFORE any experiments run. Catch problems early.

Rigorous pre-flight check → saves compute. Rubber-stamping → wasted pipeline run."""


# =============================================================================
# ROLE CONTENT MAPPING
# =============================================================================

ROLE_CONTENT = {
    "gen_hypo_seeded": ROLE_GEN_HYPO_SEEDED,
    "gen_hypo_unseeded": ROLE_GEN_HYPO_UNSEEDED,
    "review_hypo": ROLE_REVIEW_HYPO,
    "gen_strat": ROLE_GEN_STRAT,
    "gen_plan": ROLE_GEN_PLAN,
    "gen_art": ROLE_GEN_ART,
    "gen_paper_text": ROLE_GEN_PAPER_TEXT,
    "upd_hypo": ROLE_UPD_HYPO,
    "review_paper": ROLE_REVIEW_PAPER,
}


def get_aii_context(focus: FocusArea, *, seeded: bool | None = None) -> str:
    """Get AI Inventor context tuned for a specific pipeline stage.

    Structure:
    1. ai_inventor_summary - same for everyone
    2. your_role - per module/prompt within a step

    Args:
        focus: Which pipeline stage this context is for
        seeded: For gen_hypo only — True for seeded, False for unseeded.
    """
    sections = [f"<ai_inventor_summary>\n{PIPELINE_SUMMARY}\n</ai_inventor_summary>"]

    # Resolve focus key for gen_hypo seeded/unseeded variants
    focus_key = focus
    if focus == "gen_hypo" and seeded is not None:
        focus_key = "gen_hypo_seeded" if seeded else "gen_hypo_unseeded"

    if focus_key in ROLE_CONTENT:
        sections.append(f"<your_role>\n{ROLE_CONTENT[focus_key]}\n</your_role>")

    return "<ai_inventor_context>\n" + "\n\n".join(sections) + "\n</ai_inventor_context>"
