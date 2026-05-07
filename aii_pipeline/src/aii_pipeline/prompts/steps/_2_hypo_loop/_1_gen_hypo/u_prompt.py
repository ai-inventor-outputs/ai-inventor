"""User prompt for hypothesis generation."""

from ....components.read_skills import get_read_skills
from ....components.resources import get_resources_prompt
from ....components.time_budgets import get_time_budgets_overview
from ....components.tool_calling import get_tool_calling_guidance
from ....components.user_folder import get_user_folder_prompt
from ....components.user_request import get_user_request_prompt

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================


def PROMPT(
    aii_prompt: str,
    hypo_seeds_text: str | None,
    web_search: bool = False,
    previous_hypothesis_text: str = "",
    previous_review_feedback_text: str = "",
) -> str:
    resources = get_resources_prompt()

    # Inspiration block: seeds when provided, cross-domain guidance when unseeded
    if hypo_seeds_text is not None:
        inspiration_block = f"""
<hypothesis_inspiration>
<context>
Research opportunities identified from scientific literature. Use these to inspire your hypothesis.
</context>

<YOUR_INSPIRATION>
{hypo_seeds_text}
</YOUR_INSPIRATION>

<guidance>
Let these seeds inspire what PROBLEMS you tackle and how you FRAME your approach — not just as solution methods for conventional problems.
- What novel problems do these inspirations suggest?
- How do they reframe what's possible in the AII prompt?
- The inspiration can be subtle/indirect — problem framing matters more than solution methods
- Consider cross-field transfer of techniques
</guidance>
</hypothesis_inspiration>
"""
    else:
        inspiration_block = """
<hypothesis_inspiration>
<YOUR_INSPIRATION>
Human researchers overspecialize — they know their domain deeply but lack breadth to see when other fields have already solved analogous problems. Your advantage is breadth. Only propose a cross-domain transfer if it concretely outperforms existing approaches in this domain. Avoid handwavy analogies — if the imported method is vaguer or weaker than what domain experts already use, it's not worth proposing.

Explore cross-domain inspiration at three levels, from abstract to concrete. At each level, consider both established and recent developments — with slight priority for newer work, which tends to leverage more powerful tools and be less widely known.

1. CONCEPTUAL: Borrow high-level ideas, framings, or design philosophies from distant fields.
   What mental model or approach from another domain suggests a novel angle on this problem?

2. PROCEDURAL: Adapt specific problem-solving processes from other domains.
   What workflow, iterative strategy, or pipeline used elsewhere could restructure how this problem is attacked?

3. METHODOLOGICAL: Import concrete methods directly from other fields with minimal modification.
   What algorithm, formula, or technique from a different domain applies here as-is or with adaptation?

Cast wide — draw from ANY field, not just these examples: ecology, economics, physics, linguistics, game theory, control theory, materials science, cognitive science, epidemiology. The best hypotheses often come from Level 2-3 transfers that experts in the field would never encounter.
</YOUR_INSPIRATION>
</hypothesis_inspiration>
"""

    return f"""<task_preview>
You will generate 1 novel groundbreaking research hypothesis in the AII prompt below.
</task_preview>

<YOUR_AII_PROMPT>
"{aii_prompt}"
</YOUR_AII_PROMPT>
{inspiration_block}
{resources}

{get_time_budgets_overview()}

<YOUR_TASK>
Generate 1 novel groundbreaking research hypothesis in the AII prompt that is feasible with the above constraints.

{
        f'''<web_research_process>
{get_read_skills("aii-web-research-tools")}

1. DIVERGE: Brainstorm 5-7 diverse directions WITHOUT searching.
   Think across fields — what techniques from unrelated domains (ecology, economics, physics,
   linguistics, game theory, etc.) could inspire a novel mechanism? What assumptions does the field
   take for granted? Diversity matters more than depth here.

2. SEARCH: Web search for a high-level overview of each direction.
   What similar approaches exist? Is this genuinely novel or incremental? Remember: snippets
   are NOT enough for detailed understanding — treat search as discovery only.

3. FETCH & READ: MUST fetch any potentially relevant URL — you cannot assess novelty from
   snippets alone.
   - WebFetch for high-level understanding of HTML pages
   - aii_web_tools__fetch_grep for exact details, methodology, or PDFs
   Prioritize recent papers closest to your idea. If you find significant overlap, PIVOT.

4. ADVERSARIAL NOVELTY CHECK: Actively try to DISPROVE novelty. Most important step.
   WITHIN field: 5-6 different phrasings, failed/negative results, plain English without jargon.
   ACROSS fields (MANDATORY): strip to core mechanism, search 8-10 unrelated fields — same
   principle often exists under different names.
   Ask: "Is the core insight of your hypothesis new, or known things in a new wrapper?"
   "Would an expert find this genuinely surprising?"
   MANDATORY SELF-CHECK: State the core mechanism in one sentence. Does it exist in ANY
   algorithm, framework, or field? If yes — even in a different framework — ABANDON.

5. FEASIBILITY CHECK: Verify your hypothesis is testable with provided resources. What specific data/compute/tools
   needed? All available within constraints?

6. ABANDON or PROCEED:
   ABANDON if: 2+ similar papers exist; you need to argue "critical differences"; core mechanism
   exists in any context.
   Abandoning is progress — go back to step 1 in a genuinely DIFFERENT direction (not a variant).
   PROCEED only if novelty is SELF-EVIDENT — an expert would immediately see it's new without
   explanation.

7. ITERATE: Expect to repeat steps 1-6 multiple times. The first few directions will likely be
   non-novel. This is normal. Don't settle for your first idea just because you've invested time.

<CRITICAL>We want SCIENTIFIC novelty (new mechanism, principle, or insight — the contribution is
knowledge), NOT application novelty (known methods applied to a new domain — the contribution is a
product). If an expert would say "clever engineering but known science," keep searching.
Hypothesis must be feasible within available resources.</CRITICAL>

{get_tool_calling_guidance()}
</web_research_process>
'''
        if web_search
        else ""
    }
Prioritize simplicity. Use concise, approachable language. The explanation should be fully self-contained.
</YOUR_TASK>

{
        f'''<previous_hypothesis>
Your hypothesis from the previous iteration. The reviewer evaluated it below.

{previous_hypothesis_text}
</previous_hypothesis>

<previous_review_feedback>
A reviewer evaluated your previous hypothesis and provided the feedback below.

IMPORTANT: Do NOT generate a completely new hypothesis. Take the previous hypothesis above and
REVISE it to address the feedback. Keep what works, fix what was criticized.

You MUST address ALL the critiques. Do NOT repeat the same mistakes.

{previous_review_feedback_text}
</previous_review_feedback>'''
        if previous_hypothesis_text
        else ""
    }"""


FORCE_OUTPUT_PROMPT = """STOP. You have reached the maximum number of turns.

Do NOT use any more tools. Do NOT search, fetch, or implement anything.

Using ONLY the research you have gathered so far, write your final hypothesis as structured JSON output matching the Hypothesis schema. Output the structured JSON NOW."""


# =============================================================================
# HELPERS
# =============================================================================


def _format_hypo_seeds(sampled_prompts: list) -> str | None:
    """Format hypothesis inspiration prompts for hypothesis generation.

    Returns None when there are no seeds (unseeded mode), which causes
    the inspiration block to be omitted entirely from the prompt.
    """
    if not sampled_prompts:
        return None

    sections = []
    for i, prompt in enumerate(sampled_prompts, 1):
        if isinstance(prompt, dict):
            prompt_text = prompt.get("prompt", "")
        else:
            prompt_text = prompt
        sections.append(f'<inspiration id="{i}">\n{prompt_text}\n</inspiration>')

    return "\n\n".join(sections)


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    sampled_prompts: list,
    aii_prompt: str,
    web_search: bool = False,
    previous_hypothesis: dict | None = None,
    previous_review_feedback: dict | None = None,
    user_folder_path: str = "",
) -> str:
    """Generate hypothesis generation prompt with inspirations from research opportunities."""
    from aii_pipeline.utils import to_prompt_yaml

    prompt = PROMPT(
        aii_prompt=aii_prompt,
        hypo_seeds_text=_format_hypo_seeds(sampled_prompts),
        web_search=web_search,
        previous_hypothesis_text=to_prompt_yaml(previous_hypothesis) if previous_hypothesis else "",
        previous_review_feedback_text=to_prompt_yaml(previous_review_feedback)
        if previous_review_feedback
        else "",
    )
    return prompt + get_user_folder_prompt(user_folder_path) + get_user_request_prompt()


def get_force_output_prompt() -> str:
    """Prompt to force output when tool iterations are exhausted."""
    return FORCE_OUTPUT_PROMPT
