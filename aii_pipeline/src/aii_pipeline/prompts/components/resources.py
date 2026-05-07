"""Resource definitions for prompts.

Split into reusable parts so different artifacts can include only what they need.
"""

from __future__ import annotations


def get_resources_software(agent_timeout_seconds: int | None = None):
    """Software constraints."""
    from aii_pipeline.utils.context import get_pipeline_config
    from aii_pipeline.utils.pipeline_config import DEFAULT_MAX_USD_OPENROUTER_PER_ART

    cfg = get_pipeline_config()
    max_usd = cfg.max_usd_openrouter_per_art if cfg else DEFAULT_MAX_USD_OPENROUTER_PER_ART
    return f"""<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum ${max_usd:.0f} USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>"""


def get_resources_skills():
    """Skills available to agents (HTTP tools on port 8020 + Claude Code skills)."""
    return """<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-research-tools: Web search, page fetching, PDF/HTML text extraction
- aii-web-tools: Web fetch, extract text from URLs, verify citations
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>"""


# =============================================================================
# REGISTRY & PUBLIC API
# =============================================================================

_RESOURCE_SECTIONS = {
    "software": get_resources_software,
    "skills": get_resources_skills,
}

DEFAULT_SECTIONS = ["software", "skills"]

# Per-artifact-type resource needs (used by gen_strat, gen_plan to select relevant sections)
ARTIFACT_RESOURCES: dict[str, set[str]] = {
    "research": {"software"},
    "proof": {"software"},
    "dataset": {"software", "skills"},
    "experiment": {"software", "skills"},
    "evaluation": {"software", "skills"},
}


def get_resources_prompt(
    include: list[str] | None = None,
    agent_timeout_seconds: int | None = None,
) -> str:
    """Get formatted resources block for prompts.

    Args:
        include: Which resource sections to include (e.g. ["software", "skills"]).
                 Options: "software", "skills".
                 If None, includes all DEFAULT_SECTIONS.
        agent_timeout_seconds: If set, shows the time limit in the software constraints.

    Returns:
        Formatted string wrapped in <available_resources> tags.
    """
    if include is None:
        include = DEFAULT_SECTIONS

    sections = []
    for key in include:
        if key == "software":
            sections.append(get_resources_software(agent_timeout_seconds=agent_timeout_seconds))
        elif key in _RESOURCE_SECTIONS:
            sections.append(_RESOURCE_SECTIONS[key]())

    content = "\n\n".join(sections)
    return f"""<available_resources>
{content}
</available_resources>"""
