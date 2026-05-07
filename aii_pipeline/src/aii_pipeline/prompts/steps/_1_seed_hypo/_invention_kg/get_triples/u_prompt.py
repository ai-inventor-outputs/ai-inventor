"""User prompt for triples knowledge graph extraction."""

from aii_pipeline.utils import to_prompt_yaml

from .....components.todo import get_todo_header


def build_retry_prompt(verification: dict) -> str:
    """Build retry prompt for failed URL verification.

    Args:
        verification: Dict with failed_triples list from verify_wikipedia_urls

    Returns:
        Prompt string instructing agent to fix invalid URLs
    """
    failed = verification.get("failed_triples", [])
    if not failed:
        return "Some Wikipedia URLs were invalid. Please fix them using WebSearch to find correct URLs."

    items = []
    for item in failed[:5]:
        triple = item.get("triple", {})
        items.append(
            {
                "name": triple.get("name", "Unknown"),
                "url": triple.get("wikipedia_url", "No URL"),
                "error": item.get("error", "Unknown error"),
            }
        )

    overflow = f"\n... and {len(failed) - 5} more" if len(failed) > 5 else ""

    return f"""The following Wikipedia URLs are invalid:

{to_prompt_yaml(items)}{overflow}

For each invalid URL:
1. Search with "site:en.wikipedia.org [entity name]" to find the correct Wikipedia article
2. Update triples_output.json with the correct URL"""


def triples_prompt(title: str, abstract: str) -> str:
    """
    Generate prompt for analyzing a research paper to extract triples.

    Args:
        title: Paper title
        abstract: Paper abstract

    Returns:
        Prompt string for the agent
    """
    return f"""<paper>
Paper Title: {title}
Paper Abstract: {abstract}
</paper>

<paper_classification>
"contribution" = proposes something new (method, technique, dataset, framework, benchmark, etc.)
"survey" = literature review, overview, meta-analysis, position papers (only references existing work)
</paper_classification>

<entity_types>
task      - Problem being solved (image classification, theorem proving, protein folding)
method    - Technique, algorithm, procedure (gradient descent, CRISPR, induction)
data      - Datasets, databases, benchmarks (ImageNet, MNIST, arXiv corpus)
artifact  - Pre-built: trained models, proof libraries (GPT-4, Mathlib, Cas9)
tool      - Software, instruments, platforms (PyTorch, Lean prover, microscope)
concept   - Abstract ideas, theories, frameworks (attention, category theory)
other     - Entities that don't fit above categories
</entity_types>

<relations>
uses     - Anything EXISTING that the paper uses (methods, datasets, tools, concepts, tasks)
proposes - Anything NEW/NOVEL that the paper creates or introduces

VALIDATION REQUIREMENTS:
- ALL papers MUST have at least 1 "uses" (papers always build on existing work)
- CONTRIBUTION papers MUST have at least 1 "proposes" (they must create something new)
</relations>

<YOUR_TODO_LIST>
{get_todo_header()}
1. Classify paper as "contribution" or "survey" based on title/abstract.

2. List ALL entities EXPLICITLY mentioned in title/abstract. For each, determine entity_type and relation.

3. For each entity: Search for "site:en.wikipedia.org [entity name]" to find the correct Wikipedia article. The site: prefix restricts results to English Wikipedia only.

4. Write triples_output.json with paper_type and all triples. For each triple:
   - name: Use the Wikipedia article title (e.g., "Gradient descent" not "gradient descent algorithm")
   - relation: how paper relates to entity (uses or proposes)
   - entity_type: one of task, method, data, artifact, tool, concept, other
   - wikipedia_url: The Wikipedia URL from search results (must start with https://en.wikipedia.org/wiki/)
   - relevance: 1 sentence explaining why it matters

<CRITICAL_SEARCH_INSTRUCTIONS>
When searching for Wikipedia articles, ALWAYS include "site:en.wikipedia.org" at the start of your query:
  Example: "site:en.wikipedia.org gradient descent"
  Example: "site:en.wikipedia.org convolutional neural network"

This restricts search results to English Wikipedia only.
</CRITICAL_SEARCH_INSTRUCTIONS>
</YOUR_TODO_LIST>

Begin now."""


__all__ = ["build_retry_prompt", "triples_prompt"]
