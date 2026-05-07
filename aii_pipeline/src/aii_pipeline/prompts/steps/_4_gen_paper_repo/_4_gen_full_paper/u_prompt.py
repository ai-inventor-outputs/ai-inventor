"""User prompt for LaTeX compilation.

Read top-to-bottom to understand the full prompt structure.
All todos are delivered in a single prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aii_lib.agent_backend import ExpectedFile

from ....components.read_skills import get_read_skills
from ....components.todo import get_todo_header
from ....components.tool_calling import get_tool_calling_guidance
from ....components.workspace import get_workspace_prompt

if TYPE_CHECKING:
    from aii_pipeline.prompts.steps._3_invention_loop._4_gen_paper_text.out_schema import (
        PaperText,
    )

    from .._2_gen_viz.out_schema import Figure


# =============================================================================
# PROMPT SECTIONS (edit these directly)
# =============================================================================

HEADER = """{workspace}

<task>
Create a publication-ready top-conference LaTeX paper with BibTeX from <paper_text> and <available_figures>, compile to PDF.
</task>

{tool_calling}

<paper_text>
{paper_yaml}
</paper_text>

<available_figures>
{fig_list}
</available_figures>

<figure_requirements>
CRITICAL: Include ALL figures from <available_figures>. No exceptions.

- Every figure MUST use \\includegraphics{{figures/filename.jpg}}
- Do NOT skip, convert to tables, or describe without inserting
- Each needs: \\begin{{figure*|figure}}[placement], \\includegraphics, \\caption, \\label, \\end{{...}} — pick env + placement by the figure's `aspect_ratio` field (see PLACEMENT below). Constrain every \\includegraphics with `width=\\linewidth,height=0.4\\textheight,keepaspectratio` (single-column) or `width=\\textwidth,height=0.45\\textheight,keepaspectratio` (figure*). Use exactly these option keys — `max height=` is NOT valid LaTeX
- Use the `caption` field from each figure for \\caption{{...}} — do NOT invent new captions
- Place figures where their [FIGURE:fig_id] markers appear in paper_text
- VERIFICATION: paper.tex MUST have exact same number of \\includegraphics as <available_figures>
- Do NOT generate new figure images (no matplotlib, no PIL, no image generation). Use ONLY the pre-generated figures from <available_figures>. They were already created by a previous pipeline step.

PLACEMENT BY ASPECT RATIO (use the `aspect_ratio` field on each figure):
- `21:9` (architecture diagrams / hero figures): \\begin{{figure*}}[!t] (full two-column width, top of page). The hero architecture diagram should appear EARLY in the paper — typically at the top of page 2. Marker placement in paper_text already determines this; preserve it.
- `16:9` (comparisons, multi-panel results): \\begin{{figure*}}[!t] for full-width or \\begin{{figure}}[!htbp] for single-column.
- `4:3` / `1:1` / `3:2` / `3:4` / `9:16`: \\begin{{figure}}[!htbp] (single-column).
</figure_requirements>

<artifact_links>
The paper_text contains \\footnote{{Code: \\url{{...}}}} references linking to artifact source code
on GitHub. Include \\usepackage{{hyperref}} and \\usepackage{{url}}.
Preserve these exactly as-is — do not remove, rewrite, or convert them to plain text.
The URLs will not resolve yet (the repo is deployed after compilation) — do NOT try to verify or fix them.
</artifact_links>

<headings>
NEVER use inline math (``$...$``) inside ``\\section{{...}}`` / ``\\subsection{{...}}`` / ``\\subsubsection{{...}}`` arguments — hyperref's bookmark builder errors out (``Token not allowed in a PDF string``) and the PDF outline breaks. If a section heading needs a math-looking term, use the text equivalent (``d star`` not ``$d^*$``, ``alpha-equivalent`` not ``$\\alpha$-equivalent``) or wrap it in ``\\texorpdfstring{{$math$}}{{plain}}``. Inline math inside body paragraphs is fine.
</headings>

{todo_header}"""

TODOS = [
    get_read_skills("aii-paper-to-latex", "aii-semscholar-bib"),
    """Review <paper_text> and <available_figures>. Copy all figure images into ./figures/ in your workspace. Count figures — MUST include every one. Plan placements per section. Build `./references.bib` via aii_semscholar_bib__fetch — collect DOIs/ArXiv IDs from <paper_text> and batch-fetch all BibTeX in one call. Do NOT fabricate entries.""",
    """Create `./paper.tex` per aii-paper-to-latex skill's setup, write ALL sections, insert ALL figures from <available_figures>, include `./references.bib` via \\bibliography. Compile to PDF per skill's process. Fix errors.""",
    """CRITICAL VERIFICATION: Run `grep -c 'includegraphics' paper.tex`, confirm count equals figures in <available_figures>. If not, add missing figures. Verify `./paper.pdf` was created.""",
    """VISUAL REVIEW: Write Python script to convert EVERY page of paper.pdf to PNG at 150 DPI (use pdf2image or pymupdf). Then read ALL page screenshots — each page image costs ~1,600 tokens so a 15-page paper is only ~24K tokens. You MUST read every page. The ONLY exception is if all page images would not fit in your remaining context — in that case, read as many as fit and state which pages you are skipping and why. Check every page for layout issues, overlapping figures, cut-off text, bad spacing, formatting problems. Fix issues and recompile.""",
    """FINAL READ: Check page count (`pdfinfo paper.pdf` or pymupdf). Read entire paper.pdf — check for missing sections, unclear explanations, inconsistencies, typos. Fix and recompile. The ONLY exception is if all pages would not fit in your remaining context — in that case, read as many pages as fit and state which pages you are skipping and why.""",
]


# =============================================================================
# HELPERS
# =============================================================================


def _format_todos(todos: list[str]) -> str:
    """Format TODO items into a single <todos> block."""
    lines = ["<todos>"]
    for i, item in enumerate(todos, start=1):
        lines.append(f"TODO {i}. {item}")
    lines.append("</todos>")
    return "\n".join(lines)


def _format_paper_yaml(paper: PaperText | None) -> str:
    """Serialize paper to prompt YAML (excluding figures — those come from <available_figures>)."""
    if not paper:
        return ""
    return paper.to_prompt_yaml(strip_nulls=True, exclude={"figures"})


def _build_header(
    paper: PaperText | None,
    fig_list: str,
    workspace_path: str = "",
) -> str:
    """Build the header section with substitutions."""
    return HEADER.format(
        workspace=get_workspace_prompt(workspace_path) if workspace_path else "",
        tool_calling=get_tool_calling_guidance(),
        paper_yaml=_format_paper_yaml(paper),
        fig_list=fig_list,
        todo_header=get_todo_header(),
    )


def format_figures(figures: list[Figure]) -> str:
    """Format figure list for prompt using Figure.list_to_prompt_yaml().

    Args:
        figures: Figure objects with full figure metadata + file paths.

    Returns:
        YAML-formatted figure list for prompt context.
    """
    if not figures:
        return "No figures available."

    from .._2_gen_viz.out_schema import Figure

    return Figure.list_to_prompt_yaml(figures)


# =============================================================================
# EXPORTS
# =============================================================================


def get(
    paper: PaperText | None = None,
    figures: list[Figure] | None = None,
    workspace_path: str = "",
) -> str:
    """Get single prompt with all TODOs combined."""
    header = _build_header(
        paper=paper,
        fig_list=format_figures(figures or []),
        workspace_path=workspace_path,
    )
    all_todos = _format_todos(TODOS)
    return f"{header}\n{all_todos}"


# =============================================================================
# METADATA FUNCTIONS
# =============================================================================


def get_expected_out_files() -> list[ExpectedFile]:
    """All files the prompt explicitly asks the agent to create."""
    return [
        ExpectedFile("paper.tex", "LaTeX source file for the research paper"),
        ExpectedFile("paper.pdf", "Compiled PDF of the research paper"),
        ExpectedFile("references.bib", "BibTeX bibliography file"),
    ]


def get_latex_filename() -> str:
    """Standard filename for LaTeX source."""
    return "paper.tex"


def get_pdf_filename() -> str:
    """Standard filename for compiled PDF."""
    return "paper.pdf"


def get_figures_folder() -> str:
    """Standard folder name for figures in LaTeX workspace."""
    return "figures"
