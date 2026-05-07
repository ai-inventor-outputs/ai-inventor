"""User prompt for artifact demo generation (notebook conversion).

Read top-to-bottom to understand the full prompt structure.
Sequential todos are numbered continuously.
"""

from __future__ import annotations

from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from aii_lib.agent_backend import ExpectedFile

from ....components.data_files import get_reading_mini_preview_full
from ....components.read_skills import get_read_skills
from ....components.todo import get_todo_header
from ....components.tool_calling import get_tool_calling_guidance
from ....components.workspace import get_workspace_prompt

if TYPE_CHECKING:
    from ...._3_invention_loop._3_gen_art.out_schema import BaseArtifact


# =============================================================================
# PROMPT SECTIONS (edit these directly)
# =============================================================================

HEADER = """{workspace}

{tool_calling}

<task>
Convert this artifact's Python script into a demo notebook with MINIMAL changes to the original code.
Split into cells, add markdown explanations between sections, add a visualization cell at the end.
Output: mini_demo_data.json + code_demo.ipynb (notebook that loads data from GitHub URL)
</task>

<artifact_info>
{artifact_yaml}
</artifact_info>

<github_repo>
Repo URL: {repo_url}
Raw data URL: {github_code_mini_demo_data_url}

URLs won't work yet — files pushed to GitHub AFTER notebook creation.
Use local fallback pattern so notebook works locally (now) and in Colab (after deployment).
</github_repo>

<data_file_sizes>
{reading_mini_preview_full}
</data_file_sizes>

<install_dependencies_pattern>
Follow the aii-colab skill exactly. It has the install cell pattern, pre-installed package list, numpy 2.0 compat shims, and all Colab-specific rules.
</install_dependencies_pattern>

<data_loading_pattern>
`mini_demo_data.json` = curated subset for the demo.
Use this pattern for Colab compatibility (GitHub URL with local fallback):
```python
GITHUB_DATA_URL = "{github_code_mini_demo_data_url}"
import json, os

def load_data():
    try:
        import urllib.request
        with urllib.request.urlopen(GITHUB_DATA_URL) as response:
            return json.loads(response.read().decode())
    except Exception: pass
    if os.path.exists("mini_demo_data.json"):
        with open("mini_demo_data.json") as f: return json.load(f)
    raise FileNotFoundError("Could not load mini_demo_data.json")
```
</data_loading_pattern>

<notebook_structure>
--- Setup ---
Cell 1 (markdown): Title, description, what this artifact does.
Cell 2 (code): Install dependencies — follow the aii-colab skill's install cell pattern exactly. Fill in all packages imported by the artifact's code.
Cell 3 (code): Imports — copy original import block as-is, plus any additional imports needed for the notebook (e.g. matplotlib for visualization).
Cell 4 (code): Data loading helper — use the <data_loading_pattern> above.
Cell 5 (code): `data = load_data()`

--- Config ---
Config cell (code): Define ALL tunable parameters (iterations, epochs, n_samples, hidden_size, etc.) as variables at the top of this cell. Start with the ABSOLUTE MINIMUM values — the smallest that produce any output at all (e.g. 1 iteration, 2 samples, smallest array size). These get gradually increased during testing — see TODOs.

--- Processing ---
Remaining cells: One code cell per logical section of the original script. Add a markdown cell BEFORE each code cell. Copy code as closely as possible, with these changes:
  1. Replace file paths to use the loaded `data` variable.
  2. Use the config variables from the config cell (NOT hardcoded values).
  3. Minimal fixes are allowed if something doesn't work in notebook context (e.g. adjusting paths, removing CLI args, fixing imports), but keep changes to the absolute minimum.

--- Results ---
Visualization cell (code): Print key results in a readable table, plot numeric data with matplotlib if appropriate.
</notebook_structure>

<priority>
WORKING > OPTIMIZED. A small-scale demo that runs correctly is the goal. Once the notebook passes with minimum config values, scale up only if time permits — do NOT spend multiple retries chasing larger parameters. If a working version exists, finish and move on.
</priority>

<max_notebook_total_runtime>{max_notebook_total_runtime}s ({max_notebook_total_runtime_min} min)</max_notebook_total_runtime>

<test_environment>
To test-run the notebook in a clean environment (simulating Colab), create a disposable `.nb_env` in your workspace:
```bash
{python_path} -m venv .nb_env
.nb_env/bin/pip install -q pip jupyter ipykernel
.nb_env/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout={max_notebook_total_runtime} code_demo.ipynb --output code_demo.ipynb
rm -rf .nb_env
```
The timeout is set to <max_notebook_total_runtime>. The entire notebook must finish within this time.

What happens: the .venv starts empty (just jupyter). When the notebook's install cell runs, `google.colab` is NOT in sys.modules, so ALL packages get installed — non-Colab packages unconditionally, and Colab packages (numpy, pandas, etc.) at Colab's exact versions via the guard block. The result mirrors Colab's environment as closely as possible. If a cell fails, fix the notebook and re-run.
</test_environment>

{todo_header}"""

TODOS = [
    [
        get_read_skills("aii-colab", "aii-long-running-tasks"),
        "Read demo file and relevant preview_* files (preview only). Understand script structure: imports, setup, processing, output. Identify ALL tunable parameters (iterations, epochs, n_samples, hidden_size, batch_size, etc.) — these go in the config cell.",
        "Create `mini_demo_data.json`: curated subset from at most ONE dataset (no more than 100 diverse examples). CRITICAL: do NOT read/grep full output file — may crash. Use `head -c 5000` or stream first entries with Python to pick examples.",
        "Create `code_demo.ipynb` via NotebookEdit following <notebook_structure>. Set ALL config parameters to ABSOLUTE MINIMUM values — the smallest that produce any output (e.g. 1 iteration, 2 samples, smallest array sizes). Test-run using <test_environment>. Fix all errors until it passes.",
        "GRADUALLY SCALE (but don't overdo it): increase config params step by step (e.g. ~2x each round). After each increase: test-run, record runtime, fix errors. STOP SCALING as soon as results look meaningful — a working small-scale demo beats a failed large-scale one. If full original params fit within <max_notebook_total_runtime> (10% margin), use them. Otherwise keep whatever works and comment out the true original values. Do NOT spend more than 2-3 scaling rounds.",
        "Verify: (1) code_demo.ipynb contains GITHUB_DATA_URL = \"{github_code_mini_demo_data_url}\" exactly, (2) mini_demo_data.json exists, (3) uses GitHub URL pattern not just open('mini_demo_data.json').",
    ],
]


# =============================================================================
# HELPERS
# =============================================================================


def _find_python312_path() -> str:
    """Find Python 3.12 binary path for venv creation (matches Colab runtime)."""
    # Check system python3.12 first
    system = which("python3.12")
    if system:
        return system
    # Check uv-managed Python
    uv_dir = Path.home() / ".local/share/uv/python"
    if uv_dir.exists():
        for p in sorted(uv_dir.glob("cpython-3.12*"), reverse=True):
            candidate = p / "bin" / "python3.12"
            if candidate.exists():
                return str(candidate)
    # Last resort
    return "python3"


def _format_artifact_yaml(artifact: BaseArtifact | None) -> str:
    """Serialize artifact to prompt YAML.

    Shows out_expected_files (all outputs) and out_demo_files (which files
    to convert to demos), but not out_dependency_files (irrelevant for demos).
    """
    if not artifact:
        return ""
    include = {
        "id",
        "type",
        "title",
        "summary",
        "workspace_path",
        "out_expected_files",
        "out_demo_files",
    }
    return artifact.to_prompt_yaml(include=include, strip_nulls=True)


def _format_todos(
    todo_items: list[str],
    start_num: int,
    artifact_name: str,
    github_code_mini_demo_data_url: str = "",
) -> str:
    """Format a list of todo items with numbering."""
    lines = ["<todos>"]
    for i, item in enumerate(todo_items, start=start_num):
        formatted_item = item.format(
            artifact_name=artifact_name,
            github_code_mini_demo_data_url=github_code_mini_demo_data_url,
        )
        lines.append(f"TODO {i}. {formatted_item}")
    lines.append("</todos>")
    return "\n".join(lines)


def _build_header(
    artifact_name: str,
    artifact: BaseArtifact | None,
    available_files: list[ExpectedFile] | None = None,
    repo_url: str | None = None,
    github_code_mini_demo_data_url: str | None = None,
    workspace_path: str = "",
    max_notebook_total_runtime: int = 600,
) -> str:
    """Build the header section with substitutions."""
    return HEADER.format(
        artifact_name=artifact_name,
        artifact_yaml=_format_artifact_yaml(artifact),
        workspace=get_workspace_prompt(workspace_path) if workspace_path else "",
        reading_mini_preview_full=get_reading_mini_preview_full(),
        tool_calling=get_tool_calling_guidance(),
        repo_url=repo_url or "NOT_SET",
        github_code_mini_demo_data_url=github_code_mini_demo_data_url
        or "UPDATE_THIS_URL_WITH_YOUR_REPO",
        python_path=_find_python312_path(),
        max_notebook_total_runtime=max_notebook_total_runtime,
        max_notebook_total_runtime_min=max_notebook_total_runtime // 60,
        todo_header=get_todo_header(),
    )


# =============================================================================
# EXPORTS
# =============================================================================


def get_initial(
    artifact_name: str,
    artifact: BaseArtifact | None = None,
    available_files: list[ExpectedFile] | None = None,
    repo_url: str | None = None,
    github_code_mini_demo_data_url: str | None = None,
    workspace_path: str = "",
    max_notebook_total_runtime: int = 600,
) -> str:
    """Get the first prompt (header + first todos + footer)."""
    header = _build_header(
        artifact_name,
        artifact,
        available_files,
        repo_url,
        github_code_mini_demo_data_url,
        workspace_path,
        max_notebook_total_runtime=max_notebook_total_runtime,
    )
    first_todos = _format_todos(
        TODOS[0],
        start_num=1,
        artifact_name=artifact_name,
        github_code_mini_demo_data_url=github_code_mini_demo_data_url or "",
    )
    return f"{header}\n\n{first_todos}"


def get_sequential(
    artifact_name: str,
    artifact: BaseArtifact | None = None,
    available_files: list[ExpectedFile] | None = None,
    repo_url: str | None = None,
    github_code_mini_demo_data_url: str | None = None,
    workspace_path: str = "",
    max_notebook_total_runtime: int = 600,
) -> list[str]:
    """Get follow-up prompts (header + todos + footer each)."""
    header = _build_header(
        artifact_name,
        artifact,
        available_files,
        repo_url,
        github_code_mini_demo_data_url,
        workspace_path,
        max_notebook_total_runtime=max_notebook_total_runtime,
    )
    prompts = []
    current_num = len(TODOS[0]) + 1  # Start after first group

    for todo_group in TODOS[1:]:
        todos_section = _format_todos(
            todo_group,
            start_num=current_num,
            artifact_name=artifact_name,
            github_code_mini_demo_data_url=github_code_mini_demo_data_url or "",
        )
        prompts.append(f"{header}\n\n{todos_section}")
        current_num += len(todo_group)

    return prompts


def get_all_prompts(
    artifact_name: str,
    artifact: BaseArtifact | None = None,
    available_files: list[ExpectedFile] | None = None,
    repo_url: str | None = None,
    github_code_mini_demo_data_url: str | None = None,
    workspace_path: str = "",
    max_notebook_total_runtime: int = 600,
) -> list[str]:
    """Get all prompts for sequential execution."""
    initial = get_initial(
        artifact_name,
        artifact,
        available_files,
        repo_url,
        github_code_mini_demo_data_url,
        workspace_path,
        max_notebook_total_runtime=max_notebook_total_runtime,
    )
    sequential = get_sequential(
        artifact_name,
        artifact,
        available_files,
        repo_url,
        github_code_mini_demo_data_url,
        workspace_path,
        max_notebook_total_runtime=max_notebook_total_runtime,
    )
    return [initial, *sequential]


# =============================================================================
# METADATA
# =============================================================================


def get_expected_out_files() -> list[ExpectedFile]:
    """All expected output files. Used for dependency copying and verification."""
    return [
        ExpectedFile("mini_demo_data.json", "Curated subset of demo data (~5-10 examples)"),
        ExpectedFile("code_demo.ipynb", "Jupyter notebook demo with gradually scaled parameters"),
    ]
