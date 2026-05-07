"""Artifact demo converters — one per file type."""

from .gen_lean_demo import create_proof_markdown, lean_playground_url
from .gen_md_demo import create_research_markdown
from .gen_py_demo import convert_to_notebook, github_to_colab_url
