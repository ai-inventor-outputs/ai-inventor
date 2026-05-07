"""Hypothesis generation + review loop.

No package-level re-exports — callers import from the specific module
(e.g. ``from aii_pipeline.steps._2_hypo_loop.hypo_loop import
HypoLoopGroup``). Mirrors the pattern in sibling phase packages to
avoid the partial-init circular-import trap when a leaf module is
loaded dynamically (see ``_3_invention_loop/__init__.py``).
"""
