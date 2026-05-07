"""Seed hypothesis generation step.

No package-level re-exports — callers import from the specific module
(e.g. ``from aii_pipeline.steps._1_seed_hypo.seed_hypo import
run_seed_hypo_module``). Mirrors the pattern in sibling phase packages
to avoid the partial-init circular-import trap when a leaf module is
loaded dynamically (see ``_3_invention_loop/__init__.py``).
"""
