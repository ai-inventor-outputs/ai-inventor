"""Post-loop paper generation — demos, viz, deploy, full paper.

No package-level re-exports — callers import from the specific module
(e.g. ``from aii_pipeline.steps._4_gen_paper_repo.gen_paper_repo import
GenPaperRepoGroup``). Mirrors the pattern in sibling phase packages
to avoid the partial-init circular-import trap when a leaf module is
loaded dynamically (see ``_3_invention_loop/__init__.py``).
"""
