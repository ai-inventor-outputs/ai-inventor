"""Iterative invention loop — strategy, plan, artifact, paper text, review, hypothesis update.

No package-level re-exports — callers import from the specific module
(e.g. ``from aii_pipeline.steps._3_invention_loop.invention_loop import
InventionLoopGroup``). This avoids the legacy cycle where loading
this package via a leaf (e.g. ``executors.artifact_validation``) would
pre-load ``invention_loop.py`` BEFORE ``aii_pipeline.run.scaffold`` had
a chance to bind ``InventionLoopCtx``.
"""
