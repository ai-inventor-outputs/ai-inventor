"""Leaf utilities for ``aii_pipeline.run``.

Lives in its own subpackage so phase modules can import from here
without triggering ``aii_pipeline.run.__init__`` (which would re-enter
the partially-loaded phase module via the now-removed scaffold
re-exports). With the parent package init thinned, this is just a
clean home for read-only helpers.
"""
