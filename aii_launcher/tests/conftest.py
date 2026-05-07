"""Pytest config for aii_launcher tests.

Mark every test in this dir as ``unit`` so the default pytest filter
``-m "not integration and not speed_check"`` keeps them in the suite.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.unit)
