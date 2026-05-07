"""Lazy singleton state for abilities — credentials, usage.

All state initializes on first use, not at import time.
"""


def ensure_credentials_state() -> None:
    """Initialize account manager + usage polling on first use."""
    from aii_lib.abilities.ability_server.credentials import init_credentials_state

    init_credentials_state()
