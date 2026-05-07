"""
aii_lib abilities — Skills and Tools for AI agents.

- ability_server/: HTTP client + credential management
- aii_ability.py: @aii_ability decorator for registering skill functions
- Server is now part of aii_server (Django)
"""

from . import ability_server

__all__ = ["ability_server"]
