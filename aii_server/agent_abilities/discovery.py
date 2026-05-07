"""Skill discovery for Django — reuses aii_lib discovery functions.

On startup, scans .claude/skills/aii-*/scripts/*.py, imports them
(triggering @aii_ability decorators), and populates the global registry.
"""

from aii_lib.abilities.ability_server.discovery import (
    _add_to_sys_path,
    _discover_skill_dirs,
    _ensure_venvs,
    _import_scripts,
    _run_env_checks,
)
from aii_lib.abilities.aii_ability import get_registry
from loguru import logger

_discovered = False


def discover_abilities() -> int:
    """Discover and register all @aii_ability functions.

    Safe to call multiple times — only runs once.
    Returns count of discovered abilities.
    """
    global _discovered
    if _discovered:
        return len(get_registry())

    log = logger.bind(source="abilities")

    # 1. Find skill directories
    dirs = _discover_skill_dirs()
    if not dirs:
        log.warning("No skill directories found")
        _discovered = True
        return 0

    # 2. Add to sys.path so sibling imports work
    _add_to_sys_path(dirs)

    # 3. Import scripts (triggers @aii_ability decorators → fills registry)
    count = _import_scripts(dirs)
    log.info(f"Imported {count} skill scripts")

    # 4. Ensure venvs for skills that need them
    _ensure_venvs()

    # 5. Run environment checks
    _run_env_checks()

    registry = get_registry()
    log.info(f"Bootstrap complete: {len(registry)} abilities discovered")

    _discovered = True
    return len(registry)
