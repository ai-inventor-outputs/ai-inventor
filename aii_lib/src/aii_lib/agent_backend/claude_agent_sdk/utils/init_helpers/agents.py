"""Agent registry and loader - consolidated from agents/ module."""

from .agents_loader import (
    cleanup_agents,
    prepare_agents,
)
from .agents_registry import (
    AGENTS_DIR,
    ALL_AGENTS,
    PROJECT_ROOT,
    AgentDefinition,
    get_agent,
    list_agents,
    math_solver,
    math_tutor,
    palindrome_checker,
    quick_calc,
    text_analyzer,
    text_master,
    text_transformer,
)
