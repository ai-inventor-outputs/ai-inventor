"""
Agent Registry - Defines available agents and their locations.

Agents are defined as AgentDefinition objects that point to .md files.
Agents are loaded from the project's .claude/agents/ directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class AgentDefinition:
    """
    Definition of an agent that can be loaded into execution.

    Attributes:
        name: Agent name (e.g., "math-solver", "text-analyzer")
        path: Path to the agent .md file
        source: Whether agent is from project or other source
    """

    name: str
    path: Path
    source: Literal["project"]

    def __repr__(self) -> str:
        return f"Agent({self.name})"


def _find_project_root() -> Path | None:
    """Find the project root by looking for .claude/ directory.

    Searches upward from the current file.
    """
    current = Path(__file__).resolve()

    # Search up to 10 levels
    for _ in range(10):
        current = current.parent
        if (current / ".claude" / "agents").exists():
            return current
        # Stop at filesystem root
        if current.parent == current:
            break

    return None


# Get the agents directory from project .claude/agents/
PROJECT_ROOT = _find_project_root()

if PROJECT_ROOT is not None:
    AGENTS_DIR = PROJECT_ROOT / ".claude" / "agents"
else:
    # Fallback - no agents directory found
    AGENTS_DIR = None


def _discover_agents(agents_dir: Path | None) -> dict[str, AgentDefinition]:
    """
    Discover all agents in the agents directory.

    Searches for .md files with YAML frontmatter containing 'name' field.
    """
    agents = {}

    if agents_dir is None or not agents_dir.exists():
        return agents

    # Look for all .md files in agents directory
    for agent_md in agents_dir.glob("*.md"):
        # Extract agent name from frontmatter or filename
        # For now, use filename without extension as agent name
        agent_name = agent_md.stem

        agents[agent_name] = AgentDefinition(name=agent_name, path=agent_md, source="project")

    return agents


# Discover all project agents
_all_agents = _discover_agents(AGENTS_DIR)


# Export individual agent objects for convenient imports
math_solver = _all_agents.get("math-solver")
quick_calc = _all_agents.get("quick-calc")
math_tutor = _all_agents.get("math-tutor")
text_analyzer = _all_agents.get("text-analyzer")
text_transformer = _all_agents.get("text-transformer")
palindrome_checker = _all_agents.get("palindrome-checker")
text_master = _all_agents.get("text-master")

# Export list of all available agents
ALL_AGENTS = list(_all_agents.values())


def get_agent(name: str) -> AgentDefinition | None:
    """Get an agent by name."""
    return _all_agents.get(name)


def list_agents() -> list[str]:
    """List all available agent names."""
    return sorted(_all_agents.keys())
