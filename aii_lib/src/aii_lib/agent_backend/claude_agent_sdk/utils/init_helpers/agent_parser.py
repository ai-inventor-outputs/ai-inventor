"""
Agent Parser - Converts markdown agent files to programmatic definitions.

Parses .md agent files with YAML frontmatter into AgentDefinition dataclass.
"""

import re
from pathlib import Path

from claude_agent_sdk.types import AgentDefinition


def parse_agent_markdown(file_path: Path) -> AgentDefinition:
    """
    Parse a markdown agent file into SDK AgentDefinition dataclass.

    Format:
    ---
    name: agent-name
    description: Description text
    tools: Read, Write, Bash
    model: sonnet
    ---

    Prompt content here...

    Returns:
        AgentDefinition(
            description='Description text',
            prompt='Prompt content here...',
            tools=['Read', 'Write', 'Bash'],
            model='sonnet'
        )
    """
    content = file_path.read_text()

    # Extract YAML frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError(f"Agent file {file_path} missing YAML frontmatter")

    frontmatter_text, prompt = match.groups()

    # Parse frontmatter (simple YAML parsing)
    frontmatter = {}
    for line in frontmatter_text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    # Parse tools list
    tools = None
    if "tools" in frontmatter:
        tools_str = frontmatter["tools"]
        tools = [t.strip() for t in tools_str.split(",")]

    # Build agent definition
    agent_def = AgentDefinition(
        description=frontmatter.get("description", ""),
        prompt=prompt.strip(),
        tools=tools,
        model=frontmatter.get("model"),
    )

    return agent_def
