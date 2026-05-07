"""Configuration loading and management utilities."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import AgentOptions

__all__ = ["apply_cli_overrides", "load_config_from_yaml"]


def load_config_from_yaml(
    yaml_path: str | Path, _skip_post_init_log: bool = False
) -> "AgentOptions":
    """
    Load AgentOptions from YAML config file.

    Args:
        yaml_path: Path to YAML config file
        _skip_post_init_log: Internal flag to skip __post_init__ logging (used by CLI)

    Returns:
        AgentOptions instance

    Example:
        >>> from aii_lib.agent_backend.claude_agent_sdk.config import load_config_from_yaml
        >>> options = load_config_from_yaml("config.yaml")
        >>> agent = Agent(options)
        >>> result = await agent.run("Create a Python project")
    """
    from pathlib import Path as PathLib

    import yaml

    from .models import AgentOptions, SessionType

    yaml_path = PathLib(yaml_path).expanduser()

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert session_type string to enum
    if "session_type" in config:
        session_type_str = config["session_type"]
        if isinstance(session_type_str, str):
            config["session_type"] = SessionType(session_type_str.lower())

    # Convert Path objects
    if "cwd" in config and config["cwd"] is not None:
        config["cwd"] = PathLib(config["cwd"])

    if "add_dirs" in config:
        config["add_dirs"] = [PathLib(directory) for directory in config["add_dirs"]]

    # Handle system_prompt dict (SystemPromptPreset)
    # YAML will load it as dict, which is already correct format

    # Create AgentOptions with config values
    config.pop("_skip_post_init_log", None)
    return AgentOptions(**config)


def apply_cli_overrides(options: "AgentOptions", cli_args: dict[str, Any]) -> None:
    """
    Apply CLI arguments as overrides to config options.

    Args:
        options: AgentOptions instance to modify
        cli_args: Dictionary of CLI arguments (typically from vars(args))

    Example:
        >>> from aii_lib.agent_backend.claude_agent_sdk.config import load_config_from_yaml, apply_cli_overrides
        >>> options = load_config_from_yaml("config.yaml")
        >>> apply_cli_overrides(options, vars(args))
    """
    from loguru import logger

    from .models import SessionType

    # CLI-only arguments that should not be applied to AgentOptions
    SKIP_ARGS = {"config", "prompt", "prompts", "command", "verbose"}

    overridden_fields = []

    for arg_name, arg_value in cli_args.items():
        # Skip None values (not explicitly provided)
        if arg_value is None:
            continue

        # Skip CLI-only arguments
        if arg_name in SKIP_ARGS:
            continue

        # Skip if field doesn't exist in AgentOptions
        if not hasattr(options, arg_name):
            continue

        # Track what we're overriding
        old_value = getattr(options, arg_name)

        # Handle session_type conversion (string -> enum)
        if arg_name == "session_type" and isinstance(arg_value, str):
            arg_value = SessionType(arg_value.lower())

        # Apply the override
        setattr(options, arg_name, arg_value)

        # Log if we actually changed something
        if old_value != arg_value:
            overridden_fields.append(f"{arg_name}: {old_value} → {arg_value}")

    # Show user what was overridden
    if overridden_fields:
        logger.info(f"CLI overrides: {', '.join(overridden_fields)}")

    # (AgentOptions no longer exposes _log_config)
