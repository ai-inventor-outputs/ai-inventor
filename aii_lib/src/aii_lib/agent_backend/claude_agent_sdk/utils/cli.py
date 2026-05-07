"""
Run script for aii_lib agent backend.

Handles config loading, agent setup, and execution orchestration.
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger

from ..agent import Agent
from ..models import AgentOptions, AgentResponse


async def run_agent(
    prompts: str | list[str],
    config_path: str | Path | None = None,
    options: AgentOptions | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> AgentResponse:
    """
    Run the agent with prompts.

    Args:
        prompts: Single prompt string or list of prompts to execute
        config_path: Path to YAML config file (optional)
        options: AgentOptions instance (overrides config_path if provided)
        max_retries: Maximum retry attempts on failure (default: 3)
        retry_delay: Delay between retries in seconds (default: 2.0)

    Returns:
        AgentResponse with execution results

    Example:
        >>> result = await run_agent("Create a Python function", config_path="config.yaml")
        >>> print(result.final_response)
    """
    # Load options from config or use provided
    if options is None:
        if config_path:
            logger.info(f"Loading config from: {config_path}")
            options = AgentOptions.from_yaml(config_path)
        else:
            logger.info("Using default AgentOptions")
            options = AgentOptions()

    # Create and run agent
    agent = Agent(options)
    logger.info(f"Starting agent with {len(prompts) if isinstance(prompts, list) else 1} prompt(s)")

    result = await agent.run(prompts=prompts)

    logger.success("Agent completed successfully.")
    return result


def run_agent_sync(
    prompts: str | list[str],
    config_path: str | Path | None = None,
    options: AgentOptions | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> AgentResponse:
    """
    Synchronous wrapper for run_agent.

    Args:
        prompts: Single prompt string or list of prompts to execute
        config_path: Path to YAML config file (optional)
        options: AgentOptions instance (overrides config_path if provided)
        max_retries: Maximum retry attempts on failure (default: 3)
        retry_delay: Delay between retries in seconds (default: 2.0)

    Returns:
        AgentResponse with execution results

    Example:
        >>> result = run_agent_sync("Create a Python function", config_path="config.yaml")
        >>> print(result.final_response)
    """
    return asyncio.run(run_agent(prompts, config_path, options, max_retries, retry_delay))


def main():
    """
    CLI entry point for running the agent.

    Usage:
        python -m aii_lib.agent_backend --config config.yaml --prompt "Your prompt here"
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run Claude Agent")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (default: config.yaml)",
        default="config.yaml",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to execute (can be specified multiple times for sequences)",
        action="append",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts on failure (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between retries in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    if not args.prompt:
        logger.error("No prompt provided. Use --prompt to specify at least one prompt.")
        sys.exit(1)

    # Convert config path to Path object
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Run agent
    try:
        prompts = args.prompt if len(args.prompt) > 1 else args.prompt[0]
        run_agent_sync(
            prompts=prompts,
            config_path=config_path,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

    except Exception:
        logger.exception("Agent execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["main", "run_agent", "run_agent_sync"]
