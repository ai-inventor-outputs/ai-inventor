"""
CLI entry point for aii_lib agent backend.

Usage:
    python -m aii_lib.agent_backend --help
    python -m aii_lib.agent_backend run "Your prompt" --config config.yaml
    python -m aii_lib.agent_backend validate config.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from . import Agent, AgentOptions


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"


async def run_command(args: argparse.Namespace) -> None:
    """Run a prompt with the agent."""
    from loguru import logger

    from .config import apply_cli_overrides, load_config_from_yaml

    # Debug: Show what args we received
    logger.debug(
        f"Received args: config={args.config!r}, model={args.model!r}, max_turns={args.max_turns!r}"
    )

    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading config from: {config_path.resolve()}")
        options = load_config_from_yaml(config_path, _skip_post_init_log=True)

        # Apply all CLI args as overrides (automatically handles all fields)
        apply_cli_overrides(options, vars(args))
    else:
        logger.debug("No config file specified, using default AgentOptions")
        # No config file - create default options (skip post-init log since apply_cli_overrides will log)
        options = AgentOptions(permission_mode="bypassPermissions", _skip_post_init_log=True)

        # Apply all CLI args as overrides
        apply_cli_overrides(options, vars(args))

    # Create agent
    agent = Agent(options)

    # Run prompts (single or multiple)
    result = await agent.run(args.prompts)

    # Verbose mode: show summary status only (token detail now lives in the run tree).
    if args.verbose:
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.CYAN}AGENT RESULT{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"  {Colors.CYAN}Failed:{Colors.RESET} {Colors.GREEN}{result.failed}{Colors.RESET}")
        if result.error_message:
            print(f"  {Colors.CYAN}Error:{Colors.RESET} {result.error_message}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")


def validate_command(args: argparse.Namespace) -> None:
    """Validate a config file."""
    from .config import load_config_from_yaml

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"{Colors.YELLOW}Error: Config file not found: {args.config}{Colors.RESET}")
        sys.exit(1)

    try:
        options = load_config_from_yaml(config_path)
        print(f"{Colors.GREEN}✓ Config is valid: {args.config}{Colors.RESET}")

        # Show summary
        print(f"\n{Colors.CYAN}Configuration Summary:{Colors.RESET}")
        print(
            f"  {Colors.CYAN}Working directory:{Colors.RESET} {Colors.GREEN}{options.cwd or 'current'}{Colors.RESET}"
        )
        print(
            f"  {Colors.CYAN}Max turns:{Colors.RESET} {Colors.GREEN}{options.max_turns}{Colors.RESET}"
        )
        print(
            f"  {Colors.CYAN}Permission mode:{Colors.RESET} {Colors.GREEN}{options.permission_mode}{Colors.RESET}"
        )
        print(
            f"  {Colors.CYAN}Session type:{Colors.RESET} {Colors.GREEN}{options.session_type.value}{Colors.RESET}"
        )
        print(f"  {Colors.CYAN}Model:{Colors.RESET} {Colors.GREEN}{options.model}{Colors.RESET}")

        if options.custom_tool_files:
            print(
                f"  {Colors.CYAN}Custom tools:{Colors.RESET} {Colors.GREEN}{len(options.custom_tool_files)} files{Colors.RESET}"
            )
        if options.custom_agent_files:
            print(
                f"  {Colors.CYAN}Custom agents:{Colors.RESET} {Colors.GREEN}{len(options.custom_agent_files)} files{Colors.RESET}"
            )

        sys.exit(0)
    except Exception as e:
        logger.error(f"{Colors.YELLOW}✗ Config validation failed: {e}{Colors.RESET}")
        sys.exit(1)


def info_command(args: argparse.Namespace) -> None:
    """Show package information."""
    from . import __version__

    print(f"""
{Colors.CYAN}aii_lib.agent_backend{Colors.RESET} {Colors.GREEN}v{__version__}{Colors.RESET}

A Python wrapper around Claude Agent SDK with streaming mode support.

{Colors.CYAN}Features:{Colors.RESET}
  {Colors.GREEN}•{Colors.RESET} YAML configuration
  {Colors.GREEN}•{Colors.RESET} Custom tools from Python files
  {Colors.GREEN}•{Colors.RESET} Custom agents from YAML files
  {Colors.GREEN}•{Colors.RESET} Session management (NEW/RESUME/FORK)
  {Colors.GREEN}•{Colors.RESET} Usage tracking (tokens & costs)
  {Colors.GREEN}•{Colors.RESET} Sequential execution

{Colors.CYAN}Documentation:{Colors.RESET} README.md
{Colors.CYAN}Examples:{Colors.RESET} examples/
""")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="aii_agent",
        description="AII Agent - Run Claude Agent SDK with streaming support",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run one or more prompts")
    run_parser.add_argument("prompts", nargs="+", help="One or more prompts to run sequentially")
    run_parser.add_argument("-c", "--config", help="Path to YAML config file")
    run_parser.add_argument(
        "-t",
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: 1000)",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Claude model to use (e.g., haiku, sonnet, opus)",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed token usage",
    )
    run_parser.add_argument(
        "--session-type",
        choices=["new", "resume", "fork"],
        help="Session type: new (default), resume, or fork",
    )
    run_parser.add_argument(
        "--session-id",
        help="Session ID to resume or fork from (required for resume/fork)",
    )
    run_parser.add_argument(
        "--continue-seq-item",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Continue conversation for 2nd+ prompts in sequence (true/false)",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config", help="Path to YAML config file")

    # Info command
    subparsers.add_parser("info", help="Show package information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute command
    if args.command == "run":
        asyncio.run(run_command(args))
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "info":
        info_command(args)


if __name__ == "__main__":
    main()
