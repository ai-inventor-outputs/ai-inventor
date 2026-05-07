"""Knowledge graph generation workflow with verification and retry.

Generates knowledge graph triples from research papers with:
1. Initial prompt to extract triples
2. Wikipedia URL verification using aii_web_tools__fetch
3. Retry loop with conversation continuity for failed URLs

Uses aii_web_tools__search and aii_web_tools__fetch MCP tools for web access.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from ...agent_backend import Agent, AgentOptions
from ...run import emit, get_current_run
from .verify import verify_wikipedia_urls


@dataclass
class GenKGConfig:
    """Configuration for knowledge graph generation."""

    # Task identification
    paper_id: int
    paper_index: int
    title: str
    abstract: str

    # Tree position — Run.start_task requires the owning Module's id
    parent_module_id: str

    # Prompt
    prompt: str
    system_prompt: str | None = None

    # Agent settings
    model: str = "claude-haiku-4-5"
    max_turns: int | None = None
    agent_timeout: int | None = None  # Timeout for entire agent run (None = no timeout)
    agent_retries: int = 3  # Retries for entire agent on failure
    seq_prompt_timeout: int | None = None  # Timeout per prompt (None = no timeout)
    seq_prompt_retries: int = 3
    cwd: str | Path = "./"

    # MCP tools config - uses aii_web_tools__search and aii_web_tools__fetch by default
    mcp_servers: dict | None = None  # Custom MCP config (e.g., context7, HF)
    allowed_tools: list[str] | None = None  # Tool restrictions
    disallowed_tools: list[str] | None = (
        None  # Extra tools to block (merged with workflow defaults)
    )

    # Structured output
    response_schema: type[BaseModel] | None = None

    # Verification settings
    verify_retries: int = 2  # Retries for URL verification failures
    min_valid_urls: int = 0  # Minimum valid URLs before restructure vs search again

    # Retry prompt builder
    build_retry_prompt_fn: Callable[[dict], str] | None = None


@dataclass
class GenKGResult:
    """Result from knowledge graph generation."""

    paper_id: int
    paper_index: int
    title: str
    triples: list[dict] | None = None
    paper_type: str | None = None
    verified: bool = False
    verification_result: dict | None = None
    retry_attempts: int = 0
    run_dir: str | None = None
    error: str | None = None


def _fallback_retry_prompt(verification: dict) -> str:
    """Fallback retry prompt if none provided. Prefer passing build_retry_prompt_fn."""
    failed = verification.get("failed_triples", [])
    if not failed:
        return "Some Wikipedia URLs were invalid. Please fix them using WebSearch to find correct URLs."

    lines = ["The following Wikipedia URLs are invalid:\n"]
    for item in failed[:5]:
        triple = item.get("triple", {})
        name = triple.get("name", "Unknown")
        url = triple.get("wikipedia_url", "No URL")
        lines.append(f"- {name}: {url}")

    if len(failed) > 5:
        lines.append(f"... and {len(failed) - 5} more")

    lines.append(
        '\nUse WebSearch with allowed_domains=["en.wikipedia.org"] to find correct URLs and update triples_output.json.'
    )
    return "\n".join(lines)


async def generate_kg_triples(
    config: GenKGConfig,
) -> GenKGResult:
    """Generate knowledge graph triples with URL verification and retry.

    This workflow:
    1. Runs initial prompt to extract triples
    2. Verifies Wikipedia URLs exist using aii_web_tools__fetch
    3. Retries with conversation continuity if URLs are invalid

    Args:
        config: Generation configuration

    Returns:
        GenKGResult with triples and verification status
    """
    # Build task identifiers
    task_id = f"triples_paper_idx{config.paper_index}"
    task_name = f"triples_paper_idx{config.paper_index}"

    result = GenKGResult(
        paper_id=config.paper_id,
        paper_index=config.paper_index,
        title=config.title,
    )

    # Emit helper — route onto the Run bus.
    def emit_msg(level: str, msg: str):
        run = get_current_run()
        if run is None:
            return
        method = {
            "ERROR": emit.status_public_error,
            "WARNING": emit.status_public_warning,
            "WARN": emit.status_public_warning,
            "SUCCESS": emit.status_public_success,
            "INFO": emit.status_private_info,
        }.get(level.upper(), emit.status_private_info)
        method(msg)

    # Each Agent.run() emits its own agent_summary via streaming.py's
    # message_callback path; NodeStats sums them via apply_leaf_summary.
    emit.start_task(name=task_name, parent_module_id=config.parent_module_id)

    # One-shot path normalization at workflow init; not a hot per-message call.
    cwd = Path(config.cwd).resolve()

    # Create Agent with SDK native structured output
    options = AgentOptions(
        model=config.model,
        cwd=cwd,
        max_turns=config.max_turns,
        agent_timeout=config.agent_timeout,
        agent_retries=config.agent_retries,
        seq_prompt_timeout=config.seq_prompt_timeout,
        seq_prompt_retries=config.seq_prompt_retries,
        permission_mode="bypassPermissions",
        system_prompt=config.system_prompt,
        continue_seq_item=True,  # Continue conversation between prompts
        mcp_servers=config.mcp_servers,
        allowed_tools=config.allowed_tools,
        disallowed_tools=config.disallowed_tools,
        # Increase buffer size for large web search results (default 1MB too small)
        max_buffer_size=10 * 1024 * 1024,  # 10MB
        run_id=task_id,
        agent_context=task_name,
        # Structured JSON output (SDK native)
        output_format=config.response_schema.to_struct_output() if config.response_schema else None,
        # Load skills from .claude/skills/
        setting_sources=["project"],
    )

    agent = Agent(options)

    try:
        # Initial prompt
        response = await agent.run(config.prompt)

        output = response.structured_output
        if not output:
            result.error = "No output generated"
            emit_msg("WARNING", "No output generated")
            emit.end_task(task_id, name=task_name, status="failed", text="No output")
            return result

        # Extract triples from output
        if isinstance(output, dict):
            triples = output.get("triples", [])
            paper_type = output.get("paper_type")
        else:
            triples = getattr(output, "triples", [])
            paper_type = getattr(output, "paper_type", None)

        result.triples = triples
        result.paper_type = paper_type

        if not triples:
            result.error = "No triples extracted"
            emit_msg("WARNING", "No triples in output")
            emit.end_task(task_id, name=task_name, status="failed", text="No triples")
            return result

        # URL verification retry loop
        def verify_callback(msg: str):
            emit_msg("VERIFY", msg)

        for attempt in range(config.verify_retries + 1):
            result.retry_attempts = attempt

            # Verify Wikipedia URLs
            verification = verify_wikipedia_urls(triples, callback=verify_callback)
            result.verification_result = verification
            result.verified = verification.get("valid", False)

            if verification.get("valid"):
                status_text = "VALID" + (f" (retry {attempt})" if attempt > 0 else "")
                emit_msg("SUCCESS", f"All URLs verified: {status_text}")
                emit.end_task(task_id, name=task_name, status="done", text=status_text)
                return result

            # Retry if attempts left
            if attempt < config.verify_retries:
                valid_count = verification.get("verified", 0)
                failed_count = verification.get("failed", 0)

                # Build retry prompt
                if config.build_retry_prompt_fn:
                    retry_prompt = config.build_retry_prompt_fn(verification)
                else:
                    retry_prompt = _fallback_retry_prompt(verification)

                emit_msg(
                    "RETRY",
                    f"URLs failed ({valid_count} valid, {failed_count} failed), retrying...",
                )

                # Retry with conversation continuity
                full_retry_prompt = (
                    f"Your previous response had invalid Wikipedia URLs.\n\n{retry_prompt}"
                )
                response = await agent.run(full_retry_prompt)

                output = response.structured_output
                if output:
                    if isinstance(output, dict):
                        triples = output.get("triples", [])
                        paper_type = output.get("paper_type")
                    else:
                        triples = getattr(output, "triples", [])
                        paper_type = getattr(output, "paper_type", None)
                    result.triples = triples
                    result.paper_type = paper_type
                else:
                    emit_msg("RETRY", "Retry produced no output, keeping previous")

        # All retries exhausted
        if verification:
            failed = verification.get("failed", 0)
            total = verification.get("total", 0)
            status_text = f"INVALID ({failed}/{total} URLs failed)"
            emit_msg("WARNING", f"URLs invalid after retries: {status_text}")
            emit.end_task(task_id, name=task_name, status="failed", text=status_text)

        return result

    except TimeoutError:
        emit.status_public_error("Timeout")
        emit.end_task(task_id, name=task_name, status="failed", text="Timeout")
        raise

    except Exception as e:
        emit.status_public_error(f"Error: {e}")
        emit.end_task(task_id, name=task_name, status="failed", text=f"Error: {e}")
        raise


__all__ = [
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
]
