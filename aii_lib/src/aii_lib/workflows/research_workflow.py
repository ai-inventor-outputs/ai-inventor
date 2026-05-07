"""ResearchWorkflow - Research workflow with tool support.

Supports two backends:
- OpenRouter (default): Tool loop with aii_web_tools__search, aii_web_tools__fetch
- Claude Agent: MCP tools, structured output via file

Pattern (OpenRouter):
1. LLM calls with tools
2. Tool loop runs until model stops or hits max_iterations
3. Force output with custom prompt
4. Return structured output

Pattern (Claude Agent):
1. Agent runs with MCP tools
2. Writes structured output to file
3. Validates against schema
4. Retries with feedback if invalid

Usage:
    # OpenRouter
    async with OpenRouterClient(api_key=key, model=model) as client:
        result = await research_workflow(
            client=client,
            prompt="Research...",
            system="You are...",
            response_format=Hypothesis,
            config=ResearchWorkflowConfig(...),
            task_id="task_x", task_name="Task X",
        )

    # Claude Agent
    result = await research_workflow(
        prompt="Research...",
        system="You are...",
        response_format=Hypothesis,
        use_claude_agent=True,
        claude_model="claude-sonnet-4-5",
        cwd=Path("./workspace"),
        task_id="task_x", task_name="Task X",
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel as _PydanticBaseModel

# Default tools for research
from aii_lib.abilities.endpoint_names import AII_WEB_FETCH, AII_WEB_SEARCH
from aii_lib.run import get_current_run
from aii_lib.run.messages import BaseMessage

from ..abilities.aii_ability import abilities_to_openai_tools
from ..llm_backend.openrouter.or_to_json import extract_output
from ..llm_backend.tool_loop import ToolLoopResult, _emit_summary, chat
from ..utils.agent_to_llm import ClaudeAgentToLLMStructOut

RESEARCH_TOOLS = [AII_WEB_SEARCH, AII_WEB_FETCH]


@dataclass
class ResearchWorkflowConfig:
    """Configuration for research workflow."""

    max_tool_iterations: int = 10
    force_output_prompt: str = ""  # Required for OpenRouter - caller must provide
    tools: list[str] = field(default_factory=RESEARCH_TOOLS.copy)
    web_search_backend: str = "auto"
    timeout: float = 300
    # Claude agent specific
    max_retries: int = 2  # Retries if schema validation fails


@dataclass
class ResearchWorkflowResult:
    """Result from research workflow."""

    output: dict | None  # Parsed JSON output
    output_text: str | None  # Raw text output
    tool_result: ToolLoopResult | None  # Full tool loop result (None for Claude agent)
    forced_output: bool = False  # True if output was forced after max iterations
    provider: str = "openrouter"  # "openrouter" or "claude_agent"

    @property
    def success(self) -> bool:
        """True if we have any output (parsed dict OR raw text)."""
        return self.output is not None or bool(self.output_text)


def _emit_status(text: str, level: str, *, task_id: str, task_name: str) -> None:
    """Emit a typed status message onto the active Run bus."""
    run = get_current_run()
    if run is None:
        return
    run._on(
        BaseMessage(
            type=level,
            text=text,
            task_id=task_id,
            parent_id=task_id,
            task_name=task_name,
        )
    )


async def research_workflow(
    prompt: str,
    system: str | None = None,
    response_format: type[_PydanticBaseModel] | None = None,
    *,
    # Backend selection
    use_claude_agent: bool = False,
    # OpenRouter params (required if use_claude_agent=False)
    client: object = None,
    reasoning_effort: str | None = None,
    # Claude agent params (used if use_claude_agent=True)
    claude_model: str = "claude-sonnet-4-5",
    claude_max_turns: int | None = None,
    cwd: Path | None = None,
    # Common params
    config: ResearchWorkflowConfig | None = None,
    task_id: str = "",
    task_name: str = "",
) -> ResearchWorkflowResult:
    """Run research workflow with automatic tool loop and structured output.

    Each emit lands on the active Run bus via ``current_run()._on(typed_msg)`` —
    pass ``task_id`` / ``task_name`` so messages carry their identity.

    Args:
        prompt: User prompt for research task
        system: System prompt
        response_format: Pydantic model for structured output
        use_claude_agent: If True, use Claude agent instead of OpenRouter
        client: OpenRouterClient instance (required if use_claude_agent=False)
        reasoning_effort: Reasoning effort level for OpenRouter
        claude_model: Claude model name (sonnet, opus, etc.)
        claude_max_turns: Max turns for Claude agent
        cwd: Working directory for Claude agent
        config: Research configuration
        task_id: Task ID stamped onto every emitted Run-bus message.
        task_name: Display name stamped alongside task_id.

    Returns:
        ResearchWorkflowResult with parsed output and metadata
    """
    cfg = config or ResearchWorkflowConfig()

    # =========================================================================
    # CLAUDE AGENT PATH
    # =========================================================================
    if use_claude_agent:
        return await _research_workflow_claude_agent(
            prompt=prompt,
            system=system or "",
            response_format=response_format,
            model=claude_model,
            max_turns=claude_max_turns,
            cwd=cwd or Path.cwd(),
            tools=cfg.tools,
            max_retries=cfg.max_retries,
            task_id=task_id,
            task_name=task_name,
        )

    # =========================================================================
    # OPENROUTER PATH
    # =========================================================================
    if client is None:
        raise ValueError("client is required when use_claude_agent=False")

    # Get tools
    tools = abilities_to_openai_tools(cfg.tools) if cfg.tools else None

    # Run tool loop WITHOUT structured output - let model research freely
    result = await chat(
        client=client,
        prompt=prompt,
        system=system,
        tools=tools,
        max_iterations=cfg.max_tool_iterations if tools else 1,
        response_format=None,  # No structured output during tool loop
        reasoning_effort=reasoning_effort,
        web_search_backend=cfg.web_search_backend,
        timeout=cfg.timeout,
        emit_summary=False,
        task_id=task_id,
        task_name=task_name,
    )

    messages = result.messages

    # Force output at the end to apply structured format
    force_reason = "Research complete"
    if result.hit_max_iterations and result.last_response_has_tool_calls:
        force_reason = f"Tool limit ({cfg.max_tool_iterations}) reached"

    _emit_status(
        f"{force_reason}, generating structured output...",
        "status_public_progress",
        task_id=task_id,
        task_name=task_name,
    )

    # Add force prompt to messages
    force_messages = [*messages, {"role": "user", "content": cfg.force_output_prompt}]

    # Final call with structured output
    result = await chat(
        client=client,
        messages=force_messages,
        tools=None,
        response_format=response_format,
        conversation_stats=result.stats,
        timeout=cfg.timeout,
        emit_summary=False,
        task_id=task_id,
        task_name=task_name,
    )

    # Extract output
    raw_text = extract_output(result.response)
    output_text = raw_text.strip() if raw_text else None

    # Try to parse as JSON
    output = None
    if output_text:
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError:
            from ..llm_backend.openrouter.or_to_json import extract_json_from_text

            json_text = extract_json_from_text(output_text)
            if json_text:
                output = json.loads(json_text)

    _emit_summary(result.stats, client, task_id=task_id, task_name=task_name)

    return ResearchWorkflowResult(
        output=output,
        output_text=output_text,
        tool_result=result,
        forced_output=True,
        provider="openrouter",
    )


async def _research_workflow_claude_agent(
    prompt: str,
    system: str,
    response_format: type[_PydanticBaseModel] | None,
    model: str,
    max_turns: int | None,
    cwd: Path,
    tools: list[str],  # noqa: ARG001 — Claude path uses MCP tools, not these
    max_retries: int,
    task_id: str,
    task_name: str,
) -> ResearchWorkflowResult:
    """Claude agent path for research workflow."""
    effective_task_id = task_id or "research"
    effective_task_name = task_name or effective_task_id
    output_file = f"./{effective_task_id}_output.json"

    mcp_servers = None

    try:
        async with ClaudeAgentToLLMStructOut(
            schema=response_format,
            output_file=output_file,
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            max_retries=max_retries,
            system_prompt=system,
            mcp_servers=mcp_servers,
            task_id=effective_task_id,
            task_name=effective_task_name,
        ) as agent:
            result = await agent.run(prompt)

            if result.data:
                output = result.data if isinstance(result.data, dict) else result.data.model_dump()
                return ResearchWorkflowResult(
                    output=output,
                    output_text=json.dumps(output),
                    tool_result=None,
                    forced_output=False,
                    provider="claude_agent",
                )

            return ResearchWorkflowResult(
                output=None,
                output_text=None,
                tool_result=None,
                forced_output=False,
                provider="claude_agent",
            )

    except Exception as e:
        _emit_status(
            f"Claude agent failed: {e}",
            "status_public_error",
            task_id=effective_task_id,
            task_name=effective_task_name,
        )
        raise


__all__ = [
    "RESEARCH_TOOLS",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "research_workflow",
]
