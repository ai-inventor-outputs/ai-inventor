"""ClaudeAgentToLLMStructOut - Use Claude Agent as an LLM with structured JSON output.

Wraps the Claude Agent SDK to behave like an LLM client that outputs
structured JSON files matching Pydantic schemas.

Uses aii_lib for:
- Agent: Claude Code SDK wrapper
- AgentOptions: Agent configuration
- start_task / emit_summary / end_task_*: Pre/post-agent helpers

Usage:
    from pydantic import BaseModel
    from aii_lib.utils import ClaudeAgentToLLMStructOut

    class MySchema(BaseModel):
        title: str
        score: float

    async with ClaudeAgentToLLMStructOut(
        schema=MySchema,
        output_file="result.json",
        cwd="/my/project",
        system_prompt="You are a helpful assistant.",
        task_id="gen-1",
        task_name="gen-1",
    ) as agent:
        result = await agent.run("Analyze this code")
        print(result.data)  # Validated dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

# ``aii_lib.agent_backend`` imports are deferred to inside the methods
# that use them. Loading this module at top of ``aii_lib.utils`` would
# otherwise pull ``agent_backend`` in mid-load and cycle (agent_backend
# → claude/agent → agents/claude/utils/usage → aii_lib.utils.retry →
# aii_lib.utils.__init__ → agent_to_llm → here → agent_backend).
# Annotations stay as forward refs via ``from __future__ import annotations``.
if TYPE_CHECKING:
    from aii_lib.agent_backend import (  # noqa: TC004 — AgentResponse is used at runtime as a dataclass field type but kept here to avoid the agent_backend → utils import cycle described above
        Agent,
        AgentOptions,
        AgentResponse,
    )


@dataclass
class ClaudeAgentToLLMStructOutResult:
    """Result from ClaudeAgentToLLMStructOut execution."""

    data: dict  # Validated JSON data
    raw_response: AgentResponse  # Full agent response
    output_path: Path  # Where the file was written
    attempts: int  # Number of attempts needed


OUTPUT_INSTRUCTION_TEMPLATE = """

---

Output the result as JSON to: `{output_file}`

JSON Schema:
```json
{schema_json}
```

IMPORTANT: This task is NOT complete until you use the Write tool to create `{output_file}`.
"""


FEEDBACK_TEMPLATE = """
The output file has validation errors:

{errors}

Fix `{output_file}` to match the schema. All fields are required unless marked optional.
""".strip()


class ClaudeAgentToLLMStructOut:
    """Use Claude Agent as an LLM with structured JSON output.

    This class wraps the Claude Agent SDK to:
    1. Build prompts that ensure file output
    2. Validate output against Pydantic schema
    3. Retry with feedback if validation fails

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from aii_lib.utils import ClaudeAgentToLLMStructOut
        >>>
        >>> class Analysis(BaseModel):
        ...     summary: str
        ...     score: float = Field(ge=0, le=1)
        >>>
        >>> async with ClaudeAgentToLLMStructOut(
        ...     schema=Analysis,
        ...     output_file="analysis.json",
        ...     cwd="/my/project",
        ...     system_prompt="You are a code reviewer.",
        ...,
        ...     task_id="analyze-1",
        ...     task_name="analyze-1",
        ... ) as agent:
        ...     result = await agent.run("Analyze the code quality")
        ...     print(result.data["summary"])
    """

    def __init__(
        self,
        schema: type[BaseModel],
        output_file: str = "./output.json",
        cwd: str | Path = "./",
        model: str = "claude-sonnet-4-5",
        max_turns: int = 50,
        max_retries: int = 2,
        timeout_seconds: int = 3600,
        system_prompt: str | None = None,
        mcp_servers: dict | None = None,
        task_id: str | None = None,
        task_name: str | None = None,
    ):
        """Initialize ClaudeAgentToLLMStructOut.

        Args:
            schema: Pydantic model class for output validation
            output_file: Relative path for JSON output (e.g., "./result.json")
            cwd: Working directory for agent execution
            model: Claude model (sonnet, opus, haiku)
            max_turns: Max turns per attempt
            max_retries: Max retry attempts on validation failure
            timeout_seconds: Timeout for agent execution (default: 3600)
            system_prompt: System prompt for the agent
            mcp_servers: Optional MCP server config (e.g., context7, HF) instance for logging
            task_id: Task ID
            task_name: Task name for display in logs
        """
        self.schema = schema
        self.output_file = output_file
        self.cwd = Path(cwd).resolve()
        self.model = model
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt
        self.mcp_servers = mcp_servers

        self.task_id = task_id
        self.task_name = task_name

        self._agent: Agent | None = None

    def _build_prompt(self, prompt: str) -> str:
        """Append output instructions to the original prompt."""
        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)

        # Keep original prompt intact, just append output instructions
        output_instructions = OUTPUT_INSTRUCTION_TEMPLATE.format(
            output_file=self.output_file,
            schema_json=schema_json,
        )

        return prompt + output_instructions

    def _build_feedback(self, errors: str) -> str:
        """Build feedback prompt for retry."""
        return FEEDBACK_TEMPLATE.format(
            errors=errors,
            output_file=self.output_file,
        )

    def _validate_output(self) -> tuple[bool, str, dict | None]:
        """Validate the output file against schema.

        Returns:
            (is_valid, error_message, data)
        """
        output_path = self.cwd / self.output_file.lstrip("./")

        if not output_path.exists():
            return (
                False,
                f"File `{self.output_file}` not found. Use Write tool to create it.",
                None,
            )

        try:
            content = output_path.read_text()
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", None

        try:
            self.schema(**data)
            return True, "", data
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"- {loc}: {err['msg']}")
            return False, "\n".join(errors), data

    def _emit(self, message: str, level: str = "INFO") -> None:
        """Route a status message onto the Run bus."""
        if not (self.task_id and self.task_name):
            return
        from aii_lib.run import emit, get_current_run

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
        method(message)

    def _get_agent_options(self) -> AgentOptions:
        """Build AgentOptions for execution."""
        from aii_lib.agent_backend import AgentOptions  # deferred — cycle break

        options = AgentOptions(
            model=self.model,
            cwd=self.cwd,
            max_turns=self.max_turns,
            seq_prompt_timeout=self.timeout_seconds,
            permission_mode="bypassPermissions",
            system_prompt=self.system_prompt,
            continue_seq_item=True,  # Continue conversation between prompts
            run_id=self.task_id,
            agent_context=self.task_name,  # Display name for logs
            # Load skills from .claude/skills/
            setting_sources=["project"],
        )
        if self.mcp_servers:
            options.mcp_servers = self.mcp_servers
        return options

    async def run(self, prompt: str) -> ClaudeAgentToLLMStructOutResult:
        """Execute task and return validated JSON output.

        Args:
            prompt: The user prompt (task description)

        Returns:
            ClaudeAgentToLLMStructOutResult with validated data and metadata

        Raises:
            ValueError: If validation fails after all retries
            asyncio.TimeoutError: If execution times out
        """
        # Lazy aii_lib.agent_backend import — see module docstring for why
        from aii_lib.agent_backend import (
            Agent,
            end_task_error,
            end_task_failure,
            end_task_success,
            end_task_timeout,
            start_task,
        )

        # Start task (must be before any emit_message calls)
        start_task(self.task_id or "", self.task_name or self.task_id or "")

        # Log execution info
        self._emit(f"Executing with schema: {self.schema.__name__}")
        self._emit(f"Model: claude-{self.model}")
        self._emit(f"Workspace: {self.cwd}")

        # Build initial prompt
        full_prompt = self._build_prompt(prompt)

        # Create agent
        options = self._get_agent_options()
        self._agent = Agent(options)

        attempts = 0
        error_msg = ""

        try:
            for attempt in range(self.max_retries + 1):
                attempts += 1

                if attempt == 0:
                    # Initial attempt - new conversation
                    self._emit(f"Running initial prompt (attempt {attempts})")
                    response = await self._agent.run(full_prompt)
                else:
                    # Retry with feedback - continues same conversation
                    self._emit(
                        f"Retry {attempt}/{self.max_retries} - validation failed",
                        "WARN",
                    )
                    feedback = self._build_feedback(error_msg)
                    response = await self._agent.run(feedback)

                # Validate
                is_valid, error_msg, data = self._validate_output()

                if is_valid:
                    self._emit(f"Valid output on attempt {attempts}", "SUCCESS")
                    end_task_success(self.task_id or "", self.task_name or self.task_id or "")
                    return ClaudeAgentToLLMStructOutResult(
                        data=data,
                        raw_response=response,
                        output_path=self.cwd / self.output_file.lstrip("./"),
                        attempts=attempts,
                    )

                if attempt < self.max_retries:
                    self._emit(f"Validation error: {error_msg[:100]}", "WARN")

            # All retries exhausted
            error = f"Failed after {attempts} attempts. Last error: {error_msg}"
            self._emit(error, "ERROR")
            end_task_failure(self.task_id or "", self.task_name or self.task_id or "", error)
            raise ValueError(f"ClaudeAgentToLLMStructOut: {error}")

        except TimeoutError:
            end_task_timeout(
                self.task_id or "",
                self.task_name or self.task_id or "",
                self.timeout_seconds,
            )
            raise

        except Exception as e:
            if not isinstance(e, ValueError):  # Don't double-report validation failures
                end_task_error(self.task_id or "", self.task_name or self.task_id or "", str(e))
            raise

    async def close(self):
        """Cleanup resources."""
        self._agent = None

    async def __aenter__(self) -> ClaudeAgentToLLMStructOut:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close()
