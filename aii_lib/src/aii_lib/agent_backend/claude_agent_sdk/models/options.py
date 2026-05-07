"""Configuration options for Claude Agent SDK."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .enums import SessionType, SystemPromptPreset

PermissionModeValue = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


@dataclass
class ExpectedFile:
    """
    A single expected output file with its description.

    Attributes:
        path: File path relative to agent cwd (e.g., "data.py", "output/result.json")
        description: What the file should contain/purpose (shown in prompt)
    """

    path: str
    description: str = ""


@dataclass
class AgentOptions:
    """Configuration options for Claude Agent SDK.

    Exposes ALL available ClaudeAgentOptions parameters.
    """

    # Core options
    allowed_tools: list[str] | None = None  # None = use SDK defaults
    system_prompt: str | SystemPromptPreset | None = "claude_code"  # String, preset dict, or None
    permission_mode: PermissionModeValue = "bypassPermissions"
    max_turns: int | None = None  # Maximum conversation turns (None = unlimited)
    # ``llm_backend`` is a symbolic name from
    # aii_config/pipeline/harness/llm_backend.yaml::*. The SDK path only
    # supports ``claude_max`` today; ``openrouter`` is rejected at config
    # load (see ``PipelineConfig._validate_backend_pairings``) because
    # the Anthropic-format → OpenAI translation drops Anthropic
    # server-side tools (WebSearch / WebFetch). For openrouter use the
    # ``_run_task_openrouter`` direct-chat path instead.
    llm_backend: str = "claude_max"
    model: str = "claude-opus-4-7"  # Claude model
    effort: Literal["low", "medium", "high", "max"] | None = "high"  # Token effort level
    cwd: str | Path = "./"  # Working directory

    # Session management
    session_type: SessionType = SessionType.NEW  # NEW, RESUME, or FORK
    resume_session_id: str | None = None  # Session ID to resume/fork from
    continue_seq_item: bool = True  # Continue conversation for 2nd+ prompts in sequence

    # Tool restrictions (rarely used)
    disallowed_tools: list[str] | None = None

    # MCP servers
    mcp_servers: dict[str, Any] | str | Path = field(default_factory=dict)

    # Custom tools (file paths to Python files with @tool decorated functions)
    custom_tool_files: list[str | Path] = field(default_factory=list)

    # Custom agents (file paths to YAML files with agent definitions)
    custom_agent_files: list[str | Path] = field(default_factory=list)

    # Agent-level execution options (entire agent run)
    agent_timeout: int | None = None  # Timeout for entire agent run in seconds (None = no timeout)
    agent_retries: int = 3  # Max retry attempts for entire agent on failure/timeout
    container_timeout: int | None = (
        None  # Container/pod lifetime limit in seconds (None = no limit)
    )

    # Per-prompt execution options (single prompt within agent)
    seq_prompt_timeout: int | None = None  # Timeout per prompt in seconds (None = no timeout)
    seq_prompt_retries: int = 3  # Max retry attempts per prompt on failure/timeout

    # Retry context
    retry_context_messages: int = 20  # Number of last messages to include in retry prompts

    # Per-message execution options (individual SDK message within streaming loop)
    # When a single message hangs past message_timeout, raises MessageTimeoutError
    # which is retried up to message_retries times (separate budget from seq_prompt_retries).
    # When message_retries exhausted, escalates as asyncio.TimeoutError to seq_prompt retry.
    message_timeout: int | None = (
        720  # Timeout per SDK message in seconds (None = no timeout, default 12 min)
    )
    message_retries: int = 5  # Max fork+resume attempts for message-level timeouts
    custom_metadata: dict[str, Any] = field(
        default_factory=dict
    )  # Custom fields to add to every message

    # SDK native structured output
    # Pass {"type": "json_schema", "schema": <json_schema_dict>} to enable
    # SDK handles validation and retry internally — no file I/O needed
    output_format: dict[str, Any] | None = None

    # Custom force output prompt (sent when max_turns is exceeded without structured output)
    # If None, uses the generic "STOP and output NOW" template
    force_output_prompt: str | None = None

    # Expected files validation via structured output (optional, off by default)
    # Set expected_files_struct_out_field to enable automatic file existence validation.
    # The agent reports created file paths in this structured output field.
    # SDK recursively extracts all string paths and validates they exist inside workspace.
    # Requires output_format to be set with a JSON schema that includes this field.
    expected_files_struct_out_field: str | None = (
        None  # Field name in structured output (e.g., "expected_files"); None = disabled
    )
    max_expected_files_retries: int = (
        2  # Max retries for missing files (only used when expected_files_struct_out_field is set)
    )

    # Post-run validation hook: fn(structured_output) -> (valid, retry_prompt | None)
    # If provided, runs after each agent.run() call. If invalid, sends retry_prompt
    # and re-runs. Retries up to post_validate_retries times. Cost accumulates.
    post_validate: Any = None  # Callable[[dict | None], tuple[bool, str | None]]
    post_validate_retries: int = 2

    # Prompt source — "pipeline" for system-generated prompts, "human" for user-injected messages.
    # Only affects telemetry tagging (frontend renders UserMessage vs SystemInstructionsMessage).
    # Not passed to Claude SDK — Claude sees identical user messages either way.
    prompt_source: Literal["pipeline", "human"] = "pipeline"

    run_id: str | None = None  # Task ID — canonical identifier for the task this agent runs under
    agent_context: str | None = None  # Display name for logs (e.g., "data-0", "exp-1", "comp-6")

    # Advanced options
    permission_prompt_tool_name: str | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # SDK telemetry — emits the Claude CLI's built-in OTel spans/metrics/logs
    # to whatever backend ``OTEL_EXPORTER_OTLP_*`` is pointed at (configured
    # process-wide by ``aii_lib.agent_backend.claude_agent_sdk.sdk_telemetry``). Default on so
    # every agent contributes traces; set ``telemetry: false`` per-step in
    # pipeline.yaml to silence a noisy or sensitive call.
    telemetry: bool = True
    extra_args: dict[str, str | None] = field(
        default_factory=lambda: {
            "strict-mcp-config": None,
            # SDK echoes UserMessage events through receive_response so we can
            # emit ``agent_user_prompt`` from honest API-confirmed delivery.
            "replay-user-messages": None,
        }
    )
    max_buffer_size: int | None = None
    debug_stderr: Any = None
    can_use_tool: Any = None  # Callable for custom tool permissions
    hooks: dict[str, list[Any]] | None = None
    user: str | None = None
    include_partial_messages: bool = True
    setting_sources: list[Literal["user", "project", "local"]] = field(
        default_factory=list
    )  # Empty by default to prevent .mcp.json auto-loading; set ["project"] explicitly when skills are needed

    # Resource selection (auto-prepared to workspace)
    selected_agents: list[Any] = field(
        default_factory=list
    )  # Agent names or AgentDefinition objects
    selected_mcps: list[Any] = field(default_factory=list)  # MCP names or McpDefinition objects

    # Internal: SDK agent definitions (populated from selected_agents during initialization)
    agents: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AgentOptions":
        """
        Load AgentOptions from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            AgentOptions instance with loaded configuration
        """
        from pathlib import Path as PathLib

        import yaml

        config_path = PathLib(config_path)

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Convert session_type string to enum
        if "session_type" in config and isinstance(config["session_type"], str):
            config["session_type"] = SessionType(config["session_type"])

        # Convert YAML config to AgentOptions fields
        return cls(**config)

    def to_serializable_dict(self) -> dict[str, Any]:
        """
        Convert AgentOptions to a JSON-serializable dict.

        Excludes non-serializable fields (telemetry, callbacks, etc.).

        Returns:
            Dict suitable for JSON serialization and AgentOptions(**dict) reconstruction
        """
        # Fields that cannot be serialized (callables, complex objects, internal state)
        non_serializable_fields = {
            "telemetry",
            "debug_stderr",
            "can_use_tool",
            "hooks",
            "agents",  # Internal: populated during initialization
        }

        result = {}
        for field_name in self.__dataclass_fields__:
            if field_name in non_serializable_fields:
                continue

            value = getattr(self, field_name)

            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)
            # Convert lists of Paths
            elif isinstance(value, list):
                value = [str(v) if isinstance(v, Path) else v for v in value]
            # Convert Enum to string
            elif hasattr(value, "value") and hasattr(value, "name"):
                value = value.value

            result[field_name] = value

        return result
