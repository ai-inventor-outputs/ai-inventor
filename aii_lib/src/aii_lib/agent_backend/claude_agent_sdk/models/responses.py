"""Response models and result types."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptResult:
    """Per-prompt internal result.

    Cost / token / message data lands on NodeStats via streaming.py's
    ``message_callback`` — not stored here.
    """

    response: str
    session_id: str
    num_turns: int = 0
    structured_output: dict[str, Any] | None = None


@dataclass
class AgentResponse:
    """Compact result of agent execution.

    Cost / session_id / per-prompt logs are NOT stored here. Read them
    from the run tree:

        current_run().find_node(task_id).stats.total_cost
        cast(ClaudeAgentTask, current_run().find_node(task_id)).session_id
    """

    final_response: str
    structured_output: dict[str, Any] | None = None
    expected_files_valid: bool = True
    failed: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for HTTP transport (orchestrator ↔ worker)."""
        return {
            "final_response": self.final_response,
            "structured_output": self.structured_output,
            "expected_files_valid": self.expected_files_valid,
            "failed": self.failed,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponse":
        """Deserialize from HTTP transport."""
        return cls(
            final_response=data.get("final_response", ""),
            structured_output=data.get("structured_output"),
            expected_files_valid=data.get("expected_files_valid", True),
            failed=data.get("failed", False),
            error_message=data.get("error_message"),
        )
