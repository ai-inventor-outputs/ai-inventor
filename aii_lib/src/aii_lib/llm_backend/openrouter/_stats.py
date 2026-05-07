"""Conversation stats for multi-turn OpenRouter interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ConversationStats:
    """Aggregated stats for multi-turn conversations."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_cost: float = 0.0
    num_turns: int = 0
    tool_calls: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    last_response: object = None
    model: str = ""
    finish_reason: str = "unknown"

    def add_turn(self, usage: dict, cost: float, tool_calls: list[dict] | None = None):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.reasoning_tokens += usage.get("reasoning_tokens", 0)
        self.cached_tokens += usage.get("cached_tokens", 0)
        self.total_cost += cost
        self.num_turns += 1
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def get_runtime_minutes(self) -> float:
        return (datetime.now(UTC) - self.start_time).total_seconds() / 60.0
