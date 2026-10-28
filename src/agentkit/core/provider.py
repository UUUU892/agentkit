from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    messages: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_call(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        self.messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": str(args or {}),
            }
        )

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def clear(self) -> None:
        self.messages.clear()
        self.state.clear()
