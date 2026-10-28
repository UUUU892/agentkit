from __future__ import annotations

import json
from typing import Any

from .memory import Memory
from .provider import BaseModelProvider, MockProvider
from .tool import Tool


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        provider: BaseModelProvider | None = None,
        max_steps: int = 5,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.memory = memory or Memory()
        self.provider = provider or MockProvider()
        self.max_steps = max_steps

    def _build_messages(self) -> list[dict[str, Any]]:
        system_message = {
            "role": "system",
            "content": (
                self.system_prompt
                + "\n\nIf you need to call a tool, respond with a JSON object only in this form: "
                + '{"tool": "tool_name", "arguments": {"arg": "value"}}. '
                + "If no tool is needed, return a normal answer."
            ),
        }
        return [system_message, *self.memory.messages]

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        for _ in range(self.max_steps):
            messages = self._build_messages()
            raw_response = self.provider.chat(messages)
            text = str(raw_response).strip()

            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                self.memory.add_assistant(text)
                return text

            tool_name = payload.get("tool")
            if isinstance(tool_name, str) and tool_name in self.tools:
                arguments = payload.get("arguments", {})
                result = self.tools[tool_name].call(**arguments)
                final_text = f"Tool '{tool_name}' returned: {result}"
                self.memory.add_assistant(final_text)
                return final_text

            if isinstance(payload, dict):
                reply = payload.get("answer") or payload.get("response") or text
            else:
                reply = text
            self.memory.add_assistant(reply)
            return str(reply)

        error_text = "The agent reached its maximum number of steps without finishing the task."
        self.memory.add_assistant(error_text)
        return error_text

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
