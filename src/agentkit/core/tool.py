from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class BaseModelProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, Any]]) -> str:
        raise NotImplementedError


class MockProvider(BaseModelProvider):
    """A simple local provider used for testing and prototyping."""

    def chat(self, messages: list[dict[str, Any]]) -> str:
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break

        if "weather" in last_user.lower():
            return '{"tool": "get_weather", "arguments": {"city": "Shanghai"}}'

        return "I can help with that."


class OpenAIProvider(BaseModelProvider):
    """Real OpenAI chat-completions provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install the openai package: pip install openai") from exc

        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
