from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]


class SimpleLLM:
    """A tiny mock model for local prototyping."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        # In a real agent, this would call OpenAI/Claude/Gemini.
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        if "weather" in last_user.lower():
            return "TOOL:get_weather|city=Shanghai"
        return "I can help answer your question."


@dataclass
class Memory:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


class Agent:
    def __init__(self, name: str, system_prompt: str, tools: list[Tool] | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.memory = Memory()
        self.model = SimpleLLM()

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        prompt = [{"role": "system", "content": self.system_prompt}] + self.memory.messages
        response = self.model.chat(prompt)

        if response.startswith("TOOL:"):
            tool_call = response.split(":", 1)[1]
            tool_name, payload = tool_call.split("|", 1)
            tool = self.tools.get(tool_name)
            if tool is None:
                answer = f"Tool '{tool_name}' not found."
                self.memory.add_assistant(answer)
                return answer

            args: dict[str, str] = {}
            for part in payload.split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    args[key.strip()] = value.strip()

            result = tool.func(**args)
            output = f"Used tool '{tool_name}' -> {result}"
            self.memory.add_assistant(output)
            return output

        self.memory.add_assistant(response)
        return response


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 26°C."


if __name__ == "__main__":
    agent = Agent(
        name="demo-agent",
        system_prompt="You are a helpful assistant. Use tools when needed.",
        tools=[
            Tool(name="get_weather", description="Get weather for a city", func=get_weather),
        ],
    )

    result = agent.run("What is the weather in Shanghai?")
    print(result)
