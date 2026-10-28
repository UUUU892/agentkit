import os

from agentkit import Agent, Memory, MockProvider, OpenAIProvider, Tool


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 26°C."


provider = OpenAIProvider() if os.getenv("OPENAI_API_KEY") else MockProvider()

agent = Agent(
    name="weather-agent",
    system_prompt="You are a helpful assistant that can use tools.",
    memory=Memory(),
    tools=[
        Tool(
            name="get_weather",
            description="Get weather for a city.",
            func=get_weather,
        )
    ],
    provider=provider,
)

response = agent.run("What is the weather in Shanghai?")
print(response)
