"""
This file is where you will implement your agent.
The `root_agent` is used to evaluate your agent's performance.
"""

from google.adk.agents import llm_agent
from my_agent.tools import web_search

root_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent',
    description="A helpful assistant that can answer questions.",
    instruction=(
        "You are a helpful assistant that answers questions directly and concisely. "
        "When you use the web_search tool, always include citations in your answer: "
        "use numbered markers like [1], [2] inline next to claims and add a short 'Sources' section at the end "
        "listing each number with its URL and title. Prefer authoritative sources, and avoid over-citing."
    ),
    tools=[web_search],
    sub_agents=[],
)
