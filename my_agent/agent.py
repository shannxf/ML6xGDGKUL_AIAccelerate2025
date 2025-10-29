"""
This file is where you will implement your agent.
The `root_agent` is used to evaluate your agent's performance.
"""

from google.adk.agents import llm_agent
from my_agent.tools import web_search, image_ops

fast_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='fast_agent',
    description="Fast path for straightforward questions and quick lookups.",
    instruction=(
        "Answer quickly and concisely. Use tools only if needed. If using web_search or files, include inline citations [n] and a short Sources section. "
        "After receiving tool results, always produce a final answer to the user in the same turn."
    ),
    tools=[web_search, image_ops],
    sub_agents=[],
)

pro_agent = llm_agent.Agent(
    model='gemini-2.5-pro',
    name='pro_agent',
    description="High-quality path for complex, multi-step, or high-stakes questions.",
    instruction=(
        "Do deeper reasoning, check consistency across sources, and explain briefly. Always add inline citations [n] and Sources when tools were used. "
        "Stop calling tools once you can answer; then synthesize a final answer in this turn."
    ),
    tools=[web_search, image_ops],
    sub_agents=[],
)

root_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent',
    description="A helpful assistant that can answer questions.",
    instruction=(
        "You are a helpful assistant that answers questions directly and concisely. "
        "When a question requires external information or multi-step work, follow an explicit plan and call tools in sequence. "
        "Planning pattern (agent-driven orchestration):\n"
        "  1) Plan: list the steps you will take, e.g. ['web_search', 'image_ops', etc.].\n"
        "  2) Execute tools in order, one after another, stick to your initial plan. Wait for each result before the next.\n"
        "     - Call web_search(query=...) -> returns {results:[{id,title,snippet,url}], citations:[{id,title,url}], answer?:str} \n"
        "     - Call image_ops(...) -> returns {result: ..., citations:[{id,title,url}], answer?:str} \n"
        "  3) Reason: synthesize a concise answer using tool output, add citations when relevant.\n"
        "Tool contracts: treat tool outputs as structured JSON (do not assume freeform text). If a tool fails, note the failure and continue with available info.\n"
        "Always produce a final user-facing answer after the last tool call in the turn. If uncertain, state the best-supported answer and the ambiguity.\n"
        "Model routing policy: Always use pro-agent unless the task is time-sensitive or very simply factual;"
    ),
    tools=[web_search, image_ops],
    sub_agents=[fast_agent, pro_agent],
)
