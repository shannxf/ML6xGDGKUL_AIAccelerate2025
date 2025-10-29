"""
This file is where you will implement your agent.
The `root_agent` is used to evaluate your agent's performance.
"""

from google.adk.agents import llm_agent
from my_agent.tools import web_search, web_research, image_ops, calculator, get_time

fast_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='fast_agent',
    description="Fast path for straightforward questions and quick lookups.",
    instruction=(
        "Answer directly and concisely. Prefer built-in knowledge; use tools sparingly. "
        "If you used any external tools, add inline citations like [1] and a short 'Sources' list. "
        "Always end with a single line: 'Final answer: <short answer>'."
    ),
    tools=[web_search, web_research, image_ops, calculator, get_time],
    sub_agents=[],
)

pro_agent = llm_agent.Agent(
    model='gemini-2.5-pro',
    name='pro_agent',
    description="High-quality path for complex, multi-step, or high-stakes questions.",
    instruction=(
        "Think carefully and verify. Plan briefly, execute tools, then synthesize. "
        "When tools were used, include inline citations [n] and a 'Sources' list. "
        "Self-check before answering: correctness, units, edge cases, and whether the question asks for only a final value. "
        "Always end with a single line: 'Final answer: <short answer>'."
    ),
    tools=[web_search, web_research, image_ops, calculator, get_time],
    sub_agents=[],
)

root_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent',
    description="A helpful assistant that can answer questions.",
    instruction=(
        "You are a precise assistant. Keep answers short unless the user asks for detail.\n\n"
        "When needed, follow this loop:\n"
        "  - Plan: list 1-3 steps.\n"
    "  - Act: call tools in sequence (web_search/web_research, image_ops, calculator, get_time).\n"
        "  - Synthesize: combine results, add [n] citations, and perform a quick self-check (units, edge cases).\n\n"
        "Tool contracts:\n"
        "  web_search(query) -> {answer?, results:[{id,title,snippet,url}], citations}\n"
        "  web_research(query,max_results?) -> {context, citations, results, answer?}\n"
    "  image_ops(...), calculator(...), get_time(...)\n\n"
        "Routing: default to pro_agent for complex tasks; use fast_agent for simple factual questions.\n\n"
        "Output rule: Always end with a single line 'Final answer: <short answer>'."
    ),
    tools=[web_search, web_research, image_ops, calculator, get_time],
    sub_agents=[fast_agent, pro_agent],
)
