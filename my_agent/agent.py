# my_agent/agent.py
from google.adk.agents import llm_agent
from google.adk.planners import PlanReActPlanner
from my_agent.tools import (
    web_search, web_research, calculator, get_time,
    code_reasoner
)

react = PlanReActPlanner()

# === 1. REASONER (Handles 90% of hidden logic) ===
reasoner = llm_agent.Agent(
    model="gemini-2.5-pro",
    name="reasoner",
    instruction="""
You are a deterministic executor,do not show your reasoning.
- Math, logic, simulation → code_reasoner
- NEVER explain. Output ONLY the final answer.
""",
    tools=[ code_reasoner, web_search],
    planner=react,
)

# === 2. FAST AGENT (Trivial / Factual) ===
fast_agent = llm_agent.Agent(
    model="gemini-2.5-flash-lite",
    name="fast",
    instruction="Answer directly, do not show your reasoning. Use web_search only if needed.",
    tools=[web_search, web_research, calculator, get_time],
    planner=react,
)

# === 3. ROOT AGENT (Universal Router) ===
root_agent = llm_agent.Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="""
    Do not show your reasoning,
You are the universal router. Classify the question:

[CLASSIFY FIRST]
- Time, date, simple math → fast_agent
- Language rules, grammar, translation → reasoner
- Grid, table, matrix, block of text → reasoner 
- Logic puzzles, simulation → reasoner → code_reasoner
- Factual questions / anything else → fast_agent → web_search

[THEN ACT]
- Use ONE tool path.
- Cite web with [1] and Sources.
- Answer questions directly and concisely in the asked format.
""",
    tools=[web_search,web_research],
    sub_agents=[fast_agent, reasoner],
    planner=react,
)