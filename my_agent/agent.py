"""
This file is where you will implement your agent.
The `root_agent` is used to evaluate your agent's performance.
"""

from google.adk.agents import llm_agent
from my_agent.tools import web_search, web_research, image_ops, calculator, get_time
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.genai.types import ThinkingConfig, GenerateContentConfig

# ----------------------------------------------------------------------
# 1. Planner 
# ----------------------------------------------------------------------
def make_planner(include_thoughts=False):
    return BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=include_thoughts))

# ----------------------------------------------------------------------
# fast agent for trivial questions
# ----------------------------------------------------------------------
fast_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='fast_agent',
    description="Fast path for straightforward questions and quick lookups.",
    instruction=(
        "Answer directly and concisely. Prefer built-in knowledge; use tools sparingly. "
        "If you used web_search or web_research tools, add inline citations like [1] and a short 'Sources' list. "
        "Always end with a single line: 'Final answer: <short answer>'."
    ),
    tools=[web_search, web_research, image_ops, calculator, get_time],
    sub_agents=[],
)
# ----------------------------------------------------------------------
# pro agent 
# ----------------------------------------------------------------------
pro_agent = llm_agent.Agent(
    model='gemini-2.5-pro',
    name='pro_agent',
    description="High-quality path for complex, multi-step, or high-stakes questions.",
    instruction=(
        "Think carefully and verify. Plan briefly, execute tools, then synthesize. "
        "When web_search or web_research tools areused, include inline citations [n] and a 'Sources' list. "
        "Self-check before answering: correctness, units, edge cases, and whether the question asks for only a final value. "
        "Always end with a single line: 'Final answer: <short answer>'."
    ),
    tools=[web_search, web_research, image_ops, calculator, get_time],
    sub_agents=[],
)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Safe evaluation helper
# -----------------------------
def run_code(code: str) -> str:
    """Safe math evaluator."""
    try:
        return str(eval(code, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# Reasoner agent
# -----------------------------
reasoner = llm_agent.Agent(
        model="gemini-2.5-flash-lite",
        name="reasoner",
        description="An agent that plans, reasons, and uses tools to find answers.",
        instruction=(
            "You are a precision reasoner.\n"
            "- Think silently.\n"
            "- If needed, use `run_code` for math or logic.\n"
            "- NEVER explain or show steps.\n"
            "- ALWAYS output only the final, exact answer (no markdown, no punctuation)."
        ),
        tools=[web_search, web_research, image_ops, calculator, get_time],
        planner=make_planner(),
    )
# -----------------------------
# Planner agent
# -----------------------------
step_deductor = llm_agent.Agent(
        model="gemini-2.5-flash-lite",
        name="step_deductor",
        description="Breaks down a question into clear numbered steps.",
        instruction=(
            "Output only a short numbered plan (1–3 lines) with minimal text. "
            "No reasoning, no explanations, no filler words — just numbered steps in order to solve the question."
        ),
        tools=[web_search],
        planner=make_planner(),
    )

# Root agent (combined orchestrator)
root_agent = llm_agent.Agent(
    model="gemini-2.5-flash-lite",
    name="root_agent",
    description="Orchestrates reasoning by delegating to planner and reasoner.",
    instruction=(
        "You are a precise assistant. Keep answers short using proper citations and include a reference list.\n\n"
        "When needed, follow this loop:\n"
        "  - Plan: use step_deductor first.\n"
        "- Simple facts/time/math → fast_agent\n"
        "- Complex logic/math/files → pro_agent or reasoner\n"
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
    sub_agents=[fast_agent, pro_agent, reasoner, step_deductor],
)