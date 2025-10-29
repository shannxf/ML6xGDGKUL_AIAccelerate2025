"""
This file is where you will implement your agent.
The `root_agent` is used to evaluate your agent's performance.
"""

from google.adk.agents import llm_agent
from my_agent.tools import web_search
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.genai.types import ThinkingConfig, GenerateContentConfig

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

# ----------------------------------------------------------------------
# 1. Planner 
# ----------------------------------------------------------------------
planner = BuiltInPlanner(
        thinking_config=ThinkingConfig(include_thoughts=False)
    )
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
        tools=[run_code],
        planner=planner,
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
        tools=[],
        planner=planner,
    )


# # Root agent 3 (uses step_deductor)
# root_agent3 = llm_agent.Agent(
#     model='gemini-2.5-flash-lite',
#     name='agent_r3',
#     description="A helpful assistant that can answer questions.",
#     instruction=(
#         "You are the orchestrator.\n"
#         "- If the task needs step-by-step reasoning, delegate to 'step_deductor'. "
#         "If any external facts are needed, please research beforehand.\n"
#         "- When using web_search, include [1], [2] markers and a short 'Sources:' list.\n"
#     ),
#     tools=[web_search],
#     sub_agents=[step_deductor],
# )

# # Root agent 2 (uses reasoner)
# root_agent2 = llm_agent.Agent(
#     model='gemini-2.5-flash-lite',
#     name='agent_r2',
#     description="A helpful assistant that can answer questions.",
#     instruction=(
#         "You are the orchestrator.\n"
#         "- If the task needs step-by-step reasoning or external facts, delegate to 'reasoner'. "
#         "If the 'reasoner' needs any up-to-date information, research it first.\n"
#         "- If the task asks for an EXACT string/word/translation, answer directly with ONLY that string.\n"
#         "- When using web_search, include [1], [2] markers and a short 'Sources:' list.\n"
#     ),
#     tools=[web_search, run_code],
#     sub_agents=[reasoner],
# )

# Root agent (combined orchestrator)
root_agent = llm_agent.Agent(
    model="gemini-2.5-flash-lite",
    name="root_agent",
    description="Orchestrates reasoning by delegating to planner and reasoner.",
    instruction=(
        "You are an orchestrator.\n"
        "- If complex, call 'step_deductor' first.\n"
        "- Then call 'reasoner' to compute the final answer.\n"
        "- For simple or factual tasks, answer directly.\n"
        "- When using web_search, cite sources briefly with [1], [2] and a 'Sources:' line.\n"
        "- Final output: ONLY the exact answer, with no extra words, markdown, or commentary."),
    tools=[web_search],
    sub_agents=[reasoner, step_deductor],
    planner=planner,
)

""" issues with current prompts:
| Issue Type                     | Example                 | Fix                                  |
| ------------------------------ | ----------------------- | ------------------------------------ |
| Linguistic structure confusion | “Maktay Zapple Mato”    | Enforce literal structural adherence |
| Casing / exact text mismatch   | “Castle” ≠ “THE CASTLE” | Output verbatim source text          |
| Sentence reconstruction        | “THESEAGULL…”           | Normalize to readable English        |
| Empty response  ?               | Missing “23”            | Require non-empty output             |
| Format strictness              | Various                 | Post-process normalization layer     |

for the def run code... gemini tried to do run code but i didnt have that tool, idk if this is actually good to do..
"""

