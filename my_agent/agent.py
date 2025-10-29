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


# -----------------------------
# Reasoner agent
# -----------------------------
def make_reasoner():
    try:
        planner = BuiltInPlanner(
            thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=1024)
        )
        planner_type = "BuiltInPlanner"
    except Exception:
        planner = PlanReActPlanner()
        planner_type = "PlanReActPlanner"

    logger.info(f"[Reasoner] Using planner type: {planner_type}")

    return llm_agent.Agent(
        model="gemini-2.5-flash-lite",
        name="reasoner",
        description="An agent that plans, reasons, and uses tools to find answers.",
        instruction=(
            "Output discipline:\n"
            "- If the question asks for EXACT wording or a SINGLE sentence/word, output ONLY that with no extra text.\n"
            "- If the question asks to extract or translate, do not add explanations.\n"
            "You are a reasoning agent. Follow this protocol INTERNALLY:\n"
            "1) Plan minimally.\n2) Solve.\n3) Verify.\n4) Output only final answer.\n"
            "- For logic puzzles, explicitly test edge cases and contradictions before concluding.\n"
        ),
        tools=[run_code],
        planner=planner,
    )


# -----------------------------
# Planner agent
# -----------------------------
def make_step_deductor():
    try:
        planner = BuiltInPlanner(
            thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=512)
        )
        planner_type = "BuiltInPlanner"
    except Exception:
        planner = PlanReActPlanner()
        planner_type = "PlanReActPlanner"

    logger.info(f"[Planner] Using planner type: {planner_type}")

    return llm_agent.Agent(
        model="gemini-2.5-flash-lite",
        name="step_deductor",
        description="Breaks down a question into clear numbered steps.",
        instruction=(
            "Analyze the question and output a short numbered plan (1–3 sentences per step) "
            "describing exactly what needs to be done to solve it. "
            "Do NOT solve it — just describe the required steps clearly and simply. "
            "Keep it short and clear so that another agent can execute it directly."
        ),
        tools=[],
        planner=planner,
    )


# -----------------------------
# Root agents
# -----------------------------
# Create *independent* sub-agent instances for each root agent
step_deductor_agent_r3 = make_step_deductor()
reasoner_agent_r2 = make_reasoner()
step_deductor_agent_r = make_step_deductor()
reasoner_agent_r = make_reasoner()

# Shared planner setup for root_agent
try:
    planner = BuiltInPlanner(
        thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=1024)
    )
except Exception:
    planner = PlanReActPlanner()

# Root agent 3 (uses step_deductor)
root_agent3 = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent_r3',
    description="A helpful assistant that can answer questions.",
    instruction=(
        "You are the orchestrator.\n"
        "- If the task needs step-by-step reasoning, delegate to 'step_deductor'. "
        "If any external facts are needed, please research beforehand.\n"
        "- When using web_search, include [1], [2] markers and a short 'Sources:' list.\n"
    ),
    tools=[web_search],
    sub_agents=[step_deductor_agent_r3],
)

# Root agent 2 (uses reasoner)
root_agent2 = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent_r2',
    description="A helpful assistant that can answer questions.",
    instruction=(
        "You are the orchestrator.\n"
        "- If the task needs step-by-step reasoning or external facts, delegate to 'reasoner'. "
        "If the 'reasoner' needs any up-to-date information, research it first.\n"
        "- If the task asks for an EXACT string/word/translation, answer directly with ONLY that string.\n"
        "- When using web_search, include [1], [2] markers and a short 'Sources:' list.\n"
    ),
    tools=[web_search, run_code],
    sub_agents=[reasoner_agent_r2],
)

# Root agent (combined orchestrator)
root_agent = llm_agent.Agent(
    model="gemini-2.5-flash-lite",
    name="root_agent",
    description="Orchestrates reasoning by delegating to planner and reasoner.",
    instruction=(
        "You are the main orchestrator.\n"
        "- If the task is complex or requires reasoning, first delegate to 'step_deductor' to "
        "generate clear steps.\n"
        "- Then pass those steps to 'reasoner' to compute the final answer.\n"
        "- If the task is trivial or purely factual, answer directly.\n"
        "- For external facts, use 'web_search' beforehand and cite with [1], [2] markers and 'Sources:' list.\n"
        "- Always keep the final answer concise and short.\n"
    ),
    tools=[web_search],
    sub_agents=[step_deductor_agent_r, reasoner_agent_r],
    planner=planner,
)
