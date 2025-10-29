# my_agent/tools/string_reasoner.py
from typing import Dict, Any

def string_reasoner(
    rules: str,
    sentence: str,
    state: Dict[str, Any] | None = None,
) -> dict:
    """
    Apply linguistic rules to transform a sentence.
    Rules format: "V O S" or "accusative_subject" etc.
    """
    if state is None:
        state = {}
    
    # Simple rule engine (expand as needed)
    lines = [line.strip() for line in rules.split('\n') if line.strip()]
    result = sentence
    
    for rule in lines:
        if rule == "V O S":
            words = result.split()
            if len(words) >= 3:
                result = f"{words[0]} {words[2]} {words[1]}"
        elif rule == "accusative_subject":
            # "I" → "mato"
            result = result.replace("I ", "mato ")
        elif rule == "accusative_object":
            result = result.replace("apples", "zapple")
        # Add more rules as needed
    
    return {"output": result.strip(), "new_state": state}