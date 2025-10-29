import math
from typing import Optional, Dict, Any

def calculator(operation: str, a: float, b: Optional[float] = None) -> Dict[str, Any]:
    """Perform basic arithmetic and return a structured result.

    Args:
        operation: 'add','subtract','multiply','divide','power','sqrt','percent'
        a: first operand
        b: second operand where applicable

    Returns:
        {"ok": True/False, "result": float|None, "error": str|None}
    """
    op = operation.lower()
    try:
        if op == "add":
            if b is None:
                raise ValueError("b is required for add")
            return {"ok": True, "result": a + b, "error": None}
        if op == "subtract":
            if b is None:
                raise ValueError("b is required for subtract")
            return {"ok": True, "result": a - b, "error": None}
        if op == "multiply":
            if b is None:
                raise ValueError("b is required for multiply")
            return {"ok": True, "result": a * b, "error": None}
        if op == "divide":
            if b is None:
                raise ValueError("b is required for divide")
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return {"ok": True, "result": a / b, "error": None}
        if op == "power":
            if b is None:
                raise ValueError("b is required for power")
            return {"ok": True, "result": a ** b, "error": None}
        if op == "sqrt":
            if a < 0:
                raise ValueError("sqrt of negative number")
            return {"ok": True, "result": math.sqrt(a), "error": None}
        if op == "percent":
            if b is None:
                raise ValueError("b is required for percent")
            return {"ok": True, "result": (a * b) / 100.0, "error": None}
        raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        return {"ok": False, "result": None, "error": str(e)}


# Tool metadata to help automatic function calling
tool = {
    "name": "calculator",
    "description": "Simple calculator: operation (add,subtract,multiply,divide,power,sqrt,percent), a (float), b (float optional)",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation","a"]
    },
    "fn": calculator,
}
