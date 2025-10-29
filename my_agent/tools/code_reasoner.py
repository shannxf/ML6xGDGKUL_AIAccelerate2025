# my_agent/tools/code_reasoner.py
from typing import Dict, Any
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


def code_reasoner(
    code: str,
    state: Dict[str, Any] | None = None,
) -> dict:
    """
    Stateful Python REPL.
    Pass the same `state` dict on subsequent calls to keep variables.
    Returns:
        {"output": str, "error": str|None, "new_state": dict}
    """
    if state is None:
        state = {}
    out = StringIO()
    err = StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(code, state)
    except Exception:
        err.write(traceback.format_exc())
    return {
        "output": out.getvalue(),
        "error": err.getvalue().strip() or None,
        "new_state": state,
    }