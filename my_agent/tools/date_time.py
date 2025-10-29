from datetime import datetime, timezone
from typing import Optional, Dict, Any

def get_time(fmt: str = "iso", tz: str = "utc") -> Dict[str, Any]:
    """Return current time in a simple, LLM-friendly dict.

    Args:
        fmt: 'iso'|'date'|'time'|'weekday'|'year'|'timestamp'
        tz: 'utc' or 'local'

    Returns:
        {"ok": True, "format": fmt, "tz": tz, "result": str}
    """
    dt = datetime.now(timezone.utc) if tz == "utc" else datetime.now()
    if fmt == "iso":
        res = dt.isoformat()
    elif fmt == "date":
        res = dt.strftime("%Y-%m-%d")
    elif fmt == "time":
        res = dt.strftime("%H:%M:%S")
    elif fmt == "weekday":
        res = dt.strftime("%A")
    elif fmt == "year":
        res = dt.strftime("%Y")
    elif fmt == "timestamp":
        res = str(int(dt.timestamp()))
    else:
        res = dt.isoformat()

    return {"ok": True, "format": fmt, "tz": tz, "result": res}


# Tool metadata for more reliable automatic function-calling parsing
tool = {
    "name": "get_time",
    "description": "Return the current time. fmt: iso|date|time|weekday|year|timestamp; tz: utc|local",
    "input_schema": {
        "type": "object",
        "properties": {
            "fmt": {"type": "string"},
            "tz": {"type": "string"}
        }
    },
    "fn": get_time,
}
