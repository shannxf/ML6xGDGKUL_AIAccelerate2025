import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

_tavily_client = None  # Lazy-initialized Tavily client


def _get_tavily_client():
    """Return a singleton Tavily client initialized from env var.

    Requires environment variable TAVILY_API_KEY to be set (loaded from my_agent/.env by the runner).
    """
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # Fail fast with a helpful message
        raise RuntimeError(
            "TAVILY_API_KEY not set. Please add it to my_agent/.env and re-run."
        )

    try:
        from tavily import TavilyClient  # Local import to avoid hard dependency at import time
    except Exception as e:
        raise RuntimeError(
            "tavily-python package is not installed. Add 'tavily-python' to dependencies and install."
        ) from e

    _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def web_search(query: str) -> Dict[str, List[dict]]:
    """Performs a web search using Tavily and returns results with citation info.

    Args:
        query (str): The search query.

    Returns:
        dict: {
            "answer": str | None,  # Tavily's synthesized direct answer when available
            "results": [{"title": str, "snippet": str, "url": str}, ...],
            "citations": [{"id": int, "title": str, "url": str}, ...]
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    logger.info(f"Performing Tavily web search for query: {query}")

    client = _get_tavily_client()

    # Keep it basic and fast for now; return a small set of results
    try:
        resp = client.search(
            query=query,
            search_depth="basic",  # faster, good enough for a simple tool
            max_results=5,
            include_answer=True,
        )
    except Exception as e:
        # Surface a concise error while logging details
        logger.exception("Tavily search failed")
        raise RuntimeError(f"Tavily search failed: {e}") from e

    # Normalize response into our expected schema
    raw_results = resp.get("results") or []
    normalized_results = [
        {
            "title": item.get("title") or "",
            "snippet": item.get("content") or "",
            "url": item.get("url") or "",
        }
        for item in raw_results
    ]

    citations = [
        {"id": idx + 1, "title": r.get("title", ""), "url": r.get("url", "")}
        for idx, r in enumerate(normalized_results)
    ]

    normalized = {
        "answer": resp.get("answer"),
        "results": normalized_results,
        "citations": citations,
    }

    return normalized
