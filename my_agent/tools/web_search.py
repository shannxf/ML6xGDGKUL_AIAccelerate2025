import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_tavily_client = None  # Lazy-initialized Tavily client

# Simple on-disk cache to speed up repeated evaluations
_CACHE_DIR = Path(".cache")
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "web_search.json"

def _load_cache() -> Dict[str, dict]:
    try:
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text())
    except Exception:
        logger.warning("Failed to read web_search cache; starting fresh")
    return {}

def _save_cache(cache: Dict[str, dict]) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        logger.warning("Failed to write web_search cache")


def _get_tavily_client():
    """Return a singleton Tavily client initialized from env var.

    Requires environment variable TAVILY_API_KEY to be set (loaded from my_agent/.env by the runner).
    """
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # Degrade gracefully: allow agent to start without crashing
        logger.warning("TAVILY_API_KEY not set. web_search tool will return empty results.")
        return None

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

    # Check cache first
    cache = _load_cache()
    if query in cache:
        cached = cache[query]
        cached["_cache"] = True
        return cached

    client = _get_tavily_client()
    if client is None:
        # No API key: return empty result; agent should continue without failing
        return {"answer": None, "results": [], "citations": [], "error": "tavily_api_key_missing"}

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
        return {"answer": None, "results": [], "citations": [], "error": str(e)}

    # Normalize response into our expected schema
    raw_results = resp.get("results") or []
    # Include stable incremental ids on results to make citation linking simpler
    normalized_results = []
    for idx, item in enumerate(raw_results):
        normalized_results.append({
            "id": idx + 1,
            "title": item.get("title") or "",
            "snippet": item.get("content") or "",
            "url": item.get("url") or "",
        })

    citations = [
        {"id": r.get("id", idx + 1), "title": r.get("title", ""), "url": r.get("url", "")}
        for idx, r in enumerate(normalized_results)
    ]

    normalized = {
        "answer": resp.get("answer"),
        "results": normalized_results,
        "citations": citations,
    }

    # Persist to cache
    cache[query] = normalized
    _save_cache(cache)

    return normalized

def web_research(query: str, max_results: int = 5) -> Dict[str, Optional[str]]:
    """
    Lightweight research helper that returns a stitched context string from top results.

    Uses Tavily's get_search_context for speed and simplicity.

    Returns:
        {
          "context": str | None,
          "citations": [{id,title,url}],
          "results": [...],
          "answer": str | None
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    client = _get_tavily_client()
    if client is None:
        return {"context": None, "citations": [], "results": [], "answer": None, "error": "tavily_api_key_missing"}

    try:
        # First get regular results for citations
        search = client.search(query=query, search_depth="basic", max_results=max_results, include_answer=True)
        raw_results = search.get("results") or []
        normalized_results = []
        for idx, item in enumerate(raw_results):
            normalized_results.append({
                "id": idx + 1,
                "title": item.get("title") or "",
                "snippet": item.get("content") or "",
                "url": item.get("url") or "",
            })
        citations = [
            {"id": r.get("id", idx + 1), "title": r.get("title", ""), "url": r.get("url", "")}
            for idx, r in enumerate(normalized_results)
        ]

        # Then get stitched context (single string)
        # Tavily SDK provides get_search_context for synthesized context
        try:
            context = client.get_search_context(query=query, max_results=max_results)
        except Exception:
            # Fallback: concatenate snippets
            context = "\n\n".join([r.get("snippet", "") for r in normalized_results if r.get("snippet")])

        return {
            "context": context,
            "citations": citations,
            "results": normalized_results,
            "answer": search.get("answer"),
        }
    except Exception as e:
        logger.exception("web_research failed")
        return {"context": None, "citations": [], "results": [], "answer": None, "error": str(e)}
