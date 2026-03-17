"""
News headline fetcher via NewsAPI.

Searches for gold / XAUUSD / Federal Reserve headlines and returns
a lightweight summary list, cached for 5 minutes.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

_NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
_SEARCH_QUERY = "gold OR XAUUSD OR \"Federal Reserve\""


def fetch_gold_headlines(
    api_key: str,
    max_results: int = 5,
) -> list[dict]:
    """Fetch recent gold-related headlines from NewsAPI.

    Parameters
    ----------
    api_key:
        NewsAPI API key.
    max_results:
        Maximum number of articles to return (default 5).

    Returns
    -------
    list[dict]
        Each dict contains: title, source, published_at, description.
    """
    cache_key = f"gold_headlines_{max_results}"

    # Serve from cache if still fresh.
    if cache_key in _cache:
        cached_ts, cached_data = _cache[cache_key]
        if (time.time() - cached_ts) < _CACHE_TTL_SECONDS:
            logger.debug("Serving gold headlines from cache")
            return cached_data

    params = {
        "q": _SEARCH_QUERY,
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "language": "en",
        "apiKey": api_key,
    }

    try:
        response = requests.get(
            _NEWSAPI_EVERYTHING_URL, params=params, timeout=10
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch headlines from NewsAPI")
        return _cache.get(cache_key, (0, []))[1]  # stale cache or empty

    data = response.json()
    articles = data.get("articles", [])

    headlines: list[dict] = []
    for article in articles[:max_results]:
        headlines.append(
            {
                "title": article.get("title", ""),
                "source": (article.get("source") or {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "description": article.get("description", ""),
            }
        )

    _cache[cache_key] = (time.time(), headlines)
    logger.info("Fetched %d gold headlines from NewsAPI", len(headlines))
    return headlines
