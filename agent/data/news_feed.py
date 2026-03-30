"""News API polling for trading-relevant headlines."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import httpx
import structlog
from redis.asyncio import Redis

from agent.config import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword / symbol mapping
# ---------------------------------------------------------------------------

GLOBAL_KEYWORDS: list[str] = [
    "gold",
    "XAUUSD",
    "bitcoin",
    "BTC",
    "crypto",
    "silver",
    "XAGUSD",
    "USD",
    "Federal Reserve",
    "Fed",
    "CPI",
    "NFP",
    "FOMC",
    "inflation",
    "interest rate",
    "employment",
]

SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "XAUUSD": ["gold", "xau", "precious metals", "xauusd"],
    "BTCUSD": ["bitcoin", "btc", "crypto", "cryptocurrency", "btcusd"],
    "XAGUSD": ["silver", "xag", "xagusd"],
}

NEWS_CACHE_KEY = "news:raw"
NEWS_TTL_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


async def fetch_news() -> list[dict]:
    """Fetch latest headlines from NewsAPI.org."""

    if not settings.NEWS_API_KEY:
        log.warning("news.api_key_missing", hint="Set NEWS_API_KEY in .env")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": " OR ".join(GLOBAL_KEYWORDS),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": settings.NEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("news.fetch_http_error", status=exc.response.status_code)
        return []
    except httpx.RequestError as exc:
        log.error("news.fetch_request_error", error=str(exc))
        return []

    articles: list[dict] = data.get("articles", [])
    log.info("news.fetched", count=len(articles))

    return [
        _normalise_article(a)
        for a in articles
        if a.get("title")
    ]


def _normalise_article(raw: dict) -> dict:
    """Convert a raw NewsAPI article into our canonical format."""
    headline = raw.get("title", "")
    return {
        "headline": headline,
        "source": (raw.get("source") or {}).get("name", "unknown"),
        "timestamp": raw.get("publishedAt", datetime.now(timezone.utc).isoformat()),
        "symbols_mentioned": _detect_symbols(headline),
    }


def _detect_symbols(text: str) -> list[str]:
    """Return list of trading symbols mentioned in *text*."""
    lower = text.lower()
    return [
        symbol
        for symbol, kws in SYMBOL_KEYWORDS.items()
        if any(kw in lower for kw in kws)
    ]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def filter_relevant_news(articles: list[dict], symbol: str) -> list[dict]:
    """Keep only articles relevant to *symbol*."""

    keywords = SYMBOL_KEYWORDS.get(symbol.upper(), [])
    if not keywords:
        # Fall back: treat symbol itself as keyword
        keywords = [symbol.lower()]

    results: list[dict] = []
    for article in articles:
        lower_headline = article["headline"].lower()
        if any(kw in lower_headline for kw in keywords):
            results.append(article)
    return results


# ---------------------------------------------------------------------------
# Cache (Redis)
# ---------------------------------------------------------------------------


async def cache_news(redis_client: Redis, articles: list[dict]) -> None:
    """Store articles in Redis with a 1-hour TTL."""

    if not articles:
        return

    now = time.time()
    pipe = redis_client.pipeline()
    for article in articles:
        # Use headline + timestamp as a simple dedup key
        member = json.dumps(article, sort_keys=True)
        pipe.zadd(NEWS_CACHE_KEY, {member: now})

    # Trim entries older than TTL
    cutoff = now - NEWS_TTL_SECONDS
    pipe.zremrangebyscore(NEWS_CACHE_KEY, "-inf", cutoff)
    pipe.expire(NEWS_CACHE_KEY, NEWS_TTL_SECONDS)

    await pipe.execute()
    log.debug("news.cached", count=len(articles))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def get_recent_news(
    redis_client: Redis,
    symbol: str,
    minutes: int = 30,
) -> list[dict]:
    """Return cached news items for *symbol* published in the last *minutes*."""

    cutoff = time.time() - (minutes * 60)
    raw_members: list[bytes] = await redis_client.zrangebyscore(
        NEWS_CACHE_KEY, cutoff, "+inf",
    )

    articles = [json.loads(m) for m in raw_members]
    relevant = filter_relevant_news(articles, symbol)

    log.debug(
        "news.get_recent",
        symbol=symbol,
        minutes=minutes,
        total_cached=len(articles),
        relevant=len(relevant),
    )
    return relevant


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


async def news_polling_loop(redis_client: Redis) -> None:
    """Continuously poll for news at the configured interval."""

    interval = settings.NEWS_CHECK_INTERVAL_SEC
    log.info("news.polling_started", interval_sec=interval)

    while True:
        try:
            articles = await fetch_news()
            if articles:
                await cache_news(redis_client, articles)
        except Exception:
            log.exception("news.polling_error")

        await asyncio.sleep(interval)
