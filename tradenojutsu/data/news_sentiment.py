"""News sentiment analysis using LLM-powered headline scoring.

Fetches financial news headlines and uses Claude to analyze sentiment,
providing a market sentiment overlay for trading decisions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from tradenojutsu.infra.logger import get_logger

logger = get_logger("data.news")


@dataclass
class NewsItem:
    """A single news article/headline."""
    headline: str
    source: str
    url: str = ""
    published: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    relevance: float = 0.0  # 0.0 to 1.0
    summary: str = ""


@dataclass
class SentimentReport:
    """Aggregated sentiment analysis for a symbol."""
    symbol: str
    overall_sentiment: float  # -1.0 to +1.0
    sentiment_label: str  # "very_bearish", "bearish", "neutral", "bullish", "very_bullish"
    confidence: float  # 0.0 to 1.0
    news_count: int
    top_headlines: list[NewsItem] = field(default_factory=list)
    analysis: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt_context(self) -> str:
        """Format for LLM consumption."""
        lines = [
            f"News Sentiment for {self.symbol}:",
            f"  Overall: {self.sentiment_label} ({self.overall_sentiment:+.2f})",
            f"  Confidence: {self.confidence:.0%}",
            f"  Headlines analyzed: {self.news_count}",
        ]
        if self.top_headlines:
            lines.append("  Key Headlines:")
            for item in self.top_headlines[:5]:
                lines.append(f"    [{item.sentiment:+.2f}] {item.headline}")
        if self.analysis:
            lines.append(f"  Analysis: {self.analysis}")
        return "\n".join(lines)


# Symbol to search query mapping
SYMBOL_QUERIES = {
    # Metals
    "XAUUSD": "gold price XAUUSD forecast",
    "GC=F": "gold futures price forecast",
    "XAGUSD": "silver price XAGUSD forecast",
    "SI=F": "silver futures price forecast",
    # Crypto
    "BTC-USD": "Bitcoin BTC price crypto",
    "BTC/USDT": "Bitcoin BTC price crypto",
    "BTCUSD": "Bitcoin BTC price crypto",
    "ETH-USD": "Ethereum ETH price crypto",
    "ETH/USDT": "Ethereum ETH price crypto",
    # Forex
    "EURUSD=X": "EUR USD forex euro dollar",
    "EURUSD": "EUR USD forex euro dollar",
    "GBPUSD=X": "GBP USD forex pound dollar",
    "GBPUSD": "GBP USD forex pound dollar",
    "USDJPY=X": "USD JPY forex yen dollar",
    "USDJPY": "USD JPY forex yen dollar",
    "AUDUSD=X": "AUD USD forex australian dollar",
    "USDCHF=X": "USD CHF forex swiss franc",
    "USDCAD=X": "USD CAD forex canadian dollar",
    "NZDUSD=X": "NZD USD forex new zealand dollar",
}


class NewsSentimentAnalyzer:
    """Fetches news and analyzes sentiment using LLM.

    Uses free RSS/API sources for headlines, then Claude for sentiment scoring.
    Includes caching to minimize API calls.
    """

    def __init__(self, llm_model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.llm_model = llm_model
        self.api_key = api_key
        self._client = None
        self._cache: dict[str, tuple[SentimentReport, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def fetch_headlines(self, symbol: str, max_items: int = 15) -> list[NewsItem]:
        """Fetch news headlines for a symbol from free sources."""
        query = SYMBOL_QUERIES.get(symbol, f"{symbol} stock price market")
        headlines = []

        # Try Google News RSS
        try:
            rss_items = self._fetch_google_news_rss(query, max_items)
            headlines.extend(rss_items)
        except Exception as e:
            logger.warning(f"Google News RSS failed: {e}")

        # Try DuckDuckGo news (fallback)
        if not headlines:
            try:
                ddg_items = self._fetch_ddg_news(query, max_items)
                headlines.extend(ddg_items)
            except Exception as e:
                logger.warning(f"DuckDuckGo news failed: {e}")

        logger.info(f"Fetched {len(headlines)} headlines for {symbol}")
        return headlines[:max_items]

    def analyze_sentiment(self, symbol: str, headlines: list[NewsItem] | None = None) -> SentimentReport:
        """Analyze news sentiment for a symbol.

        Fetches headlines if not provided, then uses LLM for sentiment analysis.
        Results are cached for 5 minutes.
        """
        # Check cache
        cache_key = symbol
        if cache_key in self._cache:
            report, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self._cache_ttl:
                return report

        # Fetch headlines if needed
        if headlines is None:
            headlines = self.fetch_headlines(symbol)

        if not headlines:
            return SentimentReport(
                symbol=symbol,
                overall_sentiment=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                news_count=0,
                analysis="No news available for analysis.",
            )

        # LLM sentiment analysis
        report = self._llm_analyze(symbol, headlines)

        # Cache result
        self._cache[cache_key] = (report, time.time())
        return report

    def _llm_analyze(self, symbol: str, headlines: list[NewsItem]) -> SentimentReport:
        """Use LLM to analyze headline sentiment."""
        headlines_text = "\n".join(f"- {h.headline} (source: {h.source})" for h in headlines)

        prompt = f"""Analyze the sentiment of these news headlines for {symbol}.

Headlines:
{headlines_text}

Respond in JSON:
{{
    "overall_sentiment": <float -1.0 to 1.0>,
    "sentiment_label": "very_bearish" | "bearish" | "neutral" | "bullish" | "very_bullish",
    "confidence": <float 0.0 to 1.0>,
    "analysis": "<brief 1-2 sentence market sentiment summary>",
    "per_headline": [
        {{"index": 0, "sentiment": <float>, "relevance": <float 0-1>}},
        ...
    ]
}}"""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.llm_model,
                max_tokens=800,
                system="You are a financial news sentiment analyzer. Be objective and precise.",
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_json(response.content[0].text)

            # Update individual headline sentiments
            for item in result.get("per_headline", []):
                idx = item.get("index", -1)
                if 0 <= idx < len(headlines):
                    headlines[idx].sentiment = item.get("sentiment", 0)
                    headlines[idx].relevance = item.get("relevance", 0)

            # Sort by relevance for top headlines
            scored = sorted(headlines, key=lambda h: abs(h.sentiment) * h.relevance, reverse=True)

            return SentimentReport(
                symbol=symbol,
                overall_sentiment=result.get("overall_sentiment", 0),
                sentiment_label=result.get("sentiment_label", "neutral"),
                confidence=result.get("confidence", 0.5),
                news_count=len(headlines),
                top_headlines=scored[:5],
                analysis=result.get("analysis", ""),
            )

        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return SentimentReport(
                symbol=symbol,
                overall_sentiment=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                news_count=len(headlines),
                analysis=f"Analysis failed: {e}",
            )

    def _fetch_google_news_rss(self, query: str, max_items: int) -> list[NewsItem]:
        """Fetch from Google News RSS feed."""
        import xml.etree.ElementTree as ET

        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "TradeNoJutsu/1.0"})
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "")
            source = item.findtext("source", "Unknown")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")

            items.append(NewsItem(
                headline=title,
                source=source,
                url=link,
            ))

        return items

    def _fetch_ddg_news(self, query: str, max_items: int) -> list[NewsItem]:
        """Fetch from DuckDuckGo instant answer (fallback)."""
        url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&t=tradenojutsu"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = []
        for topic in data.get("RelatedTopics", [])[:max_items]:
            text = topic.get("Text", "")
            if text:
                items.append(NewsItem(headline=text, source="DuckDuckGo", url=topic.get("FirstURL", "")))

        return items

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
