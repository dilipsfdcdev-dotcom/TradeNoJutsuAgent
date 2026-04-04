"""Uses Claude API to score news sentiment for each trading symbol."""

from dataclasses import dataclass
import json
import structlog
import anthropic
import redis.asyncio as redis

from agent.config import settings

logger = structlog.get_logger()


@dataclass
class SentimentResult:
    score: float        # -1.0 to 1.0
    reasoning: str
    impact: str         # "high", "medium", "low"


async def get_sentiment(redis_client: redis.Redis, symbol: str, headlines: list[dict]) -> SentimentResult:
    """
    Score news sentiment for a symbol using Claude API.

    Process:
    1. Check Redis cache first (TTL 5 min)
    2. If no cache, collect recent headlines
    3. Build prompt asking Claude to score sentiment
    4. Parse JSON response
    5. Cache result in Redis
    """
    # Check cache
    cache_key = f"sentiment:{symbol}"
    cached = await redis_client.get(cache_key)
    if cached:
        data = json.loads(cached)
        return SentimentResult(**data)

    # If no headlines, return neutral
    if not headlines:
        result = SentimentResult(score=0.0, reasoning="No recent news", impact="low")
        await redis_client.setex(cache_key, 300, json.dumps({"score": result.score, "reasoning": result.reasoning, "impact": result.impact}))
        return result

    # Build headlines text
    headlines_text = "\n".join(
        f"- [{h.get('source', 'Unknown')}] {h['headline']} ({h.get('timestamp', 'N/A')})"
        for h in headlines[:20]  # Limit to 20 most recent
    )

    prompt = f"""Score the sentiment for {symbol} based on these recent headlines.

Headlines:
{headlines_text}

Analyze how these headlines would affect {symbol} price in the next 1-4 hours.
Consider:
- Direct mentions of the asset
- USD strength/weakness (inverse for gold/silver)
- Risk-on vs risk-off sentiment
- Central bank policy implications
- Geopolitical factors

Return ONLY valid JSON:
{{"score": <float from -1.0 (very bearish) to 1.0 (very bullish)>, "reasoning": "<2-3 sentences>", "impact": "<high|medium|low>"}}"""

    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()
        # Extract JSON from response (handle potential markdown wrapping)
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        data = json.loads(response_text)
        result = SentimentResult(
            score=max(-1.0, min(1.0, float(data["score"]))),
            reasoning=str(data["reasoning"]),
            impact=str(data.get("impact", "medium")).lower()
        )

        # Cache for 5 minutes
        await redis_client.setex(
            cache_key, 300,
            json.dumps({"score": result.score, "reasoning": result.reasoning, "impact": result.impact})
        )

        logger.info("sentiment_scored", symbol=symbol, score=result.score, impact=result.impact)
        return result

    except json.JSONDecodeError as e:
        logger.error("sentiment_parse_error", symbol=symbol, error=str(e))
        return SentimentResult(score=0.0, reasoning="Failed to parse sentiment response", impact="low")
    except Exception as e:
        logger.error("sentiment_error", symbol=symbol, error=str(e))
        return SentimentResult(score=0.0, reasoning=f"Sentiment analysis failed: {str(e)}", impact="low")
