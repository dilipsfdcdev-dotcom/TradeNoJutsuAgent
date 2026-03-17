"""
LLM-based trade confirmation gate.

Sends the proposed signal direction, recent headlines, and upcoming economic
events to a fast Claude model for a quick sanity check.  The LLM responds
with ``CONFIRM``, ``VETO``, or ``REDUCE`` plus a one-line reason.

Results are cached by a composite key of (signal direction + headline hash)
so that repeated polls within the TTL window do not incur extra API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import anthropic

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior XAUUSD trade-risk analyst.  You will be given:
1. A proposed trade direction (BUY or SELL).
2. The latest gold-related news headlines.
3. Upcoming high-impact economic events within the next few hours.

Your job is to decide whether the trade should proceed.  Respond ONLY with
a single JSON object (no markdown, no explanation outside the JSON):

{"action": "<CONFIRM|VETO|REDUCE>", "reason": "<one sentence>"}

Rules:
- CONFIRM  = the trade is consistent with the macro backdrop.
- VETO     = a clear and imminent risk contradicts the trade (e.g., FOMC in 10 min
  while buying into resistance, or a surprise geopolitical headline).
- REDUCE   = proceed but with reduced position size (e.g., conflicting signals).

Be concise.  Do NOT hallucinate events.  If uncertain, lean toward CONFIRM.\
"""

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}


def _cache_key(signal: str, headlines: list[str]) -> str:
    blob = signal + "|" + "|".join(sorted(headlines))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check(
    signal: str,
    headlines: list[str],
    upcoming_events: list[dict],
    api_key: str,
    model: str = "claude-haiku-3-5-20251001",
    timeout: int = 5,
    cache_ttl: int = 90,
) -> dict:
    """
    Ask the LLM to confirm, veto, or reduce the proposed trade.

    Parameters
    ----------
    signal : str
        ``"BUY"`` or ``"SELL"``.
    headlines : list[str]
        Latest gold/macro news headlines (up to 5 used).
    upcoming_events : list[dict]
        Economic calendar entries, each with at least ``"title"`` and
        ``"time"`` keys.  Up to 3 used.
    api_key : str
        Anthropic API key.
    model : str
        Model identifier.
    timeout : int
        Maximum seconds to wait for the API response.
    cache_ttl : int
        Seconds to keep a cached result before re-querying.

    Returns
    -------
    dict
        ``{"action": str, "reason": str}``  where action is one of
        ``CONFIRM``, ``VETO``, ``REDUCE``.
    """
    key = _cache_key(signal, headlines)

    # Check cache
    if key in _cache:
        cached_time, cached_result = _cache[key]
        if time.time() - cached_time < cache_ttl:
            logger.debug("LLM reasoner cache hit", extra={"signal": signal})
            return cached_result

    # Build user message
    top_headlines = headlines[:5]
    top_events = upcoming_events[:3]

    headline_text = "\n".join(f"- {h}" for h in top_headlines) if top_headlines else "(none)"
    event_text = "\n".join(
        f"- {e.get('title', 'Unknown')} at {e.get('time', 'N/A')}"
        for e in top_events
    ) if top_events else "(none)"

    user_message = (
        f"Proposed direction: {signal}\n\n"
        f"Recent headlines:\n{headline_text}\n\n"
        f"Upcoming events (next few hours):\n{event_text}"
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ),
            timeout=timeout,
        )

        raw_text = response.content[0].text.strip()
        result = _parse_response(raw_text)

    except asyncio.TimeoutError:
        logger.warning("LLM reasoner timed out", extra={"signal": signal})
        result = {"action": "CONFIRM", "reason": "LLM timeout – defaulting to confirm"}

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "LLM reasoner error",
            extra={"signal": signal, "error": str(exc)},
            exc_info=True,
        )
        result = {"action": "CONFIRM", "reason": f"LLM error ({type(exc).__name__}) – defaulting to confirm"}

    # Cache the result
    _cache[key] = (time.time(), result)
    logger.info("LLM reasoner result", extra={"signal": signal, **result})
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> dict:
    """
    Parse the model's JSON response, falling back gracefully.
    """
    # Strip markdown fences if present
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned unparseable response", extra={"raw": text})
        return {"action": "CONFIRM", "reason": "Unparseable LLM response – defaulting to confirm"}

    action = str(data.get("action", "CONFIRM")).upper()
    if action not in ("CONFIRM", "VETO", "REDUCE"):
        action = "CONFIRM"

    reason = str(data.get("reason", ""))
    return {"action": action, "reason": reason}
