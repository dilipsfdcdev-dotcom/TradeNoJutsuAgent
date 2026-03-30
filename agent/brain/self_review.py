"""Post-trade learning loop -- uses Claude to review trades and generate weekly summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import anthropic
import pandas as pd
import structlog

from agent.config import settings

logger = structlog.get_logger(__name__)

_REVIEW_MODEL = "claude-sonnet-4-20250514"
_REVIEW_MAX_TOKENS = 1024
_WEEKLY_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def review_trade(
    trade_data: dict,
    candles_around_trade: pd.DataFrame,
) -> dict:
    """Ask Claude to evaluate a closed trade and extract lessons.

    Parameters
    ----------
    trade_data:
        Must contain at minimum: ``symbol``, ``direction``, ``entry_price``,
        ``exit_price``, ``entry_time``, ``exit_time``, ``pnl``, ``rr_achieved``,
        ``original_reasoning``, ``stop_loss``, ``take_profit``, ``confidence``.
        May also include ``indicators_at_entry`` (dict snapshot) and
        ``risk_pct``.
    candles_around_trade:
        OHLCV DataFrame covering the period from shortly before entry to
        shortly after exit.  Used to summarise post-entry price action.

    Returns
    -------
    dict
        Keys: ``entry_quality``, ``sl_quality``, ``should_have_traded``,
        ``missed_signals``, ``lesson``, ``suggested_adjustment``.
        Returns a dict with ``"error"`` key on failure.
    """
    prompt = _build_review_prompt(trade_data, candles_around_trade)

    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=_REVIEW_MODEL,
            max_tokens=_REVIEW_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        logger.info(
            "trade_review_received",
            symbol=trade_data.get("symbol"),
            response_length=len(raw),
        )

        return _parse_json_response(raw, fallback_keys=[
            "entry_quality",
            "sl_quality",
            "should_have_traded",
            "missed_signals",
            "lesson",
            "suggested_adjustment",
        ])

    except anthropic.APIConnectionError as exc:
        logger.error("review_api_connection_error", error=str(exc))
        return {"error": f"API connection error: {exc}"}
    except anthropic.RateLimitError as exc:
        logger.error("review_rate_limit", error=str(exc))
        return {"error": f"Rate limit: {exc}"}
    except anthropic.APIStatusError as exc:
        logger.error("review_api_error", status=exc.status_code, error=str(exc))
        return {"error": f"API error ({exc.status_code}): {exc}"}
    except Exception as exc:
        logger.error("review_unexpected_error", error=str(exc))
        return {"error": f"Unexpected error: {exc}"}


async def generate_weekly_summary(
    trades: list[dict],
    reviews: list[dict],
) -> dict:
    """Aggregate trade reviews for the week and ask Claude for a strategy assessment.

    Parameters
    ----------
    trades:
        List of trade dicts (same schema as ``review_trade``'s ``trade_data``).
    reviews:
        Corresponding list of review dicts returned by ``review_trade``.

    Returns
    -------
    dict
        Keys: ``whats_working``, ``whats_failing``, ``suggested_focus``,
        ``overall_grade``, ``key_stats``.
        Returns a dict with ``"error"`` key on failure.
    """
    prompt = _build_weekly_prompt(trades, reviews)

    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=_REVIEW_MODEL,
            max_tokens=_WEEKLY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        logger.info("weekly_summary_received", response_length=len(raw))

        return _parse_json_response(raw, fallback_keys=[
            "whats_working",
            "whats_failing",
            "suggested_focus",
            "overall_grade",
            "key_stats",
        ])

    except anthropic.APIConnectionError as exc:
        logger.error("weekly_api_connection_error", error=str(exc))
        return {"error": f"API connection error: {exc}"}
    except anthropic.RateLimitError as exc:
        logger.error("weekly_rate_limit", error=str(exc))
        return {"error": f"Rate limit: {exc}"}
    except anthropic.APIStatusError as exc:
        logger.error("weekly_api_error", status=exc.status_code, error=str(exc))
        return {"error": f"API error ({exc.status_code}): {exc}"}
    except Exception as exc:
        logger.error("weekly_unexpected_error", error=str(exc))
        return {"error": f"Unexpected error: {exc}"}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_review_prompt(trade_data: dict, candles_around_trade: pd.DataFrame) -> str:
    """Assemble the single-trade review prompt for Claude."""

    symbol = trade_data.get("symbol", "UNKNOWN")
    direction = trade_data.get("direction", "?")
    entry_price = trade_data.get("entry_price", 0.0)
    exit_price = trade_data.get("exit_price", 0.0)
    entry_time = trade_data.get("entry_time", "N/A")
    exit_time = trade_data.get("exit_time", "N/A")
    pnl = trade_data.get("pnl", 0.0)
    rr_achieved = trade_data.get("rr_achieved", 0.0)
    stop_loss = trade_data.get("stop_loss", 0.0)
    take_profit = trade_data.get("take_profit", 0.0)
    confidence = trade_data.get("confidence", 0)
    risk_pct = trade_data.get("risk_pct", 0.0)
    reasoning = trade_data.get("original_reasoning", "No reasoning recorded.")

    # Indicators snapshot at entry
    indicators = trade_data.get("indicators_at_entry", {})
    indicators_text = "N/A"
    if indicators:
        lines = [f"  {k}: {v}" for k, v in indicators.items()]
        indicators_text = "\n".join(lines)

    # Summarise price action from candles
    candles_summary = _summarise_candles(candles_around_trade)

    return f"""You are a professional trading coach reviewing a completed scalping trade.
Analyse the trade below and provide an honest, actionable review.

============================================================
TRADE DETAILS
============================================================
Symbol: {symbol}
Direction: {direction}
Entry Price: {entry_price}
Exit Price: {exit_price}
Entry Time: {entry_time}
Exit Time: {exit_time}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
P&L: {pnl:+.2f}
R:R Achieved: {rr_achieved:.2f}
Confidence at Entry: {confidence}
Risk %: {risk_pct:.2f}

============================================================
ORIGINAL AI REASONING
============================================================
{reasoning}

============================================================
INDICATORS AT ENTRY
============================================================
{indicators_text}

============================================================
PRICE ACTION AFTER ENTRY
============================================================
{candles_summary}

============================================================
EVALUATION CRITERIA
============================================================
1. ENTRY TIMING: Was the entry at a good price? Could it have been better?
2. STOP LOSS PLACEMENT: Was the SL placed at a logical level? Too tight or too loose?
3. SHOULD HAVE TRADED: Given what you see, should this trade have been taken at all?
4. MISSED SIGNALS: Were there warning signs that were ignored?
5. TRADE QUALITY: Rate 1-10 overall.

============================================================
RESPONSE FORMAT -- RETURN ONLY VALID JSON, NO MARKDOWN
============================================================
{{
  "entry_quality": "<string: 'good', 'fair', or 'poor' with 1-sentence explanation>",
  "sl_quality": "<string: 'good', 'fair', or 'poor' with 1-sentence explanation>",
  "should_have_traded": "<string: 'yes', 'no', or 'marginal' with 1-sentence explanation>",
  "missed_signals": "<string: describe any missed warning signs, or 'none'>",
  "lesson": "<string: one key takeaway from this trade>",
  "suggested_adjustment": "<string: one concrete change for future trades>"
}}"""


def _build_weekly_prompt(trades: list[dict], reviews: list[dict]) -> str:
    """Assemble the weekly summary prompt for Claude."""

    # Compute aggregate stats
    total_trades = len(trades)
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    breakeven = [t for t in trades if t.get("pnl", 0) == 0]
    win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    avg_winner = (
        sum(t.get("pnl", 0) for t in winning) / len(winning) if winning else 0.0
    )
    avg_loser = (
        sum(t.get("pnl", 0) for t in losing) / len(losing) if losing else 0.0
    )
    avg_rr = (
        sum(t.get("rr_achieved", 0) for t in trades) / total_trades
        if total_trades > 0
        else 0.0
    )

    # Symbols traded
    symbols = sorted({t.get("symbol", "?") for t in trades})

    # Format individual trade summaries
    trade_lines: list[str] = []
    for i, (trade, review) in enumerate(zip(trades, reviews), start=1):
        pnl = trade.get("pnl", 0)
        symbol = trade.get("symbol", "?")
        direction = trade.get("direction", "?")
        rr = trade.get("rr_achieved", 0)
        lesson = review.get("lesson", "N/A") if not review.get("error") else "Review failed"
        entry_q = review.get("entry_quality", "N/A") if not review.get("error") else "N/A"
        trade_lines.append(
            f"  {i}. {symbol} {direction} | PnL={pnl:+.2f} | RR={rr:.2f} | "
            f"Entry={entry_q} | Lesson: {lesson}"
        )
    trades_text = "\n".join(trade_lines) if trade_lines else "  No trades this week."

    # Collect all lessons
    lessons = [
        r.get("lesson", "")
        for r in reviews
        if r.get("lesson") and not r.get("error")
    ]
    lessons_text = "\n".join(f"  - {l}" for l in lessons) if lessons else "  None."

    return f"""You are a professional trading coach providing a weekly performance review
for an AI-driven scalping system. Analyse the week's results and provide
strategic guidance.

============================================================
WEEKLY STATISTICS
============================================================
Total Trades: {total_trades}
Wins: {len(winning)} | Losses: {len(losing)} | Breakeven: {len(breakeven)}
Win Rate: {win_rate:.1f}%
Total P&L: {total_pnl:+.2f}
Average Winner: {avg_winner:+.2f}
Average Loser: {avg_loser:+.2f}
Average R:R Achieved: {avg_rr:.2f}
Symbols Traded: {', '.join(symbols)}

============================================================
TRADE-BY-TRADE SUMMARY
============================================================
{trades_text}

============================================================
LESSONS FROM INDIVIDUAL REVIEWS
============================================================
{lessons_text}

============================================================
EVALUATION CRITERIA
============================================================
1. What is working well in the current strategy?
2. What is consistently failing or underperforming?
3. What should the system focus on improving next week?
4. Overall grade for the week (A/B/C/D/F with +/- modifiers).
5. Key statistics or patterns you notice.

============================================================
RESPONSE FORMAT -- RETURN ONLY VALID JSON, NO MARKDOWN
============================================================
{{
  "whats_working": "<string: 2-3 sentences on strengths>",
  "whats_failing": "<string: 2-3 sentences on weaknesses>",
  "suggested_focus": "<string: 1-2 concrete focus areas for next week>",
  "overall_grade": "<string: letter grade like 'B+' or 'C-'>",
  "key_stats": "<string: notable statistical observations>"
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarise_candles(df: pd.DataFrame) -> str:
    """Create a concise text summary of candle price action."""
    if df is None or df.empty:
        return "No candle data available."

    rows = min(len(df), 30)  # cap to avoid huge prompts
    tail = df.tail(rows)

    high = tail["high"].max()
    low = tail["low"].min()
    open_price = tail.iloc[0]["open"]
    close_price = tail.iloc[-1]["close"]
    net_change = close_price - open_price

    # Count bullish / bearish candles
    bullish = sum(1 for _, r in tail.iterrows() if r["close"] > r["open"])
    bearish = rows - bullish

    # Largest single-candle move
    tail_copy = tail.copy()
    tail_copy["range"] = tail_copy["high"] - tail_copy["low"]
    max_range_idx = tail_copy["range"].idxmax()
    max_range = tail_copy.loc[max_range_idx, "range"]

    # Time range
    start_time = tail.iloc[0].get("time", "?")
    end_time = tail.iloc[-1].get("time", "?")
    if isinstance(start_time, (datetime, pd.Timestamp)):
        start_time = start_time.strftime("%H:%M:%S")
    if isinstance(end_time, (datetime, pd.Timestamp)):
        end_time = end_time.strftime("%H:%M:%S")

    return (
        f"Period: {start_time} to {end_time} ({rows} candles)\n"
        f"High: {high:.5f} | Low: {low:.5f} | Range: {high - low:.5f}\n"
        f"Open: {open_price:.5f} -> Close: {close_price:.5f} | "
        f"Net Change: {net_change:+.5f}\n"
        f"Bullish candles: {bullish} | Bearish candles: {bearish}\n"
        f"Largest single-candle range: {max_range:.5f}"
    )


def _parse_json_response(raw: str, fallback_keys: list[str]) -> dict:
    """Parse a JSON response from Claude, handling common formatting issues.

    On failure, returns a dict with the expected keys set to ``"parse_error"``
    plus an ``"error"`` key with the raw text excerpt.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(
            "json_parse_error",
            error=str(exc),
            raw_excerpt=raw[:300],
        )
        result = {key: "parse_error" for key in fallback_keys}
        result["error"] = f"JSON parse error: {exc}"
        result["raw_response"] = raw[:500]
        return result
