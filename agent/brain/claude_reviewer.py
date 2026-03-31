"""Post-trade Claude reviewer — rates ML model decisions and generates training labels.

Unlike ``self_review.py`` (which reviews Claude's own trade reasoning), this
module reviews the **ML models'** decisions: was the XGBoost score appropriate?
Did the LSTM direction/confidence match reality?  Should the rule engine have
filtered this trade?

The output is structured review data that feeds back into the training pipeline
as labelled data for future retraining cycles.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable

import anthropic
import structlog

from agent.config import settings

logger = structlog.get_logger(__name__)

_REVIEW_MODEL = "claude-sonnet-4-20250514"
_REVIEW_MAX_TOKENS = 512
_WEEKLY_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """Review this closed trade made by an ML-driven trading bot:

## Trade details
- Symbol: {symbol}, Direction: {direction}
- Entry: {entry_price} at {entry_time}
- Exit: {exit_price} at {exit_time}
- P&L: {pnl} ({pnl_pips} pips)
- R:R achieved: {rr_actual}
- Lot size: {lot_size}

## ML scores at entry
- XGBoost score: {xgb_score}
- LSTM confidence: {lstm_confidence}, direction: {lstm_direction}
- Confluence: {confluence}/100

## Reasoning factors (deterministic)
{reasoning_factors}

## Market conditions at entry
{conditions}

## What happened after entry
{price_action}

## Questions
1. Was the ML signal correct in identifying this setup?
2. Was the entry timing optimal?
3. Was the SL placement logical (behind structure)?
4. Did the ML models miss any warning signs?
5. Rate the overall ML decision quality 1-10.

Respond in JSON only:
{{
  "ml_signal_quality": <int 1-10>,
  "entry_timing": <int 1-10>,
  "sl_quality": <int 1-10>,
  "missed_warnings": ["..."],
  "lesson": "<one-line takeaway>",
  "suggested_feature": "<specific feature to add to ML if any, or null>",
  "should_have_traded": <true/false>,
  "suggested_xgb_label": <1 or 0>,
  "suggested_lstm_confidence": <float 0.0-1.0>
}}"""


# ---------------------------------------------------------------------------
# Default review (used on API failure — never block the pipeline)
# ---------------------------------------------------------------------------

def _default_review() -> dict:
    return {
        "ml_signal_quality": 5,
        "entry_timing": 5,
        "sl_quality": 5,
        "missed_warnings": [],
        "lesson": "Review unavailable",
        "suggested_feature": None,
        "should_have_traded": True,
        "suggested_xgb_label": None,
        "suggested_lstm_confidence": None,
    }


# ---------------------------------------------------------------------------
# Single trade review
# ---------------------------------------------------------------------------

async def review_trade(trade_data: dict, candles_around: list | None = None) -> dict:
    """Review a single closed ML-driven trade.

    Parameters
    ----------
    trade_data : dict
        Must contain: ``symbol``, ``direction``, ``entry_price``, ``exit_price``,
        ``entry_time``, ``exit_time``, ``pnl``, ``pnl_pips``, ``rr_actual``,
        ``lot_size``, ``xgb_score``, ``lstm_confidence``, ``lstm_direction``,
        ``confluence``, ``reasoning_factors`` (list[str]),
        ``conditions`` (str), ``price_action_summary`` (str).
    candles_around : list | None
        Optional candle data around the trade (not used in prompt currently,
        reserved for future visual analysis).

    Returns
    -------
    dict
        Structured review.  Returns safe defaults on any API failure.
    """
    try:
        conditions = trade_data.get("conditions", "N/A")
        price_action = trade_data.get("price_action_summary", "N/A")
        reasoning = "\n".join(trade_data.get("reasoning_factors", ["N/A"]))

        prompt = REVIEW_PROMPT.format(
            symbol=trade_data.get("symbol", "?"),
            direction=trade_data.get("direction", "?"),
            entry_price=trade_data.get("entry_price", 0),
            exit_price=trade_data.get("exit_price", 0),
            entry_time=trade_data.get("entry_time", "?"),
            exit_time=trade_data.get("exit_time", "?"),
            pnl=trade_data.get("pnl", 0),
            pnl_pips=trade_data.get("pnl_pips", 0),
            rr_actual=trade_data.get("rr_actual", 0),
            lot_size=trade_data.get("lot_size", 0),
            xgb_score=trade_data.get("xgb_score", "N/A"),
            lstm_confidence=trade_data.get("lstm_confidence", "N/A"),
            lstm_direction=trade_data.get("lstm_direction", "N/A"),
            confluence=trade_data.get("confluence", 0),
            reasoning_factors=reasoning,
            conditions=conditions,
            price_action=price_action,
        )

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await asyncio.wait_for(
            client.messages.create(
                model=_REVIEW_MODEL,
                max_tokens=_REVIEW_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=15.0,
        )

        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        review = json.loads(text)
        logger.info(
            "trade_reviewed",
            symbol=trade_data.get("symbol"),
            quality=review.get("ml_signal_quality"),
        )
        return review

    except anthropic.APIConnectionError as exc:
        logger.warning("reviewer_api_connection_error", error=str(exc))
        return _default_review()
    except anthropic.RateLimitError as exc:
        logger.warning("reviewer_rate_limit", error=str(exc))
        return _default_review()
    except anthropic.APIStatusError as exc:
        logger.warning("reviewer_api_error", status=exc.status_code)
        return _default_review()
    except asyncio.TimeoutError:
        logger.warning("reviewer_timeout")
        return _default_review()
    except json.JSONDecodeError as exc:
        logger.warning("reviewer_parse_error", error=str(exc))
        return _default_review()
    except Exception as exc:
        logger.warning("review_failed", error=str(exc))
        return _default_review()


# ---------------------------------------------------------------------------
# Weekly batch summary
# ---------------------------------------------------------------------------

async def generate_weekly_summary(trades: list[dict], reviews: list[dict]) -> dict:
    """Weekly strategy summary focused on ML model performance.

    Parameters
    ----------
    trades : list[dict]
        Closed trades for the week.
    reviews : list[dict]
        Corresponding reviews from ``review_trade``.

    Returns
    -------
    dict
        Keys: ``whats_working``, ``whats_failing``, ``ml_assessment``,
        ``improvements``, ``overall_grade``, ``retrain_priority``.
    """
    fallback = {
        "whats_working": "Review unavailable",
        "whats_failing": "Review unavailable",
        "ml_assessment": "Review unavailable",
        "improvements": [],
        "overall_grade": "N/A",
        "retrain_priority": "none",
    }

    if not trades:
        return fallback

    try:
        total = len(trades)
        wins = sum(1 for t in trades if (t.get("pnl", 0) or 0) > 0)
        losses = total - wins
        total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        avg_quality = sum(
            r.get("ml_signal_quality", 5) for r in reviews
        ) / max(len(reviews), 1)
        lessons = [r.get("lesson", "") for r in reviews if r.get("lesson")]
        features = [
            r.get("suggested_feature")
            for r in reviews
            if r.get("suggested_feature")
        ]
        bad_trades = sum(1 for r in reviews if not r.get("should_have_traded", True))

        prompt = f"""Weekly ML trading bot review:

Trades: {total} ({wins}W / {losses}L)
Total P&L: ${total_pnl:.2f}
Avg ML quality score: {avg_quality:.1f}/10
Trades that should NOT have been taken: {bad_trades}
Lessons: {'; '.join(lessons[:10])}
Suggested new features: {', '.join(str(f) for f in features[:5]) or 'None'}

Provide a strategy-level assessment:
1. What's working well?
2. What needs improvement?
3. Are the ML models making good decisions?
4. Top 3 actionable improvements
5. Should we prioritize retraining? (none/low/medium/high)

Respond in JSON only:
{{
  "whats_working": "<2-3 sentences on strengths>",
  "whats_failing": "<2-3 sentences on weaknesses>",
  "ml_assessment": "<2-3 sentences on model quality>",
  "improvements": ["<actionable item 1>", "<actionable item 2>", "<actionable item 3>"],
  "overall_grade": "<letter grade A-F with +/- modifiers>",
  "retrain_priority": "none" | "low" | "medium" | "high"
}}"""

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = await asyncio.wait_for(
            client.messages.create(
                model=_REVIEW_MODEL,
                max_tokens=_WEEKLY_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=20.0,
        )

        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        logger.info("weekly_ml_summary_received", trades=total, grade=result.get("overall_grade"))
        return result

    except anthropic.APIConnectionError as exc:
        logger.warning("weekly_api_connection_error", error=str(exc))
        return fallback
    except anthropic.RateLimitError as exc:
        logger.warning("weekly_rate_limit", error=str(exc))
        return fallback
    except anthropic.APIStatusError as exc:
        logger.warning("weekly_api_error", status=exc.status_code)
        return fallback
    except asyncio.TimeoutError:
        logger.warning("weekly_summary_timeout")
        return fallback
    except json.JSONDecodeError as exc:
        logger.warning("weekly_summary_parse_error", error=str(exc))
        return fallback
    except Exception as exc:
        logger.warning("weekly_summary_failed", error=str(exc))
        return fallback


# ---------------------------------------------------------------------------
# Training label extraction
# ---------------------------------------------------------------------------

def extract_training_labels(trade: dict, review: dict) -> dict | None:
    """Extract training labels from a trade + its review for the ML pipeline.

    Parameters
    ----------
    trade : dict
        The closed trade data.
    review : dict
        The review from ``review_trade``.

    Returns
    -------
    dict or None
        Training label data, or None if insufficient information.
    """
    xgb_label = review.get("suggested_xgb_label")
    lstm_conf_target = review.get("suggested_lstm_confidence")

    # Fall back to outcome-based labels if Claude didn't provide them
    if xgb_label is None:
        pnl = trade.get("pnl", 0) or 0
        should_have = review.get("should_have_traded", True)
        if pnl > 0 and should_have:
            xgb_label = 1
        elif pnl <= 0 or not should_have:
            xgb_label = 0

    if lstm_conf_target is None:
        quality = review.get("ml_signal_quality", 5)
        lstm_conf_target = quality / 10.0  # map 1-10 to 0.0-1.0

    return {
        "trade_id": str(trade.get("id", "")),
        "symbol": trade.get("symbol", ""),
        "timestamp": trade.get("entry_time", ""),
        "direction": trade.get("direction", ""),
        "pnl": trade.get("pnl", 0),
        "rr_achieved": trade.get("rr_actual", 0),
        # XGBoost training label
        "xgb_label": int(xgb_label) if xgb_label is not None else None,
        # LSTM training targets
        "lstm_direction_label": trade.get("direction", "wait"),
        "lstm_confidence_target": round(float(lstm_conf_target), 4) if lstm_conf_target is not None else None,
        # Context at trade time
        "xgb_score_at_trade": trade.get("xgb_score"),
        "lstm_confidence_at_trade": trade.get("lstm_confidence"),
        # Review quality indicators
        "ml_signal_quality": review.get("ml_signal_quality"),
        "entry_timing": review.get("entry_timing"),
        "should_have_traded": review.get("should_have_traded", True),
        "lesson": review.get("lesson", ""),
        "suggested_feature": review.get("suggested_feature"),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Background reviewer loop
# ---------------------------------------------------------------------------

class ReviewerLoop:
    """Background loop that reviews trades as they close.

    Usage::

        reviewer = ReviewerLoop(
            get_unreviewed_trades=my_fetch_fn,
            store_review=my_save_fn,
            store_training_label=my_label_fn,
        )
        asyncio.create_task(reviewer.run())
    """

    def __init__(
        self,
        get_unreviewed_trades: Callable | None = None,
        store_review: Callable | None = None,
        store_training_label: Callable | None = None,
        interval_sec: int = 300,
    ):
        self._get_trades = get_unreviewed_trades
        self._store_review = store_review
        self._store_label = store_training_label
        self._interval = interval_sec
        self._running = False
        self._reviewed_ids: set[str] = set()
        self._review_count = 0

    async def run(self) -> None:
        """Run the reviewer forever."""
        self._running = True
        logger.info("reviewer_loop_started", interval=self._interval)

        while self._running:
            try:
                await self._cycle()
            except Exception as exc:
                logger.error("reviewer_loop_error", error=str(exc))
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        logger.info("reviewer_loop_stopping")

    async def _cycle(self) -> None:
        """One review cycle."""
        trades = await self._safe_call(self._get_trades)
        if not trades:
            return

        unreviewed = [
            t for t in trades
            if str(t.get("id", "")) not in self._reviewed_ids
        ]

        if not unreviewed:
            return

        logger.info("reviewer_found_unreviewed", count=len(unreviewed))

        for trade in unreviewed:
            trade_id = str(trade.get("id", ""))
            try:
                review = await review_trade(trade)

                # Store review
                if self._store_review:
                    await self._safe_call(self._store_review, trade_id, review)

                # Extract and store training labels
                labels = extract_training_labels(trade, review)
                if self._store_label and labels:
                    await self._safe_call(self._store_label, trade_id, labels)

                self._reviewed_ids.add(trade_id)
                self._review_count += 1

                # Keep set bounded
                if len(self._reviewed_ids) > 10000:
                    ids_list = sorted(self._reviewed_ids)
                    self._reviewed_ids = set(ids_list[-5000:])

            except Exception as exc:
                logger.error("reviewer_trade_error", trade_id=trade_id, error=str(exc))

    @property
    def review_count(self) -> int:
        return self._review_count

    async def _safe_call(self, fn: Callable | None, *args: Any) -> Any:
        """Call a function safely; return None on any error."""
        if fn is None:
            return None
        try:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:
            logger.warning("reviewer_callback_error",
                           fn=getattr(fn, "__name__", "?"), error=str(exc))
            return None
