"""Async Claude Veto Scanner — slow loop that reads market state + news + account
and writes ALLOW / REDUCE / BLOCK decisions to the VetoRegister.

Runs every 30-60 seconds in its own asyncio task.  If the Claude API is
unreachable or returns garbage the scanner defaults to ALLOW so that
trading is **never** blocked by an infrastructure failure.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import anthropic
import structlog

from agent.brain.veto_register import VetoRegister, VetoState, veto_register
from agent.config import settings

logger = structlog.get_logger(__name__)

_VETO_MODEL = "claude-sonnet-4-20250514"
_VETO_MAX_TOKENS = 500
_DEFAULT_INTERVAL_SEC = 45
_MIN_INTERVAL_SEC = 30
_MAX_INTERVAL_SEC = 60
_MAX_CONSECUTIVE_FAILURES = 5


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

VETO_PROMPT = """You are a risk oversight system for an automated scalping bot.
The bot trades XAUUSD, BTCUSD, XAGUSD on 1M/3M using ML models.

IMPORTANT: Your DEFAULT should be ALLOW. Only use REDUCE or BLOCK for truly
dangerous situations like:
  - Account drawdown > 5%
  - 5+ consecutive losses
  - Obvious flash crash / extreme illiquidity

Do NOT block or reduce for:
  - Normal news events (the bot handles these)
  - ML accuracy at baseline 50% (models are new, this is expected)
  - Regular market volatility

## Current time: {utc_time}
## Session: {session}

## Market state per symbol:
{market_state}

## News headlines (last 30 min):
{headlines}

## Economic calendar (next 4 hours):
{events}

## Account state:
  Daily P&L: {daily_pnl_pct}%
  Drawdown from peak: {drawdown_pct}%
  Consecutive losses: {consec_losses}

## ML model health:
  XGB accuracy (last 50): {xgb_accuracy}%
  LSTM accuracy (last 50): {lstm_accuracy}%

## Your task:
For EACH symbol, respond with ONE of:
  ALLOW — let ML trade normally (this should be your default)
  REDUCE:0.5 — reduce size to 50% (only for real danger)
  BLOCK:minutes — block trading (only for extreme situations)

Respond in JSON only:
{{
  "XAUUSD": {{"action": "ALLOW", "value": null, "reason": ""}},
  "BTCUSD": {{"action": "ALLOW", "value": null, "reason": ""}},
  "XAGUSD": {{"action": "ALLOW", "value": null, "reason": ""}}
}}"""


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

async def claude_veto_scan(context: dict) -> dict | None:
    """Called every 30-60 seconds.  Non-blocking.

    Updates veto register.  If API fails, keeps last state (ALLOW-by-default).
    Returns the parsed decisions dict, or None on failure.
    """
    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        prompt = VETO_PROMPT.format(**context)

        response = await asyncio.wait_for(
            client.messages.create(
                model=_VETO_MODEL,
                max_tokens=_VETO_MAX_TOKENS,
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

        decisions = json.loads(text)
        now = datetime.now(timezone.utc)

        for symbol, decision in decisions.items():
            action = decision.get("action", "ALLOW").upper()
            reason = decision.get("reason", "")
            value = decision.get("value")

            if action == "BLOCK":
                minutes = int(value) if value else 30
                veto_register.set_veto(symbol, VetoState(
                    blocked=True, reason=reason,
                    expires_at=now + timedelta(minutes=minutes),
                    source="veto_scanner",
                ))
            elif action == "REDUCE":
                modifier = float(value) if value else 0.5
                veto_register.set_veto(symbol, VetoState(
                    blocked=False, risk_modifier=modifier, reason=reason,
                    expires_at=now + timedelta(minutes=10),
                    source="veto_scanner",
                ))
            else:  # ALLOW
                veto_register.clear_veto(symbol)

        logger.info("veto_scan_complete", decisions=decisions)
        return decisions

    except asyncio.TimeoutError:
        logger.warning("veto_scan_timeout", msg="Keeping last veto state")
        return None
    except json.JSONDecodeError as exc:
        logger.warning("veto_scan_parse_error", error=str(exc))
        return None
    except anthropic.APIConnectionError as exc:
        logger.warning("veto_api_connection_error", error=str(exc))
        return None
    except anthropic.RateLimitError as exc:
        logger.warning("veto_rate_limit", error=str(exc))
        return None
    except anthropic.APIStatusError as exc:
        logger.warning("veto_api_error", status=exc.status_code, error=str(exc))
        return None
    except Exception as exc:
        logger.warning("veto_scan_error", error=str(exc), msg="Keeping last veto state")
        return None


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_veto_context(
    symbols: list[str],
    account_info: dict,
    session: str,
    headlines: list[dict],
    events: list,
    ml_health: dict,
    market_data: dict | None = None,
) -> dict:
    """Build context dict for the veto prompt.

    Parameters
    ----------
    symbols : list[str]
        Symbols the bot trades.
    account_info : dict
        Keys: daily_pnl_pct, drawdown_pct, consecutive_losses.
    session : str
        Current trading session (asian/london/newyork).
    headlines : list[dict]
        Recent news headlines, each with a ``"headline"`` key.
    events : list
        Upcoming economic events.
    ml_health : dict
        Keys: xgb_accuracy, lstm_accuracy.
    market_data : dict | None
        Optional per-symbol market snapshots keyed by symbol.
    """
    if market_data:
        market_lines = []
        for sym in symbols:
            snap = market_data.get(sym, {})
            bid = snap.get("bid", "N/A")
            ask = snap.get("ask", "N/A")
            spread = snap.get("spread", "N/A")
            atr = snap.get("atr", "N/A")
            market_lines.append(f"  {sym}: bid={bid} ask={ask} spread={spread} atr={atr}")
    else:
        market_lines = [f"  {sym}: (data from live feed)" for sym in symbols]

    news_lines = (
        "\n".join(f"  - {h.get('headline', str(h))}" for h in headlines[:10])
        or "  No recent news"
    )
    event_lines = (
        "\n".join(f"  - {getattr(e, 'name', str(e))}" for e in events[:5])
        or "  No upcoming events"
    )

    return {
        "utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "session": session,
        "market_state": "\n".join(market_lines),
        "headlines": news_lines,
        "events": event_lines,
        "daily_pnl_pct": round(account_info.get("daily_pnl_pct", 0), 2),
        "drawdown_pct": round(account_info.get("drawdown_pct", 0), 2),
        "consec_losses": account_info.get("consecutive_losses", 0),
        "xgb_accuracy": round(ml_health.get("xgb_accuracy", 50), 1),
        "lstm_accuracy": round(ml_health.get("lstm_accuracy", 50), 1),
    }


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

class VetoScannerLoop:
    """Background scanner with adaptive interval and failure handling.

    Usage::

        scanner = VetoScannerLoop(context_builder_fn=my_fn)
        asyncio.create_task(scanner.run())
    """

    def __init__(
        self,
        context_builder_fn: Callable[[], dict] | None = None,
        interval: int = _DEFAULT_INTERVAL_SEC,
        register: VetoRegister | None = None,
    ):
        self._build_context = context_builder_fn
        self._base_interval = max(_MIN_INTERVAL_SEC, min(interval, _MAX_INTERVAL_SEC))
        self._register = register or veto_register
        self._running = False
        self._consecutive_failures = 0
        self._scan_count = 0

    async def run(self) -> None:
        """Run the veto scanner forever."""
        self._running = True
        logger.info("veto_scanner_loop_started", interval=self._base_interval)

        while self._running:
            try:
                context = self._build_context() if self._build_context else {}
                result = await claude_veto_scan(context)
                if result is not None:
                    self._consecutive_failures = 0
                    self._scan_count += 1
                else:
                    self._consecutive_failures += 1
            except Exception as exc:
                self._consecutive_failures += 1
                logger.error("veto_loop_error", error=str(exc))

            # Safety: clear all vetoes if scanner has been failing too long
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                self._clear_all_vetoes()
                self._consecutive_failures = 0

            interval = self._compute_interval()
            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Signal the scanner to stop after the current iteration."""
        self._running = False
        logger.info("veto_scanner_loop_stopping")

    def _compute_interval(self) -> float:
        """Adaptive interval: back off on failures."""
        if self._consecutive_failures > 0:
            return min(self._base_interval + self._consecutive_failures * 10, 120)
        return self._base_interval

    def _clear_all_vetoes(self) -> None:
        """Safety: clear all vetoes when scanner is unhealthy."""
        for symbol in settings.symbols_list:
            self._register.clear_veto(symbol)
        self._register.clear_veto("GLOBAL")
        logger.warning(
            "veto_all_cleared_scanner_unhealthy",
            failures=self._consecutive_failures,
        )

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


# ---------------------------------------------------------------------------
# Convenience: simple loop function (backward compatible)
# ---------------------------------------------------------------------------

async def veto_scanner_loop(context_builder_fn: Callable, interval: int = 30) -> None:
    """Simple background loop that runs veto scanning.

    For more control (failure tracking, adaptive intervals), use
    :class:`VetoScannerLoop` instead.
    """
    scanner = VetoScannerLoop(
        context_builder_fn=context_builder_fn,
        interval=interval,
    )
    await scanner.run()
