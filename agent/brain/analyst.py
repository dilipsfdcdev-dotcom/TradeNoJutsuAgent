"""Core AI analysis module -- calls Claude with structured market context to produce trade decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import anthropic
import pandas as pd
import structlog

from agent.config import settings

if TYPE_CHECKING:
    from agent.signals.mtf_analyzer import MTFState

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    action: str           # "buy", "sell", "wait"
    confidence: int       # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    risk_score: int       # 1-10
    symbol: str
    timeframe: str
    # v2 optional fields
    take_profit_2: float | None = None
    lot_size_suggestion: str = "normal"   # "normal" | "reduced" | "increased"
    risk_warnings: str = ""


_WAIT_DECISION_DEFAULTS = dict(
    action="wait",
    confidence=0,
    entry_price=0.0,
    stop_loss=0.0,
    take_profit=0.0,
    reasoning="Analysis inconclusive or error occurred",
    risk_score=10,
    timeframe="M1",
)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _format_indicator_table(df: pd.DataFrame, last_n: int = 5) -> str:
    """Format the last *last_n* rows of a candle DataFrame as a text table.

    Expected columns (added by ``compute_all_indicators``):
        time, open, high, low, close, volume,
        ema_9, ema_21, ema_50, rsi, atr, vwap,
        bb_upper, bb_middle, bb_lower, volume_delta
    """
    if df is None or df.empty:
        return "(no data)"

    tail = df.tail(last_n).copy()

    # Build a compact table
    columns = [
        ("time", "Time"),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Vol"),
        ("ema_9", "EMA9"),
        ("ema_21", "EMA21"),
        ("ema_50", "EMA50"),
        ("rsi", "RSI"),
        ("atr", "ATR"),
        ("vwap", "VWAP"),
        ("bb_upper", "BB_Up"),
        ("bb_lower", "BB_Lo"),
        ("volume_delta", "VolDelta"),
    ]

    # Filter to columns that actually exist
    present = [(col, label) for col, label in columns if col in tail.columns]

    header = " | ".join(label for _, label in present)
    sep = "-" * len(header)
    rows: list[str] = [header, sep]

    for _, row in tail.iterrows():
        parts: list[str] = []
        for col, _ in present:
            val = row[col]
            if col == "time":
                if isinstance(val, (datetime, pd.Timestamp)):
                    parts.append(val.strftime("%H:%M:%S"))
                else:
                    parts.append(str(val))
            elif col in ("volume", "volume_delta"):
                parts.append(f"{val:.0f}" if pd.notna(val) else "N/A")
            elif col == "rsi":
                parts.append(f"{val:.1f}" if pd.notna(val) else "N/A")
            else:
                parts.append(f"{val:.5f}" if pd.notna(val) else "N/A")
        rows.append(" | ".join(parts))

    return "\n".join(rows)


def _format_patterns(patterns: list) -> str:
    """Format a list of PatternSignal objects into readable text."""
    if not patterns:
        return "No patterns detected."

    lines: list[str] = []
    for p in patterns[:10]:  # limit to most recent 10
        lines.append(
            f"- {p.type} | strength={p.strength:.2f} | "
            f"price_level={p.price_level:.5f} | candle_idx={p.candle_index}"
        )
    return "\n".join(lines)


def _format_positions(positions: list) -> str:
    """Format open positions into readable text."""
    if not positions:
        return "No open positions."

    lines: list[str] = []
    for pos in positions:
        if isinstance(pos, dict):
            symbol = pos.get("symbol", "?")
            ptype = pos.get("type", "?")
            volume = pos.get("volume", 0)
            price_open = pos.get("price_open", 0)
            sl = pos.get("sl", 0)
            tp = pos.get("tp", 0)
            profit = pos.get("profit", 0)
            lines.append(
                f"- {symbol} {ptype} {volume} lots @ {price_open:.5f} | "
                f"SL={sl:.5f} TP={tp:.5f} | PnL={profit:.2f}"
            )
        else:
            # Assume MT5 position object with attributes
            lines.append(
                f"- {getattr(pos, 'symbol', '?')} "
                f"{getattr(pos, 'type', '?')} "
                f"{getattr(pos, 'volume', 0)} lots @ "
                f"{getattr(pos, 'price_open', 0):.5f} | "
                f"SL={getattr(pos, 'sl', 0):.5f} "
                f"TP={getattr(pos, 'tp', 0):.5f} | "
                f"PnL={getattr(pos, 'profit', 0):.2f}"
            )
    return "\n".join(lines)


def _format_recent_trades(trades: list) -> str:
    """Format recent trades into readable text."""
    if not trades:
        return "No recent trades."

    lines: list[str] = []
    for t in trades:
        if isinstance(t, dict):
            symbol = t.get("symbol", "?")
            action = t.get("action", "?")
            profit = t.get("profit", 0)
            entry = t.get("entry_price", 0)
            exit_p = t.get("exit_price", 0)
            lines.append(
                f"- {symbol} {action} entry={entry:.5f} exit={exit_p:.5f} PnL={profit:.2f}"
            )
        else:
            lines.append(
                f"- {getattr(t, 'symbol', '?')} "
                f"{getattr(t, 'action', '?')} "
                f"entry={getattr(t, 'entry_price', 0):.5f} "
                f"exit={getattr(t, 'exit_price', 0):.5f} "
                f"PnL={getattr(t, 'profit', 0):.2f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    symbol: str,
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    patterns: list,
    market_context,
    sentiment,
    account_info: dict,
    open_positions: list,
    recent_trades: list,
    daily_pnl: float,
    mtf_state: MTFState | None = None,
    xgb_result: dict | None = None,
    lstm_result: dict | None = None,
) -> str:
    """Assemble the full analysis prompt sent to Claude.

    If *mtf_state*, *xgb_result*, and *lstm_result* are provided the v2
    three-brain prompt is used.  Otherwise the original v1 prompt is
    returned for backward compatibility.
    """

    balance = account_info.get("balance", 0)
    equity = account_info.get("equity", 0)
    margin = account_info.get("margin", 0)
    free_margin = account_info.get("free_margin", 0)
    current_profit = account_info.get("profit", 0)

    max_daily_loss = balance * (settings.MAX_DAILY_LOSS_PCT / 100.0)
    remaining_daily_risk = max_daily_loss - abs(min(daily_pnl, 0))

    # Spread from the latest 1m candle (high - low as proxy if no tick spread)
    spread = 0.0
    if candles_1m is not None and not candles_1m.empty:
        last = candles_1m.iloc[-1]
        spread = last.get("spread", last["high"] - last["low"]) if "spread" in candles_1m.columns else 0.0

    # Key levels as comma-separated string
    key_levels_str = ", ".join(f"{lvl:.5f}" for lvl in (market_context.key_levels or [])[:8])

    # Consecutive losses from recent trades
    consecutive_losses = 0
    for t in recent_trades:
        pnl = t.get("profit", 0) if isinstance(t, dict) else getattr(t, "profit", 0)
        if pnl < 0:
            consecutive_losses += 1
        else:
            break

    # Sentiment text
    sentiment_text = "N/A"
    if sentiment is not None:
        sentiment_text = (
            f"Score: {sentiment.score:+.2f} | Impact: {sentiment.impact} | "
            f"{sentiment.reasoning}"
        )

    # -----------------------------------------------------------------
    # v2 prompt (three-brain hybrid architecture)
    # -----------------------------------------------------------------
    if mtf_state is not None and xgb_result is not None and lstm_result is not None:
        return _build_v2_prompt(
            symbol=symbol,
            candles_1m=candles_1m,
            candles_3m=candles_3m,
            patterns=patterns,
            market_context=market_context,
            sentiment_text=sentiment_text,
            account_info=account_info,
            open_positions=open_positions,
            recent_trades=recent_trades,
            daily_pnl=daily_pnl,
            balance=balance,
            equity=equity,
            margin=margin,
            free_margin=free_margin,
            current_profit=current_profit,
            remaining_daily_risk=remaining_daily_risk,
            spread=spread,
            key_levels_str=key_levels_str,
            consecutive_losses=consecutive_losses,
            mtf_state=mtf_state,
            xgb_result=xgb_result,
            lstm_result=lstm_result,
        )

    # -----------------------------------------------------------------
    # v1 prompt (original, backward-compatible)
    # -----------------------------------------------------------------

    prompt = f"""You are a professional intraday scalping analyst for {symbol}.
Analyze the following market data and decide whether to BUY, SELL, or WAIT.

============================================================
ACCOUNT STATE
============================================================
Balance: ${balance:,.2f}
Equity: ${equity:,.2f}
Margin Used: ${margin:,.2f}
Free Margin: ${free_margin:,.2f}
Open P&L: ${current_profit:,.2f}
Daily P&L: ${daily_pnl:,.2f}
Remaining Daily Risk Budget: ${remaining_daily_risk:,.2f}
Open Positions Count: {len(open_positions)}
Max Allowed Open Trades: {settings.MAX_OPEN_TRADES}

Open Positions:
{_format_positions(open_positions)}

============================================================
MARKET CONTEXT
============================================================
Symbol: {symbol}
Session: {market_context.session}
Trend (3m structure): {market_context.trend}
ATR (3m, 14-period): {market_context.atr:.5f}
Volatility Rank: {market_context.volatility_rank}
Key Levels: {key_levels_str if key_levels_str else 'None detected'}
Spread: {spread:.5f}

============================================================
TECHNICAL INDICATORS -- 1-MINUTE CANDLES (last 5)
============================================================
{_format_indicator_table(candles_1m, last_n=5)}

============================================================
TECHNICAL INDICATORS -- 3-MINUTE CANDLES (last 5)
============================================================
{_format_indicator_table(candles_3m, last_n=5)}

============================================================
DETECTED PATTERNS (most recent first)
============================================================
{_format_patterns(patterns)}

============================================================
NEWS SENTIMENT
============================================================
{sentiment_text}

============================================================
RECENT TRADES ON {symbol} (newest first)
============================================================
{_format_recent_trades(recent_trades)}
Consecutive Losses: {consecutive_losses}

============================================================
RULES -- YOU MUST FOLLOW THESE
============================================================
1. TRADE WITH THE TREND: Only take BUY signals in a bullish trend, SELL in bearish. In ranging markets, only take high-confidence setups (>= 80).
2. MINIMUM RISK:REWARD: Every trade must have at least {settings.MIN_RR_RATIO}:1 reward-to-risk ratio. Calculate (TP - Entry) / (Entry - SL) for buys.
3. Use news sentiment as context but do NOT block trades based on news alone.
4. Use recent trade history as context for position sizing, not for blocking.
5. Focus on technical setups -- if there is a valid entry, take it.
8. STOP LOSS: Must be placed beyond the nearest structure level or ATR-based distance. Never risk more than {settings.MAX_RISK_PER_TRADE_PCT}% of balance.
9. TAKE PROFIT: Place at the next key level or ATR-multiple target. Must respect minimum R:R.
10. CONFIDENCE: Be honest about confidence. Only HIGH confidence (>= 70) trades should be executed.

============================================================
RESPONSE FORMAT -- RETURN ONLY VALID JSON, NO MARKDOWN
============================================================
{{
  "ACTION": "buy" | "sell" | "wait",
  "CONFIDENCE": <integer 0-100>,
  "ENTRY_PRICE": <float -- current ask for buy, bid for sell, 0 for wait>,
  "STOP_LOSS": <float -- price level, 0 for wait>,
  "TAKE_PROFIT": <float -- price level, 0 for wait>,
  "REASONING": "<2-4 sentences explaining the decision>",
  "RISK_SCORE": <integer 1-10, where 1=very safe, 10=very risky>
}}"""

    return prompt


# ---------------------------------------------------------------------------
# v2 prompt builder (three-brain hybrid)
# ---------------------------------------------------------------------------

def _build_v2_prompt(
    *,
    symbol: str,
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    patterns: list,
    market_context,
    sentiment_text: str,
    account_info: dict,
    open_positions: list,
    recent_trades: list,
    daily_pnl: float,
    balance: float,
    equity: float,
    margin: float,
    free_margin: float,
    current_profit: float,
    remaining_daily_risk: float,
    spread: float,
    key_levels_str: str,
    consecutive_losses: int,
    mtf_state: MTFState,
    xgb_result: dict,
    lstm_result: dict,
) -> str:
    """Build the v2 three-brain hybrid prompt with MTF state + ML scores."""

    # --- Multi-timeframe section ---
    tf = mtf_state.timeframes if hasattr(mtf_state, "timeframes") else {}
    h1 = tf.get("1H", {}) if isinstance(tf, dict) else {}
    m15 = tf.get("15M", {}) if isinstance(tf, dict) else {}
    m3 = tf.get("3M", {}) if isinstance(tf, dict) else {}
    m1 = tf.get("1M", {}) if isinstance(tf, dict) else {}

    def _tf_line(label: str, d: dict) -> str:
        if not d:
            return f"  {label}: (no data)"
        trend = d.get("trend", "N/A")
        ema_pos = d.get("ema_position", "N/A")
        rsi = d.get("rsi", "N/A")
        atr = d.get("atr", "N/A")
        structure = d.get("structure", "N/A")
        return (
            f"  {label}: trend={trend} | ema_position={ema_pos} | "
            f"rsi={rsi} | atr={atr} | structure={structure}"
        )

    mtf_section = "\n".join([
        _tf_line("1H  (bias)", h1),
        _tf_line("15M (structure)", m15),
        _tf_line("3M  (confirmation)", m3),
        _tf_line("1M  (signal)", m1),
    ])

    # --- ML model scores section ---
    xgb_score = xgb_result.get("score", 0.0)
    xgb_top = xgb_result.get("top_features", [])
    xgb_features_str = ", ".join(
        f"{name}={imp}" for name, imp in xgb_top[:5]
    ) if xgb_top else "N/A"

    lstm_conf = lstm_result.get("confidence", 0.0)
    lstm_dir = lstm_result.get("direction", "N/A")
    lstm_regime = lstm_result.get("regime", "N/A")

    # --- Confluence score ---
    confluence = getattr(mtf_state, "confluence_score", None)
    if confluence is None:
        # Compute a simple confluence from available data
        confluence = round((xgb_score + lstm_conf) / 2.0 * 100, 1)
    confluence_str = f"{confluence}"

    # --- Setup narrative ---
    setup_narrative = getattr(mtf_state, "setup_narrative", "N/A")

    # --- Gates summary ---
    gates_passed = getattr(mtf_state, "gates_passed", False)
    gate_details = getattr(mtf_state, "gate_details", "")

    prompt = f"""You are an expert scalping trader. You receive a multi-timeframe \
analysis with pre-computed ML scores. Your job is to make the \
FINAL decision: trade or wait.

============================================================
ACCOUNT STATE
============================================================
Balance: ${balance:,.2f}
Equity: ${equity:,.2f}
Margin Used: ${margin:,.2f}
Free Margin: ${free_margin:,.2f}
Open P&L: ${current_profit:,.2f}
Daily P&L: ${daily_pnl:,.2f}
Remaining Daily Risk Budget: ${remaining_daily_risk:,.2f}
Open Positions Count: {len(open_positions)}
Max Allowed Open Trades: {settings.MAX_OPEN_TRADES}

Open Positions:
{_format_positions(open_positions)}

============================================================
MULTI-TIMEFRAME ANALYSIS
============================================================
{mtf_section}

Gates Passed: {gates_passed}
{f"Gate Details: {gate_details}" if gate_details else ""}

============================================================
ML MODEL SCORES
============================================================
XGBoost Score: {xgb_score:.4f}
  Top Features: {xgb_features_str}

LSTM Confidence: {lstm_conf:.4f}
  Direction: {lstm_dir}
  Regime: {lstm_regime}

============================================================
CONFLUENCE
============================================================
Confluence Score: {confluence_str}
Setup Narrative: {setup_narrative}

============================================================
MARKET CONTEXT
============================================================
Symbol: {symbol}
Session: {market_context.session}
Trend (3m structure): {market_context.trend}
ATR (3m, 14-period): {market_context.atr:.5f}
Volatility Rank: {market_context.volatility_rank}
Key Levels: {key_levels_str if key_levels_str else 'None detected'}
Spread: {spread:.5f}

============================================================
TECHNICAL INDICATORS -- 1-MINUTE CANDLES (last 5)
============================================================
{_format_indicator_table(candles_1m, last_n=5)}

============================================================
TECHNICAL INDICATORS -- 3-MINUTE CANDLES (last 5)
============================================================
{_format_indicator_table(candles_3m, last_n=5)}

============================================================
DETECTED PATTERNS (most recent first)
============================================================
{_format_patterns(patterns)}

============================================================
NEWS SENTIMENT
============================================================
{sentiment_text}

============================================================
RECENT TRADES ON {symbol} (newest first)
============================================================
{_format_recent_trades(recent_trades)}
Consecutive Losses: {consecutive_losses}

============================================================
RULES -- YOU MUST FOLLOW THESE
============================================================
1. TRADE WITH THE TREND: 1H bias sets direction. Only BUY when 1H+15M are bullish, SELL when bearish.
2. CONFLUENCE GATE: All three brains (XGBoost, LSTM, your analysis) must agree for a trade. If ML models show conflicting direction, output WAIT.
3. MINIMUM RISK:REWARD: Every trade must have at least {settings.MIN_RR_RATIO}:1 reward-to-risk ratio.
4. Use news sentiment as context but do NOT block trades based on news alone.
5. Use recent trade history as context for sizing, not blocking.
6. Focus on technical setups -- if there is a valid entry, take it.
9. STOP LOSS: Place beyond the nearest structure level or ATR-based distance. Never risk more than {settings.MAX_RISK_PER_TRADE_PCT}% of balance.
10. TAKE PROFIT: Set TP1 at the nearest structure level (partial close), TP2 at the extended target.
11. LOT SIZING: Suggest "reduced" if regime is volatile or confluence < 60, "increased" if regime is trending and confluence > 85, otherwise "normal".
12. RISK WARNINGS: Note any concerns (divergence between ML models, unusual regime, thin liquidity, etc.).

============================================================
RESPONSE FORMAT -- RETURN ONLY VALID JSON, NO MARKDOWN
============================================================
{{
  "ACTION": "buy" | "sell" | "wait",
  "CONFIDENCE": <integer 0-100>,
  "ENTRY_PRICE": <float -- current ask for buy, bid for sell, 0 for wait>,
  "STOP_LOSS": <float -- price level, 0 for wait>,
  "TAKE_PROFIT": <float -- price level for TP1, 0 for wait>,
  "TAKE_PROFIT_2": <float -- price level for TP2 (extended target), 0 for wait>,
  "LOT_SIZE_SUGGESTION": "normal" | "reduced" | "increased",
  "RISK_WARNINGS": "<any risk concerns or empty string>",
  "REASONING": "<2-4 sentences explaining the decision, referencing MTF alignment and ML scores>",
  "RISK_SCORE": <integer 1-10, where 1=very safe, 10=very risky>
}}"""

    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(response_text: str, symbol: str) -> TradeDecision:
    """Parse Claude's JSON response into a TradeDecision.

    On any parsing failure, returns a safe WAIT decision.
    """
    try:
        text = response_text.strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)

        action = str(data.get("ACTION", "wait")).lower().strip()
        if action not in ("buy", "sell", "wait"):
            action = "wait"

        confidence = int(data.get("CONFIDENCE", 0))
        confidence = max(0, min(100, confidence))

        entry_price = float(data.get("ENTRY_PRICE", 0))
        stop_loss = float(data.get("STOP_LOSS", 0))
        take_profit = float(data.get("TAKE_PROFIT", 0))
        reasoning = str(data.get("REASONING", "No reasoning provided"))
        risk_score = int(data.get("RISK_SCORE", 5))
        risk_score = max(1, min(10, risk_score))

        # v2 optional fields
        take_profit_2 = float(data.get("TAKE_PROFIT_2", 0)) or None
        lot_size_suggestion = str(data.get("LOT_SIZE_SUGGESTION", "normal")).lower().strip()
        if lot_size_suggestion not in ("normal", "reduced", "increased"):
            lot_size_suggestion = "normal"
        risk_warnings = str(data.get("RISK_WARNINGS", ""))

        return TradeDecision(
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            risk_score=risk_score,
            symbol=symbol,
            timeframe="M1",
            take_profit_2=take_profit_2,
            lot_size_suggestion=lot_size_suggestion,
            risk_warnings=risk_warnings,
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.error("analyst_parse_error", symbol=symbol, error=str(exc), raw=response_text[:300])
        return TradeDecision(symbol=symbol, **_WAIT_DECISION_DEFAULTS)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

async def analyze(
    symbol: str,
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    patterns: list,
    market_context,
    sentiment,
    account_info: dict,
    open_positions: list,
    recent_trades: list,
    daily_pnl: float,
    *,
    mtf_state: MTFState | None = None,
    xgb_result: dict | None = None,
    lstm_result: dict | None = None,
) -> TradeDecision:
    """Run Claude-based analysis and return a structured trade decision.

    Parameters
    ----------
    symbol : str
        Trading instrument (e.g. ``"XAUUSD"``).
    candles_1m : pd.DataFrame
        1-minute OHLCV DataFrame with indicators already computed.
    candles_3m : pd.DataFrame
        3-minute OHLCV DataFrame with indicators already computed.
    patterns : list[PatternSignal]
        Pattern signals detected on the 1-minute chart.
    market_context : MarketContext
        Higher-timeframe market structure snapshot.
    sentiment : SentimentResult
        Current news sentiment for the symbol.
    account_info : dict
        Keys: balance, equity, margin, free_margin, profit.
    open_positions : list
        Currently open positions (MT5 position objects or dicts).
    recent_trades : list
        Last 10 closed trades on this symbol (newest first).
    daily_pnl : float
        Realised P&L for the current trading day.
    mtf_state : MTFState | None
        v2: Multi-timeframe state from ``MTFAnalyzer.update()``.
    xgb_result : dict | None
        v2: XGBoost filter result (``{"score", "pass", "top_features"}``).
    lstm_result : dict | None
        v2: LSTM confidence result (``{"direction", "confidence", "regime", "pass"}``).

    Returns
    -------
    TradeDecision
        The AI's structured recommendation.  The caller should check
        ``decision.confidence >= 70`` before executing.
    """
    prompt = _build_prompt(
        symbol=symbol,
        candles_1m=candles_1m,
        candles_3m=candles_3m,
        patterns=patterns,
        market_context=market_context,
        sentiment=sentiment,
        account_info=account_info,
        open_positions=open_positions,
        recent_trades=recent_trades,
        daily_pnl=daily_pnl,
        mtf_state=mtf_state,
        xgb_result=xgb_result,
        lstm_result=lstm_result,
    )

    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        logger.info(
            "analyst_response_received",
            symbol=symbol,
            response_length=len(response_text),
        )

        decision = _parse_response(response_text, symbol)

        logger.info(
            "analyst_decision",
            symbol=symbol,
            action=decision.action,
            confidence=decision.confidence,
            entry=decision.entry_price,
            sl=decision.stop_loss,
            tp=decision.take_profit,
            risk_score=decision.risk_score,
        )

        return decision

    except anthropic.APIConnectionError as exc:
        logger.error("analyst_api_connection_error", symbol=symbol, error=str(exc))
        return TradeDecision(symbol=symbol, **_WAIT_DECISION_DEFAULTS)
    except anthropic.RateLimitError as exc:
        logger.error("analyst_rate_limit", symbol=symbol, error=str(exc))
        return TradeDecision(symbol=symbol, **_WAIT_DECISION_DEFAULTS)
    except anthropic.APIStatusError as exc:
        logger.error("analyst_api_error", symbol=symbol, status=exc.status_code, error=str(exc))
        return TradeDecision(symbol=symbol, **_WAIT_DECISION_DEFAULTS)
    except Exception as exc:
        logger.error("analyst_unexpected_error", symbol=symbol, error=str(exc))
        return TradeDecision(symbol=symbol, **_WAIT_DECISION_DEFAULTS)
