"""
Telegram alert message formatters and sender for the XAUUSD trading agent.

Pure formatting functions (``format_*``) return ready-to-send HTML strings.
:class:`TelegramAlerts` wraps them with the bot's ``send_message`` method so
the main trading loop can fire alerts with a single call.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from xauusd_agent.infra.logger import get_logger

if TYPE_CHECKING:
    from xauusd_agent.telegram_bot.bot import TradingBot

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEP = "\u2501" * 21  # ━━━━━━━━━━━━━━━━━━━━━
_SEP_LONG = "\u2501" * 29  # longer separator for wider blocks


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_usd(value: float) -> str:
    """Format a USD value with comma separators and sign."""
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def _fmt_usd_precise(value: float) -> str:
    """Format a USD value with two decimal places and sign."""
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _fmt_price(price: float) -> str:
    """Format a gold price to two decimals with ``$`` prefix."""
    return f"${price:,.2f}"


def _sign(value: float) -> str:
    """Return ``+`` prefix for positive values."""
    return f"+{value}" if value >= 0 else str(value)


def _htf_arrow(score: float) -> str:
    """Return directional arrow based on HTF score."""
    if score > 0.1:
        return "\u2191"  # up arrow
    if score < -0.1:
        return "\u2193"  # down arrow
    return "\u2194"  # left-right arrow


def _format_hold_duration(
    open_time: datetime | str | None,
    close_time: datetime | str | None,
) -> str:
    """Return a human-readable hold duration string."""
    if open_time is None or close_time is None:
        return "N/A"

    if isinstance(open_time, str):
        open_time = datetime.fromisoformat(open_time)
    if isinstance(close_time, str):
        close_time = datetime.fromisoformat(close_time)

    delta = close_time - open_time
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "N/A"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes} minutes"


# ---------------------------------------------------------------------------
# Trade open formatter
# ---------------------------------------------------------------------------


def format_trade_open(trade: dict) -> str:
    """Format a trade-opened alert message.

    Expected *trade* keys::

        ticket, direction, symbol, open_price, lot_size, sl_price, tp_price,
        risk_usd, balance_at_entry, risk_pct, sl_pips, tp_pips, tp_ratio,
        htf_direction, htf_score, regime, buy_score/sell_score, threshold,
        rsi_m3, rsi_m5, llm_action
    """
    direction = trade.get("direction", "BUY").upper()
    icon = "\U0001f7e2" if direction == "BUY" else "\U0001f534"  # green / red
    ticket = trade.get("ticket", 0)
    symbol = trade.get("symbol", "XAUUSD")

    entry = _fmt_price(trade.get("open_price", 0))
    lot = trade.get("lot_size", 0)
    risk_pct = trade.get("risk_pct", 0)
    balance = trade.get("balance_at_entry", 0)
    sl = _fmt_price(trade.get("sl_price", 0))
    tp = _fmt_price(trade.get("tp_price", 0))
    sl_pips = trade.get("sl_pips", 0)
    tp_pips = trade.get("tp_pips", 0)
    risk_usd = trade.get("risk_usd", 0)
    tp_ratio = trade.get("tp_ratio", 0)

    htf_dir = trade.get("htf_direction", "N/A")
    htf_score = trade.get("htf_score", 0)
    regime = trade.get("regime", "N/A")
    score_val = (
        trade.get("buy_score") if direction == "BUY" else trade.get("sell_score")
    )
    score_val = score_val if score_val is not None else 0
    threshold = trade.get("threshold", 0)
    rsi_m3 = trade.get("rsi_m3", 0)
    rsi_m5 = trade.get("rsi_m5", 0)
    llm_action = trade.get("llm_action", "N/A")

    arrow = _htf_arrow(htf_score)

    lines = [
        f"{icon} <b>{direction} {symbol}</b> \u2014 #{ticket}",
        _SEP,
        f"Entry:  {entry}",
        f"Lot:    {lot:.2f} ({risk_pct:.1f}% of ${balance:,.0f})",
        f"SL:     {sl} ({_sign(round(-abs(sl_pips), 1))} pips)",
        f"TP:     {tp} ({_sign(round(abs(tp_pips), 1))} pips)",
        f"Risk:   ${abs(risk_usd):,.0f} | R target: {tp_ratio:.1f}R",
        _SEP,
        f"HTF:    {htf_dir} ({_sign(round(htf_score, 2))}) {arrow}",
        f"Regime: {regime}",
        f"Score:  {direction} {score_val:.0f} / threshold {threshold:.0f}",
        f"M3 RSI: {rsi_m3:.0f} | M5 RSI: {rsi_m5:.0f}",
        f"LLM:    {llm_action}",
    ]

    msg = "\n".join(lines)
    logger.debug("Formatted trade_open alert", extra={"ticket": ticket})
    return msg


# ---------------------------------------------------------------------------
# Trade close formatter
# ---------------------------------------------------------------------------


def format_trade_close(trade: dict) -> str:
    """Format a trade-closed alert message.

    Expected *trade* keys::

        ticket, direction, symbol, profit_usd, profit_R, profit_pips,
        open_time, close_time, profit_locked_usd, ratchet_triggered, balance
    """
    profit_usd = trade.get("profit_usd", 0)
    profit_r = trade.get("profit_R", 0)
    is_win = profit_usd >= 0
    icon = "\u2705" if is_win else "\u274c"
    result_icon = "\u2705" if is_win else "\u274c"

    direction = trade.get("direction", "BUY").upper()
    symbol = trade.get("symbol", "XAUUSD")
    ticket = trade.get("ticket", 0)

    profit_pips = trade.get("profit_pips", 0)
    balance = trade.get("balance", 0)
    locked = trade.get("profit_locked_usd", 0)
    ratchet = trade.get("ratchet_triggered", False)

    held_str = _format_hold_duration(
        trade.get("open_time"), trade.get("close_time")
    )

    lines = [
        f"{icon} <b>CLOSED</b> \u2014 {symbol} {direction} #{ticket}",
        _SEP_LONG,
        f"Result:  {_fmt_usd(profit_usd)} ({_sign(round(profit_r, 2))}R) {result_icon}",
        f"Pips:    {_sign(round(profit_pips, 1))}",
        f"Held:    {held_str}",
    ]

    if ratchet and locked:
        lines.append(f"Locked:  ${locked:,.0f} was protected by ratchet")

    lines += [
        _SEP_LONG,
        f"Balance: ${balance:,.0f}",
    ]

    msg = "\n".join(lines)
    logger.debug("Formatted trade_close alert", extra={"ticket": ticket})
    return msg


# ---------------------------------------------------------------------------
# Ratchet alert formatter
# ---------------------------------------------------------------------------


def format_ratchet_alert(
    position: dict,
    new_sl: float,
    profit_r: float,
    ratchet_level: str,
) -> str:
    """Format a stop-loss ratchet alert.

    Args:
        position:      Dict with ticket, direction, symbol, sl_price,
                        open_price, profit_usd.
        new_sl:        The new stop-loss price.
        profit_r:      Current profit in R-multiples.
        ratchet_level: Level name, e.g. ``"BREAKEVEN"``, ``"0.5R"``, ``"1.0R"``.
    """
    ticket = position.get("ticket", 0)
    direction = position.get("direction", "BUY").upper()
    symbol = position.get("symbol", "XAUUSD")
    old_sl = position.get("sl_price", position.get("current_sl", 0))
    profit_usd = position.get("profit_usd", position.get("profit", 0))

    is_breakeven = ratchet_level.upper() in ("BREAKEVEN", "BE", "0.5R")
    label = "BREAKEVEN LOCKED" if is_breakeven else "SL RATCHET"

    lines = [
        f"\U0001f512 <b>{label}</b> \u2014 {symbol} {direction} #{ticket}",
        _SEP,
        f"Profit: {_fmt_usd(profit_usd)} ({_sign(round(profit_r, 2))}R)",
        f"SL: {_fmt_price(old_sl)} \u2192 {_fmt_price(new_sl)}",
    ]

    if is_breakeven:
        lines.append("<i>You cannot lose on this trade now.</i>")
    else:
        open_price = position.get("open_price", 0)
        if open_price and direction == "BUY":
            locked_pct = ((new_sl - open_price) / (open_price or 1)) * 100
            lines.append(f"<i>Locking {abs(locked_pct):.0f}% of profit.</i>")
        elif open_price and direction == "SELL":
            locked_pct = ((open_price - new_sl) / (open_price or 1)) * 100
            lines.append(f"<i>Locking {abs(locked_pct):.0f}% of profit.</i>")
        else:
            lines.append("<i>Locking profit.</i>")

    msg = "\n".join(lines)
    logger.debug(
        "Formatted ratchet alert",
        extra={"ticket": ticket, "level": ratchet_level},
    )
    return msg


# ---------------------------------------------------------------------------
# Daily summary formatter
# ---------------------------------------------------------------------------


def format_daily_summary(data: dict) -> str:
    """Format the end-of-day summary.

    Expected *data* keys::

        date, trades_total, wins, losses, open_count, pnl_usd, pnl_pct,
        ratchet_saves, ratchet_protected_usd, avg_winner_r, avg_loser_r,
        regime_summary, htf_avg, learning_summary, balance
    """
    date_val = data.get("date")
    if isinstance(date_val, datetime):
        date_str = date_val.strftime("%d %b %Y")
    elif isinstance(date_val, str):
        date_str = date_val
    else:
        date_str = datetime.now(timezone.utc).strftime("%d %b %Y")

    trades_total = data.get("trades_total", 0)
    wins = data.get("wins", 0)
    losses = data.get("losses", 0)
    open_count = data.get("open_count", 0)
    pnl_usd = data.get("pnl_usd", 0)
    pnl_pct = data.get("pnl_pct", 0)
    ratchet_saves = data.get("ratchet_saves", 0)
    ratchet_protected = data.get("ratchet_protected_usd", 0)
    avg_winner_r = data.get("avg_winner_r", 0)
    avg_loser_r = data.get("avg_loser_r", 0)
    regime_summary = data.get("regime_summary", "N/A")
    htf_avg = data.get("htf_avg", 0)
    learning_summary = data.get("learning_summary", "No changes")
    balance = data.get("balance", 0)

    # Win/loss/open breakdown
    wl_parts = [f"{wins}W", f"{losses}L"]
    if open_count:
        wl_parts.append(f"{open_count} open")
    wl_str = " / ".join(wl_parts)

    # HTF description
    abs_htf = abs(htf_avg)
    if abs_htf >= 0.5:
        htf_desc = "Strong bull" if htf_avg > 0 else "Strong bear"
    elif abs_htf >= 0.2:
        htf_desc = "Moderate bull" if htf_avg > 0 else "Moderate bear"
    else:
        htf_desc = "Neutral"

    lines = [
        f"\U0001f4ca <b>DAILY SUMMARY</b> \u2014 {date_str}",
        _SEP_LONG,
        f"Trades:       {trades_total}  ({wl_str})",
        f"P&amp;L:          {_fmt_usd(pnl_usd)} ({_sign(round(pnl_pct, 1))}%)",
        f"Ratchet saves: {ratchet_saves} trades (protected ${ratchet_protected:,.0f})",
        f"Avg winner:   {_sign(round(avg_winner_r, 2))}R",
        f"Avg loser:    {_sign(round(avg_loser_r, 2))}R",
        _SEP_LONG,
        f"Regime:  {regime_summary}",
        f"HTF avg: {_sign(round(htf_avg, 2))} ({htf_desc})",
        f"Learning: {learning_summary}",
        f"Balance: ${balance:,.0f}",
    ]

    msg = "\n".join(lines)
    logger.debug("Formatted daily summary")
    return msg


# ---------------------------------------------------------------------------
# Learning cycle formatter
# ---------------------------------------------------------------------------


def format_learning_alert(changes: list) -> str:
    """Format a learning-cycle completion alert.

    *changes* is a list of dicts, each with:
        ``param_name``, ``old_value``, ``new_value``, ``reason``
    """
    if not changes:
        return (
            "\U0001f9e0 <b>LEARNING CYCLE COMPLETE</b>\n\nNo parameter changes."
        )

    trades_analysed = (
        changes[0].get("trades_analysed", "N/A") if changes else "N/A"
    )

    lines = [
        "\U0001f9e0 <b>LEARNING CYCLE COMPLETE</b>",
        _SEP_LONG,
        f"Analysed: {trades_analysed} trades",
        "",
        "Changes:",
    ]

    for change in changes:
        name = change.get("param_name", "?")
        old = change.get("old_value", "?")
        new = change.get("new_value", "?")
        reason = change.get("reason", "")

        if isinstance(old, float):
            old = f"{old:g}"
        if isinstance(new, float):
            new = f"{new:g}"

        bullet = f"\u2022 {name}: {old} \u2192 {new}"
        if reason:
            bullet += f" ({reason})"
        lines.append(bullet)

    msg = "\n".join(lines)
    logger.debug("Formatted learning alert", extra={"changes": len(changes)})
    return msg


# ---------------------------------------------------------------------------
# Regime change formatter
# ---------------------------------------------------------------------------


def format_regime_change(
    old_regime: str,
    new_regime: str,
    details: dict | None = None,
) -> str:
    """Format a market-regime change alert.

    *details* may contain:
        ``atr_ratio``, ``lot_adjustment`` (e.g. ``"75%"``), ``htf_score``
    """
    details = details or {}
    atr_ratio = details.get("atr_ratio")
    lot_adj = details.get("lot_adjustment")

    lines = [
        "\u26a1 <b>REGIME CHANGE</b>",
        _SEP,
        f"{old_regime} \u2192 {new_regime}",
    ]

    if atr_ratio is not None:
        lines.append(f"ATR ratio: {atr_ratio:.1f}x average")

    if lot_adj is not None:
        lines.append(f"Lot size reduced to {lot_adj}")

    msg = "\n".join(lines)
    logger.debug(
        "Formatted regime change alert",
        extra={"old": old_regime, "new": new_regime},
    )
    return msg


# ---------------------------------------------------------------------------
# Error alert formatter
# ---------------------------------------------------------------------------


def format_error_alert(error_msg: str) -> str:
    """Format a critical error alert."""
    lines = [
        "\U0001f6a8 <b>CRITICAL ERROR</b>",
        _SEP,
        error_msg,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TelegramAlerts  --  high-level sender class
# ---------------------------------------------------------------------------


class TelegramAlerts:
    """Wraps formatter functions with :meth:`TradingBot.send_message` calls.

    Each method formats the alert, sends it, and returns ``True``/``False``.

    Parameters
    ----------
    bot : TradingBot
        The bot instance that owns the ``send_message`` method.
    """

    def __init__(self, bot: TradingBot) -> None:
        self.bot = bot

    async def trade_opened(
        self, trade: dict, signal: dict | None = None, htf_bias: dict | None = None
    ) -> bool:
        """Format and send a trade-open alert.

        *signal* and *htf_bias* are merged into *trade* before formatting so
        callers can pass data from different sources.
        """
        merged = dict(trade)
        if signal:
            merged.setdefault("buy_score", signal.get("buy_score"))
            merged.setdefault("sell_score", signal.get("sell_score"))
            merged.setdefault("threshold", signal.get("entry_threshold"))
            merged.setdefault("rsi_m3", signal.get("rsi_m3"))
            merged.setdefault("rsi_m5", signal.get("rsi_m5"))
            merged.setdefault("llm_action", signal.get("llm_action"))
        if htf_bias:
            merged.setdefault("htf_direction", htf_bias.get("direction"))
            merged.setdefault("htf_score", htf_bias.get("score"))
        return await self.bot.send_message(format_trade_open(merged))

    async def trade_closed(self, trade: dict) -> bool:
        """Format and send a trade-close alert."""
        return await self.bot.send_message(format_trade_close(trade))

    async def ratchet_update(
        self,
        position: dict,
        new_sl: float,
        profit_r: float,
        level: str,
    ) -> bool:
        """Format and send a ratchet SL-move alert."""
        return await self.bot.send_message(
            format_ratchet_alert(position, new_sl, profit_r, level)
        )

    async def daily_summary(self, summary_data: dict) -> bool:
        """Format and send the daily summary."""
        return await self.bot.send_message(format_daily_summary(summary_data))

    async def learning_update(self, changes: list) -> bool:
        """Format and send learning cycle results."""
        return await self.bot.send_message(format_learning_alert(changes))

    async def regime_change(
        self,
        old_regime: str,
        new_regime: str,
        details: dict | None = None,
    ) -> bool:
        """Alert when regime changes."""
        return await self.bot.send_message(
            format_regime_change(old_regime, new_regime, details)
        )

    async def error_alert(self, error_msg: str) -> bool:
        """Send critical error alert."""
        return await self.bot.send_message(format_error_alert(error_msg))
