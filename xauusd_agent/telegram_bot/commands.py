"""
Telegram command handlers for the XAUUSD trading agent.

All handlers live inside :class:`TelegramCommands` so they can share
a DB pool and an optional executor reference without relying on
``context.bot_data`` alone.

All responses use ``parse_mode="HTML"`` for safe formatting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.ext import ContextTypes

from xauusd_agent.infra.logger import get_logger

if TYPE_CHECKING:
    import asyncpg

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "\u2501" * 21  # ━━━━━━━━━━━━━━━━━━━━━
_SEP_LONG = "\u2501" * 29


def _fmt_usd(v: float) -> str:
    if v >= 0:
        return f"+${v:,.2f}"
    return f"-${abs(v):,.2f}"


def _sign(v: float, decimals: int = 2) -> str:
    return f"+{v:.{decimals}f}" if v >= 0 else f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# DB micro-helpers (keeps handlers self-contained)
# ---------------------------------------------------------------------------


async def _get_state(pool: asyncpg.Pool, key: str) -> str | None:
    """Read a single key from ``bot_state``."""
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value FROM bot_state WHERE key = $1", key
        )
    return row["value"] if row else None


async def _set_state(pool: asyncpg.Pool, key: str, value: str) -> None:
    """Upsert a key in ``bot_state``."""
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO bot_state (key, value, updated_at) VALUES ($1, $2, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()",
            key,
            value,
        )


async def _push_command(
    pool: asyncpg.Pool, command: str, value: str = ""
) -> None:
    """Insert a command into ``bot_commands`` for the main loop."""
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO bot_commands (command, value) VALUES ($1, $2)",
            command,
            value,
        )


async def _reply(update: Update, text: str) -> None:
    """Send an HTML-formatted reply."""
    await update.message.reply_text(text, parse_mode="HTML")


# ---------------------------------------------------------------------------
# TelegramCommands
# ---------------------------------------------------------------------------


class TelegramCommands:
    """Collection of ``/command`` handlers for the trading bot.

    Parameters
    ----------
    db_pool : asyncpg.Pool
        Shared database connection pool.
    executor : object, optional
        Trade executor (used for emergency halt).
    """

    def __init__(self, db_pool: asyncpg.Pool, executor=None) -> None:
        self.db: asyncpg.Pool = db_pool
        self.executor = executor

    # ------------------------------------------------------------------ #
    # /status                                                              #
    # ------------------------------------------------------------------ #

    async def cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Bot state, balance, equity, regime, HTF bias, open positions."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            status = await _get_state(pool, "status") or "UNKNOWN"
            balance = await _get_state(pool, "balance") or "N/A"
            equity = await _get_state(pool, "equity") or "N/A"
            regime = await _get_state(pool, "regime") or "N/A"
            htf_direction = await _get_state(pool, "htf_direction") or "N/A"
            htf_score = await _get_state(pool, "htf_score") or "N/A"
            risk_pct = await _get_state(pool, "risk_pct") or "N/A"

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS cnt FROM trades WHERE status = 'OPEN'"
                )
                open_count = row["cnt"] if row else 0

            lines = [
                "\U0001f4cb <b>BOT STATUS</b>",
                _SEP,
                f"Status:    <b>{status}</b>",
                (
                    f"Balance:   ${float(balance):,.2f}"
                    if balance != "N/A"
                    else f"Balance:   {balance}"
                ),
                (
                    f"Equity:    ${float(equity):,.2f}"
                    if equity != "N/A"
                    else f"Equity:    {equity}"
                ),
                f"Risk:      {risk_pct}%",
                f"Open pos:  {open_count}",
                _SEP,
                f"Regime:    {regime}",
                f"HTF bias:  {htf_direction} ({htf_score})",
            ]
            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_status failed")
            await _reply(update, "\u274c Error fetching status.")

    # ------------------------------------------------------------------ #
    # /trades                                                              #
    # ------------------------------------------------------------------ #

    async def cmd_trades(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """All open trades: entry, current P&L, current SL, locked profit."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT ticket, direction, open_price, sl_price, tp_price, "
                    "lot_size, profit_usd, profit_R, profit_locked_usd, "
                    "ratchet_triggered "
                    "FROM trades WHERE status = 'OPEN' ORDER BY open_time"
                )

            if not rows:
                await _reply(update, "\U0001f4ad No open trades.")
                return

            lines = [f"\U0001f4c8 <b>OPEN TRADES</b> ({len(rows)})", _SEP]
            for r in rows:
                pnl = r["profit_usd"] or 0
                pnl_r = r["profit_R"] or 0
                locked = r["profit_locked_usd"] or 0
                ratchet_icon = "\U0001f512" if r["ratchet_triggered"] else ""
                lines.append(
                    f"#{r['ticket']} {r['direction']} @ ${r['open_price']:,.2f}\n"
                    f"  SL: ${r['sl_price']:,.2f} | TP: ${r['tp_price']:,.2f}\n"
                    f"  P&amp;L: {_fmt_usd(pnl)} ({_sign(pnl_r)}R) "
                    f"{ratchet_icon}"
                )
                if locked:
                    lines.append(f"  Locked: ${locked:,.0f}")
                lines.append("")

            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_trades failed")
            await _reply(update, "\u274c Error fetching trades.")

    # ------------------------------------------------------------------ #
    # /stop                                                                #
    # ------------------------------------------------------------------ #

    async def cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Pause signal loop.  Ratchet NEVER stops."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            await _push_command(pool, "stop")
            await _set_state(pool, "status", "PAUSED")
            await _reply(
                update,
                "\u23f8\ufe0f <b>PAUSED</b>\n"
                "Signal loop stopped. Ratchet SL management remains active.\n"
                "Use /resume to restart.",
            )
            logger.info("Bot paused via /stop command")

        except Exception:
            logger.exception("cmd_stop failed")
            await _reply(update, "\u274c Error executing stop.")

    # ------------------------------------------------------------------ #
    # /resume                                                              #
    # ------------------------------------------------------------------ #

    async def cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Resume signal loop.  Set status=RUNNING."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            await _push_command(pool, "resume")
            await _set_state(pool, "status", "RUNNING")
            await _reply(
                update,
                "\u25b6\ufe0f <b>RESUMED</b>\nSignal loop restarted.",
            )
            logger.info("Bot resumed via /resume command")

        except Exception:
            logger.exception("cmd_resume failed")
            await _reply(update, "\u274c Error executing resume.")

    # ------------------------------------------------------------------ #
    # /halt                                                                #
    # ------------------------------------------------------------------ #

    async def cmd_halt(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """EMERGENCY: close ALL positions + stop everything."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            await _push_command(pool, "halt")
            await _set_state(pool, "status", "HALTED")
            await _reply(
                update,
                "\U0001f6a8 <b>EMERGENCY HALT</b>\n"
                "Closing ALL positions and stopping all loops.\n"
                "Manual intervention required to restart.",
            )
            logger.warning("EMERGENCY HALT triggered via /halt command")

        except Exception:
            logger.exception("cmd_halt failed")
            await _reply(update, "\u274c Error executing halt.")

    # ------------------------------------------------------------------ #
    # /risk                                                                #
    # ------------------------------------------------------------------ #

    async def cmd_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/risk X -- set risk % (e.g. ``/risk 0.8``).  Validates 0.1-2.0."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        args = context.args
        if not args:
            current = await _get_state(pool, "risk_pct") or "N/A"
            await _reply(
                update, f"Current risk: <b>{current}%</b>\nUsage: /risk 0.8"
            )
            return

        try:
            value = float(args[0])
        except (ValueError, IndexError):
            await _reply(update, "\u26a0\ufe0f Invalid value. Usage: /risk 0.8")
            return

        if not 0.1 <= value <= 2.0:
            await _reply(
                update, "\u26a0\ufe0f Risk must be between 0.1% and 2.0%."
            )
            return

        try:
            await _push_command(pool, "risk", str(value))
            await _set_state(pool, "risk_pct", str(value))
            await _reply(
                update, f"\u2705 Risk updated to <b>{value:.1f}%</b>"
            )
            logger.info("Risk updated to %s via /risk command", value)

        except Exception:
            logger.exception("cmd_risk failed")
            await _reply(update, "\u274c Error updating risk.")

    # ------------------------------------------------------------------ #
    # /pnl                                                                 #
    # ------------------------------------------------------------------ #

    async def cmd_pnl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Today / this week / this month P&L breakdown."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=now.weekday())
            month_start = today_start.replace(day=1)

            _PNL_SQL = (
                "SELECT COALESCE(SUM(profit_usd), 0) AS pnl, "
                "COUNT(*) FILTER (WHERE profit_usd > 0) AS wins, "
                "COUNT(*) FILTER (WHERE profit_usd <= 0) AS losses "
                "FROM trades WHERE close_time >= $1 AND status != 'OPEN'"
            )

            async with pool.acquire() as conn:
                day_row = await conn.fetchrow(_PNL_SQL, today_start)
                week_row = await conn.fetchrow(_PNL_SQL, week_start)
                month_row = await conn.fetchrow(_PNL_SQL, month_start)

            def _row_str(label: str, row: Any) -> str:
                pnl = float(row["pnl"])
                wins = row["wins"]
                losses = row["losses"]
                total = wins + losses
                wr = (wins / total * 100) if total > 0 else 0
                return (
                    f"{label}:  {_fmt_usd(pnl)}  "
                    f"({wins}W/{losses}L, {wr:.0f}% WR)"
                )

            lines = [
                "\U0001f4b0 <b>P&amp;L BREAKDOWN</b>",
                _SEP,
                _row_str("Today", day_row),
                _row_str("Week ", week_row),
                _row_str("Month", month_row),
            ]
            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_pnl failed")
            await _reply(update, "\u274c Error fetching P&L.")

    # ------------------------------------------------------------------ #
    # /summary                                                             #
    # ------------------------------------------------------------------ #

    async def cmd_summary(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Force daily summary now."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT profit_usd, profit_R, ratchet_triggered, "
                    "profit_locked_usd, status "
                    "FROM trades WHERE open_time >= $1 OR "
                    "(close_time >= $1 AND status != 'OPEN')",
                    today_start,
                )

            wins = sum(
                1
                for r in rows
                if r["status"] != "OPEN" and (r["profit_usd"] or 0) > 0
            )
            losses = sum(
                1
                for r in rows
                if r["status"] != "OPEN" and (r["profit_usd"] or 0) <= 0
            )
            open_count = sum(1 for r in rows if r["status"] == "OPEN")
            total = len(rows)
            pnl = sum(
                float(r["profit_usd"] or 0)
                for r in rows
                if r["status"] != "OPEN"
            )

            ratchet_saves = sum(1 for r in rows if r["ratchet_triggered"])
            ratchet_protected = sum(
                float(r["profit_locked_usd"] or 0)
                for r in rows
                if r["ratchet_triggered"]
            )

            winner_rs = [
                float(r["profit_R"])
                for r in rows
                if r["status"] != "OPEN"
                and (r["profit_usd"] or 0) > 0
                and r["profit_R"]
            ]
            loser_rs = [
                float(r["profit_R"])
                for r in rows
                if r["status"] != "OPEN"
                and (r["profit_usd"] or 0) <= 0
                and r["profit_R"]
            ]

            avg_w = sum(winner_rs) / len(winner_rs) if winner_rs else 0
            avg_l = sum(loser_rs) / len(loser_rs) if loser_rs else 0

            balance = await _get_state(pool, "balance") or "0"
            balance_f = float(balance)
            pnl_pct = (pnl / balance_f * 100) if balance_f else 0
            regime = await _get_state(pool, "regime") or "N/A"
            htf_score = float(await _get_state(pool, "htf_score") or "0")

            from xauusd_agent.telegram_bot.alerts import format_daily_summary

            msg = format_daily_summary(
                {
                    "date": now,
                    "trades_total": total,
                    "wins": wins,
                    "losses": losses,
                    "open_count": open_count,
                    "pnl_usd": pnl,
                    "pnl_pct": pnl_pct,
                    "ratchet_saves": ratchet_saves,
                    "ratchet_protected_usd": ratchet_protected,
                    "avg_winner_r": avg_w,
                    "avg_loser_r": avg_l,
                    "regime_summary": regime,
                    "htf_avg": htf_score,
                    "learning_summary": "Use /learned for details",
                    "balance": balance_f,
                }
            )
            await _reply(update, msg)

        except Exception:
            logger.exception("cmd_summary failed")
            await _reply(update, "\u274c Error generating summary.")

    # ------------------------------------------------------------------ #
    # /bias                                                                #
    # ------------------------------------------------------------------ #

    async def cmd_bias(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Current HTF bias: D1/H4/H1 scores + direction + strength."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT htf_score, d1_score, h4_score, h1_score "
                    "FROM regime_log ORDER BY timestamp DESC LIMIT 1"
                )

            if not row:
                await _reply(update, "\U0001f4ad No HTF data yet.")
                return

            htf = float(row["htf_score"] or 0)
            d1 = float(row["d1_score"] or 0)
            h4 = float(row["h4_score"] or 0)
            h1 = float(row["h1_score"] or 0)

            abs_htf = abs(htf)
            if abs_htf >= 0.5:
                strength = "Strong"
            elif abs_htf >= 0.2:
                strength = "Moderate"
            else:
                strength = "Weak"

            direction = "BULL" if htf >= 0 else "BEAR"

            lines = [
                "\U0001f9ed <b>HTF BIAS</b>",
                _SEP,
                f"Direction: <b>{direction}</b> ({strength})",
                f"Composite: {_sign(htf)}",
                "",
                f"D1 score:  {_sign(d1)}",
                f"H4 score:  {_sign(h4)}",
                f"H1 score:  {_sign(h1)}",
            ]
            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_bias failed")
            await _reply(update, "\u274c Error fetching bias.")

    # ------------------------------------------------------------------ #
    # /regime                                                              #
    # ------------------------------------------------------------------ #

    async def cmd_regime(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Current market regime + ATR ratio."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            regime = await _get_state(pool, "regime") or "N/A"

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT regime, atr_ratio, atr_m5, timestamp "
                    "FROM regime_log ORDER BY timestamp DESC LIMIT 1"
                )

            atr_ratio = float(row["atr_ratio"]) if row and row["atr_ratio"] else 0
            atr_m5 = float(row["atr_m5"]) if row and row["atr_m5"] else 0
            ts = row["timestamp"] if row else None
            ts_str = ts.strftime("%H:%M UTC") if ts else "N/A"

            lines = [
                "\U0001f30a <b>MARKET REGIME</b>",
                _SEP,
                f"Regime:    <b>{regime}</b>",
                f"ATR ratio: {atr_ratio:.2f}x",
                f"ATR M5:    {atr_m5:.2f}",
                f"Updated:   {ts_str}",
            ]
            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_regime failed")
            await _reply(update, "\u274c Error fetching regime.")

    # ------------------------------------------------------------------ #
    # /threshold                                                           #
    # ------------------------------------------------------------------ #

    async def cmd_threshold(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Current signal entry threshold."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            threshold = await _get_state(pool, "entry_threshold") or "N/A"
            await _reply(
                update, f"\U0001f3af <b>Entry Threshold:</b> {threshold}"
            )

        except Exception:
            logger.exception("cmd_threshold failed")
            await _reply(update, "\u274c Error fetching threshold.")

    # ------------------------------------------------------------------ #
    # /learned                                                             #
    # ------------------------------------------------------------------ #

    async def cmd_learned(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Last learning cycle: what changed and why."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT param_name, old_value, new_value, reason, "
                    "trades_analysed, win_rate, timestamp "
                    "FROM learning_log ORDER BY timestamp DESC LIMIT 10"
                )

            if not rows:
                await _reply(
                    update, "\U0001f4ad No learning cycles recorded yet."
                )
                return

            # Group by the most recent timestamp (within 60 s)
            latest_ts = rows[0]["timestamp"]
            recent = [
                r
                for r in rows
                if abs((r["timestamp"] - latest_ts).total_seconds()) < 60
            ]

            ts_str = latest_ts.strftime("%d %b %Y %H:%M UTC")
            trades_analysed = recent[0]["trades_analysed"] if recent else "N/A"
            win_rate = recent[0]["win_rate"] if recent else None

            lines = [
                "\U0001f9e0 <b>LAST LEARNING CYCLE</b>",
                _SEP,
                f"Time:     {ts_str}",
                f"Analysed: {trades_analysed} trades",
            ]
            if win_rate is not None:
                lines.append(f"Win rate: {win_rate:.1f}%")

            lines.append("")
            lines.append("Changes:")
            for r in recent:
                old = f"{r['old_value']:g}" if r["old_value"] is not None else "?"
                new = f"{r['new_value']:g}" if r["new_value"] is not None else "?"
                bullet = f"\u2022 {r['param_name']}: {old} \u2192 {new}"
                if r["reason"]:
                    bullet += f"\n  <i>{r['reason']}</i>"
                lines.append(bullet)

            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_learned failed")
            await _reply(update, "\u274c Error fetching learning data.")

    # ------------------------------------------------------------------ #
    # /scores                                                              #
    # ------------------------------------------------------------------ #

    async def cmd_scores(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Last signal's component scores (debug mode)."""
        pool = self.db
        if pool is None:
            await _reply(update, "\u26a0\ufe0f Database not available.")
            return

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT timestamp, action, buy_score, sell_score, "
                    "entry_threshold, ppo_action, ppo_confidence, "
                    "lgbm_probability, llm_action, llm_reason, "
                    "htf_score, htf_direction, regime, "
                    "rsi_m3, rsi_m5, macd_m3, spread_pips, "
                    "was_executed, skip_reason, is_counter_trend, news_blackout "
                    "FROM signals ORDER BY timestamp DESC LIMIT 1"
                )

            if not row:
                await _reply(update, "\U0001f4ad No signals recorded yet.")
                return

            ts_str = row["timestamp"].strftime("%H:%M:%S UTC")
            action = row["action"]
            executed = "\u2705" if row["was_executed"] else "\u274c"
            skip = f" ({row['skip_reason']})" if row["skip_reason"] else ""

            lines = [
                f"\U0001f50d <b>LAST SIGNAL</b> \u2014 {ts_str}",
                _SEP,
                f"Action:   <b>{action}</b> {executed}{skip}",
                f"Buy:      {row['buy_score'] or 0:.1f} | Sell: {row['sell_score'] or 0:.1f}",
                f"Threshold: {row['entry_threshold'] or 0:.1f}",
                "",
                f"PPO:      action={row['ppo_action']} conf={row['ppo_confidence'] or 0:.2f}",
                f"LGBM:     prob={row['lgbm_probability'] or 0:.3f}",
                f"LLM:      {row['llm_action'] or 'N/A'}",
                "",
                f"HTF:      {row['htf_direction'] or 'N/A'} ({row['htf_score'] or 0:.2f})",
                f"Regime:   {row['regime'] or 'N/A'}",
                f"RSI:      M3={row['rsi_m3'] or 0:.1f} M5={row['rsi_m5'] or 0:.1f}",
                f"MACD M3:  {row['macd_m3'] or 0:.4f}",
                f"Spread:   {row['spread_pips'] or 0:.1f} pips",
            ]

            if row["is_counter_trend"]:
                lines.append("\u26a0\ufe0f Counter-trend signal")
            if row["news_blackout"]:
                lines.append("\U0001f4f0 News blackout active")

            await _reply(update, "\n".join(lines))

        except Exception:
            logger.exception("cmd_scores failed")
            await _reply(update, "\u274c Error fetching scores.")
