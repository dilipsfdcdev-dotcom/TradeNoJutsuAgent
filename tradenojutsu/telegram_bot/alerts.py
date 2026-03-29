"""Telegram bot for trade alerts, status updates, and remote commands.

Sends real-time notifications for:
- Trade entries and exits
- Daily performance summaries
- AI reasoning highlights
- Learning cycle results
- Risk alerts (drawdown warnings)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from tradenojutsu.data.models import PerformanceMetrics, Signal, Trade
from tradenojutsu.infra.logger import get_logger

logger = get_logger("telegram")


class TelegramAlerter:
    """Sends trading alerts and updates via Telegram.

    Supports both synchronous (for callbacks) and async sending.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot = None
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning("Telegram alerts disabled (no bot_token or chat_id)")

    async def _get_bot(self):
        if self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat."""
        if not self._enabled:
            return False
        try:
            bot = await self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def alert_trade_opened(self, trade: Trade, signal: Signal) -> None:
        """Alert when a new trade is opened."""
        emoji = "🟢" if trade.direction.value == "long" else "🔴"
        msg = (
            f"{emoji} <b>NEW TRADE</b>\n\n"
            f"<b>{trade.direction.value.upper()}</b> {trade.symbol}\n"
            f"Entry: <code>{trade.entry_price:.4f}</code>\n"
            f"Stop Loss: <code>{trade.stop_loss:.4f}</code>\n"
            f"Take Profit: <code>{trade.take_profit:.4f}</code>\n"
            f"Size: <code>{trade.quantity:.4f}</code>\n"
            f"Strategy: {trade.strategy}\n"
            f"Score: {signal.score:.0f}/100 ({signal.strength.value})\n\n"
            f"<i>Reasoning: {signal.reasoning[:200]}...</i>"
        )
        await self.send_message(msg)

    async def alert_trade_closed(self, trade: Trade) -> None:
        """Alert when a trade is closed."""
        pnl = trade.pnl
        emoji = "✅" if pnl > 0 else "❌"
        pnl_sign = "+" if pnl > 0 else ""

        msg = (
            f"{emoji} <b>TRADE CLOSED</b>\n\n"
            f"<b>{trade.direction.value.upper()}</b> {trade.symbol}\n"
            f"Entry: <code>{trade.entry_price:.4f}</code>\n"
            f"Exit: <code>{trade.exit_price:.4f}</code>\n"
            f"PnL: <b>{pnl_sign}{pnl:.2f}</b> ({trade.pnl_pct:+.1f}%)\n"
            f"Strategy: {trade.strategy}"
        )
        await self.send_message(msg)

    async def alert_daily_summary(self, metrics: PerformanceMetrics, daily_pnl: float) -> None:
        """Send daily performance summary."""
        pnl_sign = "+" if daily_pnl > 0 else ""
        emoji = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "➡️"

        msg = (
            f"{emoji} <b>DAILY SUMMARY</b>\n\n"
            f"Today's PnL: <b>{pnl_sign}{daily_pnl:.2f}</b>\n\n"
            f"<b>Overall Performance:</b>\n"
            f"Total Trades: {metrics.total_trades}\n"
            f"Win Rate: {metrics.win_rate:.1%}\n"
            f"Profit Factor: {metrics.profit_factor:.2f}\n"
            f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {metrics.max_drawdown:.2%}\n"
            f"Total PnL: {metrics.total_pnl:+.2f}"
        )
        await self.send_message(msg)

    async def alert_learning_cycle(self, result: dict[str, Any]) -> None:
        """Alert when a learning cycle completes."""
        changes = result.get("changes", [])
        if not changes:
            return

        changes_text = "\n".join(
            f"  • {c['param']}: {c['old']} → {c['new']} ({c['direction']})"
            for c in changes
        )

        msg = (
            f"🧠 <b>LEARNING CYCLE #{result.get('cycle', '?')}</b>\n\n"
            f"<b>Performance:</b> {result.get('metrics', 'N/A')}\n"
            f"<b>Reward:</b> {result.get('reward', 0):.3f}\n\n"
            f"<b>Parameter Changes:</b>\n{changes_text}"
        )
        await self.send_message(msg)

    async def alert_risk_warning(self, warning: str, details: str = "") -> None:
        """Alert on risk limit approaching or hit."""
        msg = (
            f"⚠️ <b>RISK ALERT</b>\n\n"
            f"{warning}\n"
            f"{details}"
        )
        await self.send_message(msg)

    async def alert_ai_thinking(self, symbol: str, decision: dict) -> None:
        """Share interesting AI reasoning (for high-confidence decisions)."""
        confidence = decision.get("confidence", 0)
        if confidence < 70:
            return  # Only share high-confidence thinking

        thinking = decision.get("thinking", "")[:300]
        action = decision.get("decision", "wait")
        emoji = {"long": "📗", "short": "📕", "wait": "📒"}.get(action, "📒")

        msg = (
            f"{emoji} <b>AI THINKING — {symbol}</b>\n\n"
            f"Decision: <b>{action.upper()}</b> (confidence: {confidence}%)\n\n"
            f"<i>{thinking}...</i>"
        )
        await self.send_message(msg)


class TelegramCommandHandler:
    """Handles incoming Telegram commands for remote control.

    Commands:
    /status - Current portfolio status
    /trades - Recent trades
    /performance - Performance metrics
    /stop - Stop the agent
    """

    def __init__(self, bot_token: str, chat_id: str, agent=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.agent = agent
        self._enabled = bool(bot_token and chat_id)

    async def start_listening(self) -> None:
        """Start listening for commands (runs as background task)."""
        if not self._enabled:
            return

        from telegram import Update
        from telegram.ext import Application, CommandHandler

        app = Application.builder().token(self.bot_token).build()

        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("trades", self._cmd_trades))
        app.add_handler(CommandHandler("performance", self._cmd_performance))
        app.add_handler(CommandHandler("help", self._cmd_help))

        logger.info("Telegram command handler started")
        await app.run_polling(allowed_updates=Update.ALL_TYPES)

    async def _cmd_status(self, update, context) -> None:
        from tradenojutsu.infra.database import get_open_trades
        trades = get_open_trades()
        if not trades:
            await update.message.reply_html("📊 <b>No open positions</b>")
            return

        lines = ["📊 <b>OPEN POSITIONS</b>\n"]
        for t in trades:
            emoji = "🟢" if t["direction"] == "long" else "🔴"
            lines.append(
                f"{emoji} {t['direction'].upper()} {t['symbol']} "
                f"@ {t['entry_price']:.4f} | SL: {t['stop_loss']:.4f}"
            )
        await update.message.reply_html("\n".join(lines))

    async def _cmd_trades(self, update, context) -> None:
        from tradenojutsu.infra.database import get_recent_trades
        trades = get_recent_trades(10)
        if not trades:
            await update.message.reply_html("📝 <b>No recent trades</b>")
            return

        lines = ["📝 <b>RECENT TRADES</b>\n"]
        for t in trades:
            pnl = t.get("pnl", 0) or 0
            emoji = "✅" if pnl > 0 else "❌"
            lines.append(
                f"{emoji} {t['direction'].upper()} {t['symbol']} | "
                f"PnL: {pnl:+.2f} ({t.get('strategy', '?')})"
            )
        await update.message.reply_html("\n".join(lines))

    async def _cmd_performance(self, update, context) -> None:
        from tradenojutsu.infra.database import get_recent_trades
        import numpy as np
        trades = get_recent_trades(100)
        closed = [t for t in trades if t.get("pnl") is not None]
        if not closed:
            await update.message.reply_html("📈 <b>No performance data yet</b>")
            return

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / len(closed) * 100

        msg = (
            f"📈 <b>PERFORMANCE</b>\n\n"
            f"Trades: {len(closed)}\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Total PnL: {sum(pnls):+.2f}\n"
            f"Best: {max(pnls):+.2f}\n"
            f"Worst: {min(pnls):+.2f}\n"
            f"Avg: {np.mean(pnls):+.2f}"
        )
        await update.message.reply_html(msg)

    async def _cmd_help(self, update, context) -> None:
        msg = (
            "🤖 <b>TradeNoJutsu Commands</b>\n\n"
            "/status - Open positions\n"
            "/trades - Recent trades\n"
            "/performance - Performance stats\n"
            "/help - This message"
        )
        await update.message.reply_html(msg)
