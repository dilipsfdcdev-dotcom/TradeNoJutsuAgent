"""Interactive Telegram command system for remote control of TradeNoJutsu Agent.

Provides 12+ slash commands for monitoring, controlling, and tuning the agent
directly from Telegram. Uses python-telegram-bot with HTML message formatting.

Commands:
    /status      - Bot state, open positions, capital, regime
    /trades      - Recent closed trades with PnL
    /stop        - Pause the signal loop
    /resume      - Resume the signal loop
    /halt        - Emergency: close ALL positions + stop
    /risk X      - Set risk percentage dynamically
    /pnl         - Today/week/all-time PnL breakdown
    /performance - Full performance metrics (win rate, sharpe, PF, DD)
    /regime      - Current market regime for each symbol
    /threshold X - View/set signal entry threshold
    /learned     - Last learning cycle results
    /help        - List all commands
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from tradenojutsu.infra.logger import get_logger

logger = get_logger("telegram.commands")


class TelegramCommands:
    """Handles interactive Telegram commands for the trading agent.

    Accepts an optional ``agent`` reference (a ``TradeNoJutsuAgent`` instance).
    When *agent* is ``None`` the commands that require live agent state
    (/stop, /resume, /halt, /regime) will return graceful "not connected"
    messages; database-backed commands (/trades, /pnl, /performance, etc.)
    still work.
    """

    def __init__(self, bot_token: str, chat_id: str, agent: Any = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.agent = agent
        self._app: Application | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_application(self) -> Application:
        """Build and return the ``Application`` with all command handlers registered."""
        app = Application.builder().token(self.bot_token).build()

        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("trades", self.cmd_trades))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CommandHandler("resume", self.cmd_resume))
        app.add_handler(CommandHandler("halt", self.cmd_halt))
        app.add_handler(CommandHandler("risk", self.cmd_risk))
        app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        app.add_handler(CommandHandler("performance", self.cmd_performance))
        app.add_handler(CommandHandler("regime", self.cmd_regime))
        app.add_handler(CommandHandler("threshold", self.cmd_threshold))
        app.add_handler(CommandHandler("learned", self.cmd_learned))
        app.add_handler(CommandHandler("help", self.cmd_help))

        self._app = app
        return app

    async def start_polling(self) -> None:
        """Build the application and start polling for commands."""
        app = self.build_application()
        logger.info("Telegram command bot starting polling...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram command bot is now listening")

    async def stop_polling(self) -> None:
        """Gracefully stop the polling loop."""
        if self._app is not None:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram command bot stopped")

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _authorized(self, update: Update) -> bool:
        """Only allow messages from the configured chat_id."""
        return str(update.effective_chat.id) == str(self.chat_id)

    async def _deny(self, update: Update) -> None:
        await update.message.reply_html("<b>Unauthorized.</b>")

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Bot state, open positions, capital, and regime."""
        if not self._authorized(update):
            return await self._deny(update)

        from tradenojutsu.infra.database import get_open_trades

        open_trades = get_open_trades()

        # Agent state
        if self.agent is not None:
            running = getattr(self.agent, "_running", False)
            mode = getattr(self.agent, "mode", "unknown")
            capital = getattr(self.agent.risk_manager, "capital", 0)
            symbols = getattr(self.agent, "symbols", [])
            state_label = "RUNNING" if running else "PAUSED"
        else:
            running = None
            mode = "N/A"
            capital = 0
            symbols = []
            state_label = "DISCONNECTED"

        lines = [
            f"<b>STATUS</b>",
            "",
            f"State: <code>{state_label}</code>",
            f"Mode: <code>{mode}</code>",
        ]

        if capital:
            lines.append(f"Capital: <code>${capital:,.2f}</code>")
        if symbols:
            lines.append(f"Symbols: <code>{', '.join(symbols)}</code>")

        lines.append(f"\n<b>Open Positions ({len(open_trades)})</b>")

        if not open_trades:
            lines.append("<i>No open positions</i>")
        else:
            for t in open_trades:
                direction = t["direction"].upper()
                emoji = "🟢" if t["direction"] == "long" else "🔴"
                lines.append(
                    f"{emoji} <b>{direction}</b> {t['symbol']} "
                    f"@ <code>{t['entry_price']:.4f}</code> | "
                    f"SL: <code>{t['stop_loss']:.4f}</code> | "
                    f"TP: <code>{t['take_profit']:.4f}</code> | "
                    f"Qty: <code>{t['quantity']:.4f}</code>"
                )

        await update.message.reply_html("\n".join(lines))

    # ------------------------------------------------------------------
    # /trades
    # ------------------------------------------------------------------

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Recent closed trades with PnL."""
        if not self._authorized(update):
            return await self._deny(update)

        from tradenojutsu.infra.database import get_recent_trades

        # Optional count argument: /trades 20
        args = context.args
        n = 10
        if args:
            try:
                n = max(1, min(int(args[0]), 50))
            except (ValueError, IndexError):
                pass

        trades = get_recent_trades(n)

        if not trades:
            await update.message.reply_html("<b>RECENT TRADES</b>\n\n<i>No closed trades yet.</i>")
            return

        lines = [f"<b>RECENT TRADES</b> (last {len(trades)})\n"]
        total_pnl = 0.0
        for t in trades:
            pnl = t.get("pnl") or 0
            total_pnl += pnl
            pnl_pct = t.get("pnl_pct") or 0
            emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
            lines.append(
                f"{emoji} {t['direction'].upper()} {t['symbol']} | "
                f"PnL: <code>{pnl:+.2f}</code> ({pnl_pct:+.1f}%) | "
                f"{t.get('strategy', '?')}"
            )

        lines.append(f"\nTotal: <b>{total_pnl:+.2f}</b>")
        await update.message.reply_html("\n".join(lines))

    # ------------------------------------------------------------------
    # /stop
    # ------------------------------------------------------------------

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Pause the signal loop (sets ``_running = False``)."""
        if not self._authorized(update):
            return await self._deny(update)

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        self.agent._running = False
        logger.info("Agent paused via /stop command")
        await update.message.reply_html(
            "⏸ <b>Agent paused.</b>\n\n"
            "The signal loop has been stopped. Open positions remain active.\n"
            "Use /resume to restart."
        )

    # ------------------------------------------------------------------
    # /resume
    # ------------------------------------------------------------------

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Resume the signal loop."""
        if not self._authorized(update):
            return await self._deny(update)

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        was_running = getattr(self.agent, "_running", False)
        self.agent._running = True
        logger.info("Agent resumed via /resume command")
        if was_running:
            await update.message.reply_html("▶ <b>Agent was already running.</b>")
        else:
            await update.message.reply_html(
                "▶ <b>Agent resumed.</b>\n\n"
                "The signal loop is now active again."
            )

    # ------------------------------------------------------------------
    # /halt
    # ------------------------------------------------------------------

    async def cmd_halt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Emergency: close ALL paper positions and stop the agent."""
        if not self._authorized(update):
            return await self._deny(update)

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        # Stop the loop first
        self.agent._running = False
        logger.warning("HALT command received - closing all positions")

        # Close all open positions via the paper trader
        paper_trader = getattr(self.agent, "paper_trader", None)
        closed_count = 0
        closed_pnl = 0.0

        if paper_trader is not None:
            for trade_id, trade in list(paper_trader.open_positions.items()):
                # Use entry_price as emergency exit price (best we can do
                # without a live price feed in the command handler)
                exit_price = trade.entry_price
                closed_trade = paper_trader.force_close(
                    trade_id, exit_price, reason="HALT - emergency close"
                )
                if closed_trade is not None:
                    closed_count += 1
                    closed_pnl += closed_trade.pnl

        await update.message.reply_html(
            "🛑 <b>EMERGENCY HALT</b>\n\n"
            f"Agent stopped.\n"
            f"Positions closed: <code>{closed_count}</code>\n"
            f"Realized PnL: <code>{closed_pnl:+.2f}</code>\n\n"
            "<i>Use /resume to restart the agent.</i>"
        )

    # ------------------------------------------------------------------
    # /risk X
    # ------------------------------------------------------------------

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """View or set risk percentage.  Usage: ``/risk`` or ``/risk 1.5``."""
        if not self._authorized(update):
            return await self._deny(update)

        args = context.args

        # View current
        if not args:
            if self.agent is not None:
                rm = self.agent.risk_manager
                await update.message.reply_html(
                    "<b>RISK SETTINGS</b>\n\n"
                    f"Risk per trade: <code>{rm.risk_per_trade_pct:.2f}%</code>\n"
                    f"Max risk per trade: <code>{rm.risk_per_trade_max:.2f}%</code>\n"
                    f"Max concurrent: <code>{rm.max_concurrent}</code>\n"
                    f"Max daily trades: <code>{rm.max_daily_trades}</code>\n"
                    f"Daily DD limit: <code>{rm.daily_drawdown_max_pct:.1f}%</code>\n"
                    f"Portfolio risk limit: <code>{rm.max_portfolio_risk_pct:.1f}%</code>"
                )
            else:
                await update.message.reply_html("<b>No agent connected.</b>")
            return

        # Set new value
        try:
            new_risk = float(args[0])
        except ValueError:
            await update.message.reply_html("Usage: <code>/risk 1.5</code> (value in %)")
            return

        if not 0.1 <= new_risk <= 5.0:
            await update.message.reply_html(
                "Risk must be between <code>0.1%</code> and <code>5.0%</code>."
            )
            return

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        old_risk = self.agent.risk_manager.risk_per_trade_pct
        self.agent.risk_manager.risk_per_trade_pct = new_risk

        # Also update settings so it persists across restarts
        try:
            from tradenojutsu.config import load_settings, save_settings
            settings = load_settings()
            settings.setdefault("risk", {})["risk_per_trade_pct"] = new_risk
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to persist risk setting: {e}")

        logger.info(f"Risk changed via Telegram: {old_risk}% -> {new_risk}%")
        await update.message.reply_html(
            f"<b>Risk updated</b>\n\n"
            f"Old: <code>{old_risk:.2f}%</code>\n"
            f"New: <code>{new_risk:.2f}%</code>"
        )

    # ------------------------------------------------------------------
    # /pnl
    # ------------------------------------------------------------------

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Today / this week / all-time PnL breakdown."""
        if not self._authorized(update):
            return await self._deny(update)

        from tradenojutsu.infra.database import get_recent_trades

        trades = get_recent_trades(500)
        closed = [t for t in trades if t.get("pnl") is not None]

        if not closed:
            await update.message.reply_html(
                "<b>PnL BREAKDOWN</b>\n\n<i>No closed trades yet.</i>"
            )
            return

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())

        today_pnl = 0.0
        today_count = 0
        week_pnl = 0.0
        week_count = 0
        all_pnl = 0.0
        all_count = len(closed)

        for t in closed:
            pnl = t["pnl"]
            all_pnl += pnl
            exit_time_str = t.get("exit_time")
            if exit_time_str:
                try:
                    exit_time = datetime.fromisoformat(exit_time_str)
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.replace(tzinfo=timezone.utc)
                    if exit_time >= today_start:
                        today_pnl += pnl
                        today_count += 1
                    if exit_time >= week_start:
                        week_pnl += pnl
                        week_count += 1
                except (ValueError, TypeError):
                    pass

        def _fmt(val: float) -> str:
            emoji = "📈" if val > 0 else "📉" if val < 0 else "➖"
            return f"{emoji} <code>{val:+.2f}</code>"

        await update.message.reply_html(
            "<b>PnL BREAKDOWN</b>\n\n"
            f"<b>Today</b> ({today_count} trades): {_fmt(today_pnl)}\n"
            f"<b>This Week</b> ({week_count} trades): {_fmt(week_pnl)}\n"
            f"<b>All Time</b> ({all_count} trades): {_fmt(all_pnl)}"
        )

    # ------------------------------------------------------------------
    # /performance
    # ------------------------------------------------------------------

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Full performance metrics: win rate, Sharpe, profit factor, drawdown."""
        if not self._authorized(update):
            return await self._deny(update)

        from tradenojutsu.infra.database import get_recent_trades

        trades = get_recent_trades(500)
        closed = [t for t in trades if t.get("pnl") is not None]

        if not closed:
            await update.message.reply_html(
                "<b>PERFORMANCE</b>\n\n<i>No closed trades yet.</i>"
            )
            return

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(closed)
        win_rate = len(wins) / total * 100 if total else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_pnl = sum(pnls)
        expectancy = total_pnl / total if total else 0

        # Sharpe ratio (annualised, assuming ~252 trading days)
        import statistics
        if len(pnls) >= 2:
            mean_r = statistics.mean(pnls)
            std_r = statistics.stdev(pnls)
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls[::-1]:  # oldest first (reverse since DB returns newest first)
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0

        await update.message.reply_html(
            "<b>PERFORMANCE METRICS</b>\n\n"
            f"Total Trades: <code>{total}</code>\n"
            f"Win Rate: <code>{win_rate:.1f}%</code> ({len(wins)}W / {len(losses)}L)\n"
            f"Profit Factor: <code>{profit_factor:.2f}</code>\n"
            f"Sharpe Ratio: <code>{sharpe:.2f}</code>\n"
            f"Max Drawdown: <code>{max_dd:.2f}</code> ({max_dd_pct:.1f}%)\n"
            f"Total PnL: <code>{total_pnl:+.2f}</code>\n\n"
            f"Avg Win: <code>{avg_win:+.2f}</code>\n"
            f"Avg Loss: <code>{avg_loss:+.2f}</code>\n"
            f"Expectancy: <code>{expectancy:+.2f}</code>/trade\n"
            f"Best: <code>{max(pnls):+.2f}</code>\n"
            f"Worst: <code>{min(pnls):+.2f}</code>"
        )

    # ------------------------------------------------------------------
    # /regime
    # ------------------------------------------------------------------

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Current market regime for each symbol."""
        if not self._authorized(update):
            return await self._deny(update)

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        symbols = getattr(self.agent, "symbols", [])
        timeframes = getattr(self.agent, "timeframes", ["15m"])
        analyzer = getattr(self.agent, "analyzer", None)
        data_fetcher = getattr(self.agent, "data_fetcher", None)

        if not symbols or analyzer is None or data_fetcher is None:
            await update.message.reply_html("<b>Agent not fully initialised.</b>")
            return

        lines = ["<b>MARKET REGIME</b>\n"]
        tf = timeframes[0] if timeframes else "15m"

        for symbol in symbols:
            try:
                df = data_fetcher.fetch_ohlcv(symbol, tf, lookback_days=30)
                if df.empty:
                    lines.append(f"  {symbol}: <i>no data</i>")
                    continue
                df = analyzer.compute_indicators(df)
                state = analyzer.build_market_state(symbol, df)
                regime_emoji = {
                    "trending_up": "📈",
                    "trending_down": "📉",
                    "ranging": "↔",
                    "high_volatility": "⚡",
                    "low_volatility": "😴",
                }.get(state.regime.value, "❓")
                lines.append(
                    f"{regime_emoji} <b>{symbol}</b>: "
                    f"<code>{state.regime.value}</code> | "
                    f"Price: <code>{state.price:.4f}</code> | "
                    f"Vol: <code>{state.volatility:.2f}%</code>"
                )
            except Exception as e:
                lines.append(f"  {symbol}: <i>error - {e}</i>")

        await update.message.reply_html("\n".join(lines))

    # ------------------------------------------------------------------
    # /threshold X
    # ------------------------------------------------------------------

    async def cmd_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """View or set signal entry threshold.  Usage: ``/threshold`` or ``/threshold 70``."""
        if not self._authorized(update):
            return await self._deny(update)

        args = context.args

        # View
        if not args:
            if self.agent is not None:
                thresh = getattr(self.agent.combiner, "entry_threshold", "?")
                await update.message.reply_html(
                    f"<b>ENTRY THRESHOLD</b>\n\n"
                    f"Current: <code>{thresh}</code> / 100"
                )
            else:
                await update.message.reply_html("<b>No agent connected.</b>")
            return

        # Set
        try:
            new_val = float(args[0])
        except ValueError:
            await update.message.reply_html("Usage: <code>/threshold 70</code> (0-100)")
            return

        if not 0 <= new_val <= 100:
            await update.message.reply_html("Threshold must be between <code>0</code> and <code>100</code>.")
            return

        if self.agent is None:
            await update.message.reply_html("<b>No agent connected.</b>")
            return

        old_val = getattr(self.agent.combiner, "entry_threshold", None)
        self.agent.combiner.entry_threshold = new_val

        # Persist
        try:
            from tradenojutsu.config import load_settings, save_settings
            settings = load_settings()
            settings.setdefault("signals", {})["entry_threshold"] = new_val
            save_settings(settings)
        except Exception as e:
            logger.error(f"Failed to persist threshold setting: {e}")

        logger.info(f"Entry threshold changed via Telegram: {old_val} -> {new_val}")
        await update.message.reply_html(
            f"<b>Threshold updated</b>\n\n"
            f"Old: <code>{old_val}</code>\n"
            f"New: <code>{new_val}</code>"
        )

    # ------------------------------------------------------------------
    # /learned
    # ------------------------------------------------------------------

    async def cmd_learned(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Last learning cycle results."""
        if not self._authorized(update):
            return await self._deny(update)

        from tradenojutsu.infra.database import get_connection

        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM learning_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()

        if not rows:
            await update.message.reply_html(
                "<b>LEARNING LOG</b>\n\n<i>No learning cycles recorded yet.</i>"
            )
            return

        # Group by cycle number
        latest_cycle = rows[0]["cycle_num"]
        cycle_rows = [dict(r) for r in rows if r["cycle_num"] == latest_cycle]

        lines = [f"<b>LEARNING CYCLE #{latest_cycle}</b>\n"]

        for entry in cycle_rows:
            param = entry["param_changed"]
            old_v = entry["old_value"]
            new_v = entry["new_value"]
            reason = entry.get("reason", "")
            lines.append(
                f"  <code>{param}</code>: {old_v} → {new_v}"
            )
            if reason:
                lines.append(f"    <i>{reason[:120]}</i>")

        # Show performance before/after if available
        first = cycle_rows[0]
        try:
            perf_before = json.loads(first.get("performance_before", "{}"))
            perf_after = json.loads(first.get("performance_after", "{}"))
            if perf_before or perf_after:
                lines.append("\n<b>Performance</b>")
                for key in ("win_rate", "profit_factor", "sharpe_ratio", "total_pnl"):
                    before = perf_before.get(key, "?")
                    after = perf_after.get(key, "?")
                    if before != "?" or after != "?":
                        lines.append(f"  {key}: {before} → {after}")
        except (json.JSONDecodeError, TypeError):
            pass

        lines.append(f"\n<i>Timestamp: {first.get('created_at', 'N/A')}</i>")
        await update.message.reply_html("\n".join(lines))

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all available commands."""
        if not self._authorized(update):
            return await self._deny(update)

        await update.message.reply_html(
            "<b>TradeNoJutsu Commands</b>\n\n"
            "/status — Bot state, open positions, capital, regime\n"
            "/trades [N] — Recent closed trades with PnL (default 10)\n"
            "/stop — Pause the signal loop\n"
            "/resume — Resume the signal loop\n"
            "/halt — Emergency: close ALL positions + stop\n"
            "/risk [X] — View or set risk % (e.g. /risk 1.5)\n"
            "/pnl — Today / week / all-time PnL breakdown\n"
            "/performance — Full metrics (win rate, Sharpe, PF, DD)\n"
            "/regime — Current market regime per symbol\n"
            "/threshold [X] — View or set entry threshold (0-100)\n"
            "/learned — Last learning cycle results\n"
            "/help — This message"
        )


# ------------------------------------------------------------------
# Convenience entry-point
# ------------------------------------------------------------------

async def start_bot(bot_token: str, chat_id: str, agent: Any = None) -> TelegramCommands:
    """Create a ``TelegramCommands`` instance and start polling.

    Returns the ``TelegramCommands`` object so the caller can later call
    ``stop_polling()`` or swap in a live ``agent`` reference.

    Usage::

        cmds = await start_bot(token, chat_id, agent=my_agent)
        # ... later ...
        await cmds.stop_polling()
    """
    cmds = TelegramCommands(bot_token, chat_id, agent=agent)
    await cmds.start_polling()
    return cmds
