"""
Telegram bot entry point for the XAUUSD trading agent.

Builds the ``python-telegram-bot`` v20 :class:`Application`, registers all
command handlers, starts long-polling, and exposes :meth:`send_message` so that
the main trading loop can push formatted alerts to the operator.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from xauusd_agent.infra.logger import get_logger
from xauusd_agent.telegram_bot.alerts import TelegramAlerts
from xauusd_agent.telegram_bot.commands import TelegramCommands

if TYPE_CHECKING:
    import asyncpg

logger = get_logger(__name__)

# Telegram enforces a 4 096-char UTF-8 limit per message.
_MAX_MSG_LEN = 4096


class TradingBot:
    """Async Telegram bot for monitoring and controlling the XAUUSD agent.

    Parameters
    ----------
    db_pool : asyncpg.Pool
        Shared database connection pool.
    executor : object, optional
        Trade executor instance (used by ``/halt`` to close all positions).
    """

    def __init__(self, db_pool: asyncpg.Pool, executor=None) -> None:
        self.token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
        self.db: asyncpg.Pool = db_pool
        self.executor = executor
        self.app: Application | None = None
        self.alerts = TelegramAlerts(self)
        self.commands = TelegramCommands(db_pool, executor)
        self._started: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialize bot application and register all command handlers."""
        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN is not set; bot will not start")
            return

        self.app = Application.builder().token(self.token).build()

        # Shared data accessible from every handler via context.bot_data
        self.app.bot_data["db"] = self.db
        self.app.bot_data["chat_id"] = self.chat_id

        # ---- Register command handlers --------------------------------
        _HANDLERS = {
            "status": self.commands.cmd_status,
            "trades": self.commands.cmd_trades,
            "stop": self.commands.cmd_stop,
            "resume": self.commands.cmd_resume,
            "halt": self.commands.cmd_halt,
            "risk": self.commands.cmd_risk,
            "pnl": self.commands.cmd_pnl,
            "summary": self.commands.cmd_summary,
            "bias": self.commands.cmd_bias,
            "regime": self.commands.cmd_regime,
            "threshold": self.commands.cmd_threshold,
            "learned": self.commands.cmd_learned,
            "scores": self.commands.cmd_scores,
        }

        for name, callback in _HANDLERS.items():
            self.app.add_handler(CommandHandler(name, callback))

        # Global error handler
        self.app.add_error_handler(_error_handler)

        # ---- Set the bot-command menu (visible in Telegram UI) --------
        menu_commands = [
            BotCommand("status", "Bot state, balance, regime"),
            BotCommand("trades", "Open trades with P&L"),
            BotCommand("stop", "Pause signal loop (ratchet stays active)"),
            BotCommand("resume", "Resume signal loop"),
            BotCommand("halt", "EMERGENCY: close all + stop"),
            BotCommand("risk", "Set risk % (e.g. /risk 0.8)"),
            BotCommand("pnl", "P&L: today / week / month"),
            BotCommand("summary", "Force daily summary now"),
            BotCommand("bias", "HTF bias: D1/H4/H1 scores"),
            BotCommand("regime", "Market regime + ATR ratio"),
            BotCommand("threshold", "Current entry threshold"),
            BotCommand("learned", "Last learning cycle details"),
            BotCommand("scores", "Last signal component scores"),
        ]

        try:
            await self.app.initialize()
            await self.app.bot.set_my_commands(menu_commands)
            logger.info(
                "Telegram bot initialised with %d commands", len(_HANDLERS)
            )
        except Exception:
            logger.exception("Failed to initialise bot / set command menu")

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a message to the configured ``TELEGRAM_CHAT_ID``.

        Handles missing configuration, oversized messages, and network errors
        gracefully so that a Telegram failure never crashes the trading loop.

        Returns ``True`` on success, ``False`` on failure (never raises).
        """
        if not self.app or not self.chat_id:
            logger.warning(
                "Cannot send Telegram message: bot or chat_id not configured"
            )
            return False

        try:
            for chunk in _split_message(text):
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                )
            return True
        except Exception:
            logger.exception("Failed to send Telegram message")
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """Start the bot polling loop.

        This method blocks until :meth:`stop` is called.  It is designed to
        run as a concurrent ``asyncio`` task alongside the main trading loop.
        """
        if self.app is None:
            logger.error("Bot not set up; call setup() first")
            return

        if self._started:
            logger.warning("start_polling() called but already running")
            return

        logger.info("Starting Telegram bot polling")
        try:
            await self.app.start()
            await self.app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )
            self._started = True
            logger.info("Telegram bot is now polling for updates")

            # Block until stop() signals us to exit.
            self._stop_event = asyncio.Event()
            await self._stop_event.wait()
        except Exception:
            logger.exception("Telegram polling encountered an error")
        finally:
            self._started = False
            logger.info("Telegram bot polling ended")

    async def stop(self) -> None:
        """Gracefully stop the bot."""
        if self.app is None:
            return

        logger.info("Stopping Telegram bot")
        try:
            # Signal the polling loop to exit
            if hasattr(self, "_stop_event"):
                self._stop_event.set()

            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped gracefully")
        except Exception:
            logger.exception("Error during Telegram bot shutdown")
        finally:
            self._started = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_message(text: str) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= _MAX_MSG_LEN:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= _MAX_MSG_LEN:
            chunks.append(text)
            break

        # Prefer splitting on a newline boundary
        split_at = text.rfind("\n", 0, _MAX_MSG_LEN)
        if split_at == -1:
            split_at = _MAX_MSG_LEN

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


async def _error_handler(
    update: object | None,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Log exceptions raised inside handler callbacks."""
    logger.exception("Telegram handler error", exc_info=context.error)
