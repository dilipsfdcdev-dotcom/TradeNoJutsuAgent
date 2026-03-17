"""
MT5 connection manager for the XAUUSD trading agent.

Handles initialization, login, reconnection, and symbol info retrieval
against the MetaTrader 5 terminal.
"""

from __future__ import annotations

import os
import time
from typing import Any

import MetaTrader5 as mt5

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_MAX_CONNECT_RETRIES = 3
_RETRY_DELAY_S = 5


class MT5Connector:
    """Manages the lifecycle of a MetaTrader 5 terminal connection."""

    def __init__(self) -> None:
        self.connected: bool = False
        self.login: int = int(os.getenv("MT5_LOGIN", "0"))
        self.password: str = os.getenv("MT5_PASSWORD", "")
        self.server: str = os.getenv("MT5_SERVER", "")
        self.path: str = os.getenv("MT5_PATH", "")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize the MT5 terminal and log in.

        Retries up to ``_MAX_CONNECT_RETRIES`` times with a
        ``_RETRY_DELAY_S`` second pause between attempts.

        Returns:
            ``True`` if the connection and login succeeded.
        """
        for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
            logger.info(
                "MT5 connection attempt %d/%d",
                attempt,
                _MAX_CONNECT_RETRIES,
                extra={"server": self.server, "login": self.login},
            )

            # Shut down any stale session before re-initialising.
            mt5.shutdown()

            init_kwargs: dict[str, Any] = {}
            if self.path:
                init_kwargs["path"] = self.path

            if not mt5.initialize(**init_kwargs):
                error = mt5.last_error()
                logger.error(
                    "MT5 initialize failed (attempt %d): %s",
                    attempt,
                    error,
                )
                if attempt < _MAX_CONNECT_RETRIES:
                    time.sleep(_RETRY_DELAY_S)
                continue

            if self.login:
                authorised = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                )
                if not authorised:
                    error = mt5.last_error()
                    logger.error(
                        "MT5 login failed (attempt %d): %s",
                        attempt,
                        error,
                    )
                    mt5.shutdown()
                    if attempt < _MAX_CONNECT_RETRIES:
                        time.sleep(_RETRY_DELAY_S)
                    continue

            self.connected = True
            terminal = mt5.terminal_info()
            logger.info(
                "MT5 connected successfully",
                extra={
                    "company": getattr(terminal, "company", ""),
                    "build": getattr(terminal, "build", ""),
                    "trade_allowed": getattr(terminal, "trade_allowed", None),
                },
            )
            return True

        self.connected = False
        logger.critical(
            "MT5 connection failed after %d attempts", _MAX_CONNECT_RETRIES
        )
        return False

    def ensure_connected(self) -> bool:
        """Verify the terminal is reachable; reconnect if necessary.

        Returns:
            ``True`` if a live connection is available after the check.
        """
        info = mt5.terminal_info()
        if info is not None and getattr(info, "connected", False):
            self.connected = True
            return True

        logger.warning("MT5 terminal not responsive — attempting reconnect")
        self.connected = False
        return self.connect()

    def disconnect(self) -> None:
        """Shut down the MT5 terminal connection."""
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def get_symbol_info(self, symbol: str = "XAUUSD") -> dict[str, Any] | None:
        """Return key trading parameters for *symbol*.

        Returns:
            A dict with ``trade_tick_value``, ``trade_tick_size``,
            ``volume_min``, ``volume_max``, ``volume_step``, ``point``,
            and ``digits``, or ``None`` on failure.
        """
        if not self.ensure_connected():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(
                "Failed to retrieve symbol info for %s: %s",
                symbol,
                mt5.last_error(),
            )
            # Attempt to enable the symbol in MarketWatch and retry.
            if not mt5.symbol_select(symbol, True):
                logger.error("Could not select symbol %s in MarketWatch", symbol)
                return None
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

        return {
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "point": info.point,
            "digits": info.digits,
        }
