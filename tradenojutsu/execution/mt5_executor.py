"""MetaTrader 5 connector and trade executor.

Combines connection management (login, reconnect, symbol info) and
order execution (open, modify, close) in a single module.

The ``import MetaTrader5`` is guarded behind ``try/except`` so the
rest of the codebase can be imported and tested on machines where the
MT5 terminal is not installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tradenojutsu.data.models import Direction
from tradenojutsu.infra.logger import get_logger

logger = get_logger("execution.mt5")

# ── Guarded MT5 import ─────────────────────────────────────────────
try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]

    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False
    logger.debug("MetaTrader5 package not installed - MT5 features disabled")


def _require_mt5() -> None:
    """Raise early if the MetaTrader5 package is missing."""
    if not _MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 Python package is not installed.  "
            "Install it with:  pip install MetaTrader5"
        )


# ── MT5 Connector ─────────────────────────────────────────────────

@dataclass
class MT5Connector:
    """Manages the connection to the MetaTrader 5 terminal.

    Parameters
    ----------
    login : int
        MT5 account number.
    password : str
        Account password.
    server : str
        Broker server name.
    path : str | None
        Path to the MT5 terminal executable (optional).
    timeout : int
        Connection timeout in milliseconds.
    max_retries : int
        How many times ``connect`` will retry on failure.
    retry_delay : float
        Seconds to wait between retries.
    """

    login: int = 0
    password: str = ""
    server: str = ""
    path: str | None = None
    timeout: int = 10_000
    max_retries: int = 3
    retry_delay: float = 2.0
    _connected: bool = field(default=False, init=False, repr=False)

    # ── lifecycle ──────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise MT5 and log in.  Retries up to *max_retries* times."""
        _require_mt5()

        for attempt in range(1, self.max_retries + 1):
            init_kwargs: dict[str, Any] = {"login": self.login, "timeout": self.timeout}
            if self.path:
                init_kwargs["path"] = self.path

            if not mt5.initialize(**init_kwargs):
                logger.warning(
                    f"MT5 initialize failed (attempt {attempt}/{self.max_retries}): "
                    f"{mt5.last_error()}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

            authorised = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
            )
            if authorised:
                self._connected = True
                info = mt5.account_info()
                logger.info(
                    f"MT5 connected: account={self.login} "
                    f"server={self.server} "
                    f"balance={getattr(info, 'balance', '?')}"
                )
                return True

            logger.warning(
                f"MT5 login failed (attempt {attempt}/{self.max_retries}): "
                f"{mt5.last_error()}"
            )
            mt5.shutdown()
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        self._connected = False
        logger.error("MT5 connection failed after all retries")
        return False

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        if _MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def reconnect(self) -> bool:
        """Disconnect then reconnect."""
        self.disconnect()
        return self.connect()

    @property
    def is_connected(self) -> bool:
        if not _MT5_AVAILABLE or not self._connected:
            return False
        info = mt5.terminal_info()
        return info is not None and getattr(info, "connected", False)

    def ensure_connected(self) -> bool:
        """Reconnect if the connection dropped.  Returns True if connected."""
        if self.is_connected:
            return True
        logger.warning("MT5 connection lost - reconnecting")
        return self.reconnect()

    # ── symbol helpers ─────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Any | None:
        """Return MT5 symbol info, enabling the symbol if needed."""
        _require_mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            # Try selecting / enabling the symbol first
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not found or could not be enabled")
                return None
            info = mt5.symbol_info(symbol)
        return info

    def get_tick(self, symbol: str) -> Any | None:
        """Return the latest tick for *symbol*."""
        _require_mt5()
        return mt5.symbol_info_tick(symbol)

    def get_point(self, symbol: str) -> float:
        """Return the point size for *symbol* (e.g. 0.00001 for EURUSD)."""
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.00001  # sensible default for forex
        return info.point

    def get_digits(self, symbol: str) -> int:
        """Return the number of decimal digits for *symbol*."""
        info = self.get_symbol_info(symbol)
        if info is None:
            return 5
        return info.digits


# ── MT5 Executor ───────────────────────────────────────────────────

class MT5Executor:
    """Executes trades on MetaTrader 5 via the ``MT5Connector``.

    Provides a clean interface for opening, modifying, and closing
    positions without the caller needing to construct raw MT5 request
    dicts.
    """

    def __init__(self, connector: MT5Connector) -> None:
        self.connector = connector

    # ── helpers ────────────────────────────────────────────────────

    def _ensure(self) -> None:
        if not self.connector.ensure_connected():
            raise ConnectionError("Cannot establish MT5 connection")

    @staticmethod
    def _check_result(result: Any, action: str) -> bool:
        """Log and return True if the trade request succeeded."""
        if result is None:
            logger.error(f"{action}: result is None")
            return False
        retcode = getattr(result, "retcode", None)
        if retcode == mt5.TRADE_RETCODE_DONE:
            return True
        comment = getattr(result, "comment", "")
        logger.error(f"{action} failed: retcode={retcode}  comment={comment}")
        return False

    # ── open ──────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        direction: Direction,
        volume: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "TradeNoJutsu",
        magic: int = 123456,
        deviation: int = 20,
    ) -> int | None:
        """Send a market order.  Returns the ticket number or ``None``."""
        _require_mt5()
        self._ensure()

        tick = self.connector.get_tick(symbol)
        if tick is None:
            logger.error(f"Cannot get tick for {symbol}")
            return None

        if direction == Direction.LONG:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == Direction.SHORT:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            logger.error(f"Invalid direction for trade: {direction}")
            return None

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if self._check_result(result, f"open_trade {direction.value} {symbol}"):
            ticket = result.order
            logger.info(
                f"[MT5] Opened {direction.value} {symbol} vol={volume} "
                f"@ {price:.5f} SL={stop_loss:.5f} TP={take_profit:.5f} "
                f"ticket={ticket}"
            )
            return ticket
        return None

    # ── modify ────────────────────────────────────────────────────

    def modify_sl(
        self,
        ticket: int,
        new_sl: float,
        new_tp: float | None = None,
    ) -> bool:
        """Modify the stop-loss (and optionally TP) of an open position."""
        _require_mt5()
        self._ensure()

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"modify_sl: position {ticket} not found")
            return False
        pos = position[0]

        tp = new_tp if new_tp is not None else pos.tp

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": float(new_sl),
            "tp": float(tp),
        }

        result = mt5.order_send(request)
        return self._check_result(result, f"modify_sl ticket={ticket}")

    # ── close ─────────────────────────────────────────────────────

    def close_position(
        self,
        ticket: int,
        comment: str = "TradeNoJutsu close",
        deviation: int = 20,
    ) -> bool:
        """Close a single open position by ticket."""
        _require_mt5()
        self._ensure()

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"close_position: ticket {ticket} not found")
            return False
        pos = position[0]

        # Opposite order type to close
        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": deviation,
            "magic": pos.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if self._check_result(result, f"close_position ticket={ticket}"):
            logger.info(
                f"[MT5] Closed ticket={ticket} {pos.symbol} vol={pos.volume} @ {price:.5f}"
            )
            return True
        return False

    def close_all(self, symbol: str | None = None, comment: str = "TradeNoJutsu close all") -> int:
        """Close all open positions, optionally filtered by *symbol*.

        Returns the number of positions successfully closed.
        """
        _require_mt5()
        self._ensure()

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if self.close_position(pos.ticket, comment=comment):
                closed += 1
        return closed

    # ── query ─────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return open positions as a list of plain dicts.

        Each dict contains at least: ``ticket``, ``symbol``, ``type``,
        ``volume``, ``price_open``, ``price_current``, ``sl``, ``tp``,
        ``profit``, ``magic``, ``comment``.
        """
        _require_mt5()
        self._ensure()

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if not positions:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": p.type,
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "magic": p.magic,
                "comment": p.comment,
            }
            for p in positions
        ]
