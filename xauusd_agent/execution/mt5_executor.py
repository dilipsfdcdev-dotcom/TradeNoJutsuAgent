"""
MT5 order execution for the XAUUSD trading agent.

Provides helpers for opening, modifying, closing, and querying positions
through the MetaTrader 5 terminal.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5

from xauusd_agent.execution.mt5_connector import MT5Connector
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_ORDER_RETRY_DELAY_S = 1


class MT5Executor:
    """Thin wrapper around ``mt5.order_send`` with retry and logging."""

    def __init__(
        self,
        connector: MT5Connector,
        magic_number: int = 20250317,
    ) -> None:
        self.connector = connector
        self.magic = magic_number

    # ------------------------------------------------------------------
    # Open
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> dict[str, Any]:
        """Send a market order and return an execution report.

        Args:
            symbol:    Trading instrument (e.g. ``"XAUUSD"``).
            direction: ``"BUY"`` or ``"SELL"``.
            lot:       Position volume.
            sl:        Stop-loss price.
            tp:        Take-profit price.
            comment:   Free-text order comment.

        Returns:
            ``{success, ticket, price, retcode, comment}``
        """
        if not self.connector.ensure_connected():
            return self._fail_result("MT5 not connected")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return self._fail_result(f"No tick data for {symbol}")

        order_type = (
            mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if direction.upper() == "BUY" else tick.bid

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = self._send_order(request, symbol=symbol, direction=direction)
        if result["success"]:
            return result

        # Retry once with FOK filling if IOC was rejected.
        logger.warning(
            "IOC fill rejected (retcode=%s), retrying with FOK",
            result["retcode"],
        )
        time.sleep(_ORDER_RETRY_DELAY_S)

        # Refresh price for the retry.
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            request["price"] = tick.ask if direction.upper() == "BUY" else tick.bid
        request["type_filling"] = mt5.ORDER_FILLING_FOK

        return self._send_order(request, symbol=symbol, direction=direction)

    # ------------------------------------------------------------------
    # Modify SL
    # ------------------------------------------------------------------

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Move the stop-loss of an open position, preserving the TP.

        Returns:
            ``True`` if the modification succeeded.
        """
        if not self.connector.ensure_connected():
            return False

        position = self._get_position_by_ticket(ticket)
        if position is None:
            logger.error("Position %d not found for SL modification", ticket)
            return False

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": float(new_sl),
            "tp": float(position.tp),
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(
                "modify_sl order_send returned None for ticket %d: %s",
                ticket,
                mt5.last_error(),
            )
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "modify_sl failed for ticket %d: retcode=%d comment=%s",
                ticket,
                result.retcode,
                result.comment,
            )
            return False

        logger.info(
            "SL modified for ticket %d: new_sl=%.5f",
            ticket,
            new_sl,
            extra={"ticket": ticket, "new_sl": new_sl},
        )
        return True

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_position(self, ticket: int) -> dict[str, Any]:
        """Close a position by sending an opposite market order.

        Returns:
            ``{success, close_price, profit}``
        """
        if not self.connector.ensure_connected():
            return {"success": False, "close_price": 0.0, "profit": 0.0}

        position = self._get_position_by_ticket(ticket)
        if position is None:
            logger.error("Position %d not found for close", ticket)
            return {"success": False, "close_price": 0.0, "profit": 0.0}

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return {"success": False, "close_price": 0.0, "profit": 0.0}

        # Opposite direction to close.
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "price": close_price,
            "position": ticket,
            "deviation": 20,
            "magic": self.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(
                "close_position order_send returned None for ticket %d: %s",
                ticket,
                mt5.last_error(),
            )
            return {"success": False, "close_price": 0.0, "profit": 0.0}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "close_position failed for ticket %d: retcode=%d comment=%s",
                ticket,
                result.retcode,
                result.comment,
            )
            return {
                "success": False,
                "close_price": 0.0,
                "profit": position.profit,
            }

        logger.info(
            "Position %d closed at %.5f, profit=%.2f",
            ticket,
            result.price,
            position.profit,
            extra={"ticket": ticket, "close_price": result.price},
        )
        return {
            "success": True,
            "close_price": result.price,
            "profit": position.profit,
        }

    def close_all_positions(self, symbol: str = "XAUUSD") -> list[dict[str, Any]]:
        """Close every open position for *symbol*.

        Returns:
            A list of close-result dicts, one per position.
        """
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            logger.info("No open positions to close for %s", symbol)
            return []

        results: list[dict[str, Any]] = []
        for pos in positions:
            res = self.close_position(pos.ticket)
            res["ticket"] = pos.ticket
            results.append(res)

        closed = sum(1 for r in results if r["success"])
        logger.info(
            "close_all_positions: %d/%d closed for %s",
            closed,
            len(results),
            symbol,
        )
        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_open_positions(self, symbol: str = "XAUUSD") -> list[dict[str, Any]]:
        """Return a list of open-position dicts for *symbol*."""
        if not self.connector.ensure_connected():
            return []

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logger.warning(
                "positions_get returned None for %s: %s", symbol, mt5.last_error()
            )
            return []

        result: list[dict[str, Any]] = []
        for pos in positions:
            direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            result.append(
                {
                    "ticket": pos.ticket,
                    "direction": direction,
                    "lot_size": pos.volume,
                    "open_price": pos.price_open,
                    "current_sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "open_time": datetime.fromtimestamp(
                        pos.time, tz=timezone.utc
                    ).isoformat(),
                    "comment": pos.comment,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_position_by_ticket(ticket: int) -> Any | None:
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return None
        return positions[0]

    def _send_order(
        self,
        request: dict[str, Any],
        *,
        symbol: str,
        direction: str,
    ) -> dict[str, Any]:
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(
                "order_send returned None: %s",
                error,
                extra={"symbol": symbol, "direction": direction},
            )
            return self._fail_result(f"order_send returned None: {error}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "Order rejected: retcode=%d comment=%s",
                result.retcode,
                result.comment,
                extra={
                    "symbol": symbol,
                    "direction": direction,
                    "retcode": result.retcode,
                },
            )
            return {
                "success": False,
                "ticket": 0,
                "price": 0.0,
                "retcode": result.retcode,
                "comment": result.comment,
            }

        logger.info(
            "Order filled: %s %s %.2f lots @ %.5f — ticket %d",
            direction,
            symbol,
            request["volume"],
            result.price,
            result.order,
            extra={
                "ticket": result.order,
                "price": result.price,
                "retcode": result.retcode,
            },
        )
        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "retcode": result.retcode,
            "comment": result.comment,
        }

    @staticmethod
    def _fail_result(msg: str) -> dict[str, Any]:
        return {
            "success": False,
            "ticket": 0,
            "price": 0.0,
            "retcode": -1,
            "comment": msg,
        }
