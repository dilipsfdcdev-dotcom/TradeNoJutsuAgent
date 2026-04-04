"""MT5 order execution with retry logic and error handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import structlog

from agent.data.mt5_feed import _ensure_connected

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    price_filled: float | None
    slippage: float | None
    time: datetime | None
    error_code: int | None
    error_message: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(code: int, message: str) -> OrderResult:
    """Shorthand for a failed OrderResult."""
    return OrderResult(
        success=False,
        ticket=None,
        price_filled=None,
        slippage=None,
        time=None,
        error_code=code,
        error_message=message,
    )


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

def send_market_order(
    symbol: str,
    direction: str,  # "buy" or "sell"
    lot: float,
    sl: float,
    tp: float,
    comment: str = "",
    magic: int = 0,
    max_retries: int = 3,
) -> OrderResult:
    """Send a market order to MT5 with retry on requote."""
    _ensure_connected()

    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL

    for attempt in range(max_retries):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return _fail(-1, "Failed to get tick")

        price = tick.ask if direction == "buy" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            logger.error("order_send_null", symbol=symbol, attempt=attempt)
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            slippage = abs(result.price - price) if result.price else None
            logger.info(
                "order_filled",
                ticket=result.order,
                price=result.price,
                slippage=slippage,
            )
            return OrderResult(
                success=True,
                ticket=result.order,
                price_filled=result.price,
                slippage=slippage,
                time=datetime.now(tz=timezone.utc),
                error_code=None,
                error_message=None,
            )

        if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            logger.warning("requote", symbol=symbol, attempt=attempt)
            time.sleep(0.1)
            continue

        logger.error("order_failed", retcode=result.retcode, comment=result.comment)
        return _fail(result.retcode, result.comment)

    return _fail(-1, "Max retries exceeded on requote")


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------

def send_limit_order(
    symbol: str,
    direction: str,  # "buy" or "sell"
    lot: float,
    price: float,
    sl: float,
    tp: float,
    comment: str = "",
    magic: int = 0,
    max_retries: int = 3,
) -> OrderResult:
    """Place a pending limit order on MT5 with retry on transient errors."""
    _ensure_connected()

    order_type = (
        mt5.ORDER_TYPE_BUY_LIMIT if direction == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
    )

    for attempt in range(max_retries):
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            logger.error("limit_order_send_null", symbol=symbol, attempt=attempt)
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("limit_order_placed", ticket=result.order, price=price)
            return OrderResult(
                success=True,
                ticket=result.order,
                price_filled=price,
                slippage=None,
                time=datetime.now(tz=timezone.utc),
                error_code=None,
                error_message=None,
            )

        if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            logger.warning("limit_requote", symbol=symbol, attempt=attempt)
            time.sleep(0.1)
            continue

        logger.error(
            "limit_order_failed", retcode=result.retcode, comment=result.comment,
        )
        return _fail(result.retcode, result.comment)

    return _fail(-1, "Max retries exceeded for limit order")


# ---------------------------------------------------------------------------
# Modify SL / TP
# ---------------------------------------------------------------------------

def modify_order(
    ticket: int,
    new_sl: float | None = None,
    new_tp: float | None = None,
) -> bool:
    """Modify SL and/or TP of an open position identified by *ticket*.

    Returns True on success, False otherwise.
    """
    _ensure_connected()

    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error("modify_position_not_found", ticket=ticket)
        return False

    pos = position[0]
    sl = new_sl if new_sl is not None else pos.sl
    tp = new_tp if new_tp is not None else pos.tp

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": sl,
        "tp": tp,
    }

    result = mt5.order_send(request)

    if result is None:
        logger.error("modify_order_send_null", ticket=ticket)
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("position_modified", ticket=ticket, sl=sl, tp=tp)
        return True

    logger.error(
        "modify_failed", ticket=ticket, retcode=result.retcode, comment=result.comment,
    )
    return False


# ---------------------------------------------------------------------------
# Close position
# ---------------------------------------------------------------------------

def close_position(ticket: int, lot: float | None = None) -> bool:
    """Close a position fully or partially.

    Parameters
    ----------
    ticket : int
        The position ticket to close.
    lot : float, optional
        Volume to close.  If *None*, the entire position is closed.

    Returns True on success.
    """
    _ensure_connected()

    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error("close_position_not_found", ticket=ticket)
        return False

    pos = position[0]
    volume = lot if lot is not None else pos.volume

    # Opposite direction to close
    close_type = (
        mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    )

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        logger.error("close_tick_failed", symbol=pos.symbol)
        return False

    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": pos.magic,
        "comment": f"close #{ticket}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        logger.error("close_order_send_null", ticket=ticket)
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("position_closed", ticket=ticket, volume=volume, price=result.price)
        return True

    logger.error(
        "close_failed", ticket=ticket, retcode=result.retcode, comment=result.comment,
    )
    return False


# ---------------------------------------------------------------------------
# Close all positions
# ---------------------------------------------------------------------------

def close_all_positions(symbol: str | None = None) -> int:
    """Close all open positions, optionally filtered by *symbol*.

    Returns the number of positions successfully closed.
    """
    _ensure_connected()

    positions = (
        mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    )

    if not positions:
        logger.info("no_positions_to_close", symbol=symbol)
        return 0

    closed = 0
    for pos in positions:
        if close_position(pos.ticket):
            closed += 1

    logger.info("close_all_done", requested=len(positions), closed=closed, symbol=symbol)
    return closed


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_open_positions(symbol: str | None = None) -> list[dict]:
    """Return open positions as a list of dicts.

    Each dict contains: ticket, symbol, direction, volume, price_open,
    sl, tp, profit, time, comment.
    """
    _ensure_connected()

    positions = (
        mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    )

    if not positions:
        return []

    results: list[dict] = []
    for pos in positions:
        results.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "direction": "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume": pos.volume,
            "price_open": pos.price_open,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
            "comment": pos.comment,
        })

    logger.info("open_positions_fetched", count=len(results), symbol=symbol)
    return results


def get_order_history(days: int = 7) -> list[dict]:
    """Return closed trades from the last *days* days.

    Each dict contains: ticket, symbol, direction, volume, price_open,
    price_close, profit, time_open, time_close, comment.
    """
    _ensure_connected()

    now = datetime.now(tz=timezone.utc)
    from_date = now - timedelta(days=days)

    deals = mt5.history_deals_get(from_date, now)
    if not deals:
        logger.info("no_deal_history", days=days)
        return []

    # Filter to entry+exit deals (type IN/OUT) and pair them by position id.
    # We focus on "out" deals which represent closed trades and carry the
    # profit information.
    results: list[dict] = []
    for deal in deals:
        # DEAL_ENTRY_OUT == 1 means the deal closed (or partially closed) a position
        if deal.entry != 1:
            continue

        results.append({
            "ticket": deal.position_id,
            "symbol": deal.symbol,
            "direction": "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell",
            "volume": deal.volume,
            "price_open": deal.price,   # close-side price
            "price_close": deal.price,
            "profit": deal.profit,
            "time_open": datetime.fromtimestamp(deal.time, tz=timezone.utc),
            "time_close": datetime.fromtimestamp(deal.time, tz=timezone.utc),
            "comment": deal.comment,
        })

    logger.info("order_history_fetched", count=len(results), days=days)
    return results
