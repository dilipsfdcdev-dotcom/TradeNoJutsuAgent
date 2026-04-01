"""Active management of open positions — trailing stops, partial closes, time exits.

Runs every tick (or every 5 seconds) via :func:`manage_positions`.

Trailing-stop progression:
    1. profit > 1.5x risk  -> move SL to breakeven + 1 pip
    2. profit > 2.0x risk  -> trail SL at 1x ATR behind price
    3. profit > 3.0x risk  -> close 50 %, trail remainder at 0.5x ATR

Partial-close:
    TP1 hit -> close 50 %, move SL to breakeven
    TP2 hit -> close remaining 50 %

Time-based (scalp):
    > 15 min without TP and in profit -> tighten SL to breakeven
    > 30 min                          -> close at market
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from agent.execution.mt5_executor import (
    close_position,
    get_open_positions,
    modify_order,
)
from agent.data.mt5_feed import get_tick
from agent.signals.indicators import compute_atr

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pip-size lookup (price movement that equals 1 pip)
# ---------------------------------------------------------------------------

_PIP_SIZES: dict[str, float] = {
    "XAUUSD": 0.10,    # gold: 1 pip = $0.10
    "XAGUSD": 0.010,   # silver: 1 pip = $0.010
    "BTCUSD": 1.0,     # bitcoin: 1 pip = $1.00
}

_DEFAULT_PIP_SIZE: float = 0.0001  # standard forex fallback


# ---------------------------------------------------------------------------
# Time thresholds (seconds)
# ---------------------------------------------------------------------------

_SCALP_TIGHTEN_SECONDS = 15 * 60   # 15 minutes
_SCALP_CLOSE_SECONDS = 30 * 60     # 30 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pip_size(symbol: str) -> float:
    """Return the pip size for *symbol*."""
    return _PIP_SIZES.get(symbol, _DEFAULT_PIP_SIZE)


def _profit_pips(symbol: str, entry_price: float, current_price: float, direction: str) -> float:
    """Calculate unrealised profit in pips.

    Parameters
    ----------
    symbol:
        Instrument name — used to look up pip size.
    entry_price:
        Position entry price.
    current_price:
        Current market price (bid for buys, ask for sells).
    direction:
        ``"buy"`` or ``"sell"``.

    Returns
    -------
    float
        Positive = in profit, negative = in loss.
    """
    pip = _pip_size(symbol)
    if direction == "buy":
        return (current_price - entry_price) / pip
    return (entry_price - current_price) / pip


def _risk_pips(symbol: str, entry_price: float, sl_price: float) -> float:
    """Return the initial risk distance in pips (always positive)."""
    pip = _pip_size(symbol)
    return abs(entry_price - sl_price) / pip


def _position_age_seconds(open_time: datetime) -> float:
    """Return how many seconds a position has been open."""
    now = datetime.now(timezone.utc)
    # Handle naive datetimes by assuming UTC
    if open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=timezone.utc)
    return (now - open_time).total_seconds()


def _current_price_for(tick: dict, direction: str) -> float | None:
    """Return bid for buys (close at bid) or ask for sells (close at ask).

    Returns ``None`` when the tick is empty.
    """
    if not tick:
        return None
    if direction == "buy":
        return tick.get("bid")
    return tick.get("ask")


# ---------------------------------------------------------------------------
# Core management actions
# ---------------------------------------------------------------------------

def move_to_breakeven(ticket: int, entry_price: float, direction: str) -> bool:
    """Move the stop-loss to breakeven + 1 pip in the position's favour.

    Returns ``True`` on success.
    """
    # We don't know the symbol here, so we accept a small approximation:
    # +1 pip on the safe side.  Callers should use the overload that takes
    # symbol when precision matters.
    return _move_to_breakeven(ticket, entry_price, direction, symbol=None)


def _move_to_breakeven(
    ticket: int,
    entry_price: float,
    direction: str,
    symbol: str | None,
) -> bool:
    pip = _pip_size(symbol) if symbol else 0.0001
    if direction == "buy":
        new_sl = entry_price + pip
    else:
        new_sl = entry_price - pip

    new_sl = round(new_sl, 5)
    ok = modify_order(ticket, sl=new_sl)
    if ok:
        log.info(
            "sl_moved_to_breakeven",
            ticket=ticket,
            new_sl=new_sl,
            direction=direction,
        )
    else:
        log.warning("breakeven_move_failed", ticket=ticket, new_sl=new_sl)
    return ok


def trail_stop(
    ticket: int,
    current_price: float,
    direction: str,
    atr: float,
    multiplier: float = 1.0,
) -> bool:
    """Trail the stop-loss at *multiplier* x ATR behind the current price.

    Parameters
    ----------
    ticket:
        MT5 position ticket.
    current_price:
        Latest market price for the position.
    direction:
        ``"buy"`` or ``"sell"``.
    atr:
        Current ATR value (in price, not pips).
    multiplier:
        How many ATRs behind price to set the stop.

    Returns ``True`` on success.
    """
    trail_distance = atr * multiplier
    if direction == "buy":
        new_sl = round(current_price - trail_distance, 5)
    else:
        new_sl = round(current_price + trail_distance, 5)

    ok = modify_order(ticket, sl=new_sl)
    if ok:
        log.info(
            "sl_trailed",
            ticket=ticket,
            new_sl=new_sl,
            atr=atr,
            multiplier=multiplier,
            direction=direction,
        )
    else:
        log.warning("trail_stop_failed", ticket=ticket, new_sl=new_sl)
    return ok


def partial_close(ticket: int, percent: float) -> bool:
    """Close *percent* (0-100) of a position's volume.

    Returns ``True`` on success.
    """
    if not 0 < percent <= 100:
        log.error("invalid_partial_close_percent", percent=percent, ticket=ticket)
        return False

    ok = close_position(ticket, percent=percent)
    if ok:
        log.info("partial_close_ok", ticket=ticket, percent=percent)
    else:
        log.warning("partial_close_failed", ticket=ticket, percent=percent)
    return ok


# ---------------------------------------------------------------------------
# Time-based exit logic
# ---------------------------------------------------------------------------

def check_time_exits() -> list[int]:
    """Check all open positions for time-based exit conditions.

    Scalp rules:
    - > 15 min in profit without TP: tighten SL to breakeven
    - > 30 min: close at market (scalp premise has failed)

    Returns
    -------
    list[int]
        Tickets of positions that were **closed** due to time.
    """
    positions = get_open_positions()
    closed_tickets: list[int] = []

    for pos in positions:
        ticket: int = pos.get("ticket")
        symbol: str = pos.get("symbol", "")
        direction: str = pos.get("direction", "buy")
        entry_price: float = pos.get("price_open", pos.get("entry_price", 0.0))
        open_time = pos.get("time", pos.get("open_time"))

        if open_time is None or ticket is None:
            continue

        if isinstance(open_time, str):
            open_time = datetime.fromisoformat(open_time)

        age = _position_age_seconds(open_time)

        # --- 30-minute hard close ---
        if age > _SCALP_CLOSE_SECONDS:
            log.warning(
                "time_exit_30min",
                ticket=ticket,
                symbol=symbol,
                age_seconds=round(age),
            )
            if close_position(ticket):
                closed_tickets.append(ticket)
            continue

        # --- 15-minute tighten ---
        if age > _SCALP_TIGHTEN_SECONDS:
            tick = get_tick(symbol)
            price = _current_price_for(tick, direction)
            if price is None:
                continue

            profit = _profit_pips(symbol, entry_price, price, direction)
            if profit > 0:
                log.info(
                    "time_tighten_15min",
                    ticket=ticket,
                    symbol=symbol,
                    profit_pips=round(profit, 1),
                    age_seconds=round(age),
                )
                _move_to_breakeven(ticket, entry_price, direction, symbol)
            else:
                log.debug(
                    "time_exit_in_loss_let_sl_handle",
                    ticket=ticket,
                    symbol=symbol,
                    profit_pips=round(profit, 1),
                )

    return closed_tickets


# ---------------------------------------------------------------------------
# Main entry point — called every tick / 5 seconds
# ---------------------------------------------------------------------------

def manage_positions(atr_values: dict[str, float]) -> None:
    """Run trailing-stop, partial-close, and time-exit logic for every open position.

    Parameters
    ----------
    atr_values:
        Mapping of ``symbol -> current ATR`` (in price terms, not pips).
        Must contain entries for every symbol with open positions.
    """
    positions = get_open_positions()

    if not positions:
        return

    log.debug("position_manager_tick", open_count=len(positions))

    for pos in positions:
        ticket: int = pos.get("ticket")
        symbol: str = pos.get("symbol", "")
        direction: str = pos.get("direction", "buy")
        entry_price: float = pos.get("price_open", pos.get("entry_price", 0.0))
        sl_price: float = pos.get("sl", 0.0)
        tp_price: float = pos.get("tp", 0.0)
        tp2_price: float = pos.get("tp2", 0.0)
        open_time = pos.get("time", pos.get("open_time"))

        if ticket is None:
            continue

        tick = get_tick(symbol)
        price = _current_price_for(tick, direction)
        if price is None:
            log.warning("no_tick_for_position", ticket=ticket, symbol=symbol)
            continue

        atr = atr_values.get(symbol, 0.0)
        if atr <= 0:
            log.warning("missing_atr_for_symbol", symbol=symbol, ticket=ticket)

        # Current profit metrics
        profit = _profit_pips(symbol, entry_price, price, direction)
        risk = _risk_pips(symbol, entry_price, sl_price) if sl_price else 0.0

        log.debug(
            "position_status",
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            profit_pips=round(profit, 1),
            risk_pips=round(risk, 1),
        )

        # ---------------------------------------------------------------
        # Partial-close on TP hits
        # ---------------------------------------------------------------
        _handle_tp_hits(pos, ticket, symbol, direction, entry_price, price, tp_price, tp2_price)

        # ---------------------------------------------------------------
        # Trailing-stop progression (only when risk > 0)
        # ---------------------------------------------------------------
        if risk > 0:
            _handle_trailing_stop(
                ticket=ticket,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                price=price,
                profit=profit,
                risk=risk,
                atr=atr,
            )

    # ---------------------------------------------------------------
    # Time-based exits (independent pass)
    # ---------------------------------------------------------------
    closed = check_time_exits()
    if closed:
        log.info("time_exits_applied", tickets=closed)


# ---------------------------------------------------------------------------
# Internal: trailing-stop progression
# ---------------------------------------------------------------------------

def _handle_trailing_stop(
    *,
    ticket: int,
    symbol: str,
    direction: str,
    entry_price: float,
    price: float,
    profit: float,
    risk: float,
    atr: float,
) -> None:
    """Apply the 3-tier trailing-stop logic for a single position."""

    ratio = profit / risk if risk else 0.0

    # Tier 3: profit > 3x risk -> close 50 %, trail tighter (0.5x ATR)
    if ratio > 3.0:
        log.info(
            "trailing_tier3",
            ticket=ticket,
            symbol=symbol,
            ratio=round(ratio, 2),
        )
        partial_close(ticket, percent=50.0)
        if atr > 0:
            trail_stop(ticket, price, direction, atr, multiplier=0.5)
        return

    # Tier 2: profit > 2x risk -> trail at 1x ATR
    if ratio > 2.0:
        log.info(
            "trailing_tier2",
            ticket=ticket,
            symbol=symbol,
            ratio=round(ratio, 2),
        )
        if atr > 0:
            trail_stop(ticket, price, direction, atr, multiplier=1.0)
        return

    # Tier 1: profit > 1.5x risk -> breakeven + 1 pip
    if ratio > 1.5:
        log.info(
            "trailing_tier1",
            ticket=ticket,
            symbol=symbol,
            ratio=round(ratio, 2),
        )
        _move_to_breakeven(ticket, entry_price, direction, symbol)
        return


# ---------------------------------------------------------------------------
# Internal: TP-hit partial closes
# ---------------------------------------------------------------------------

def _handle_tp_hits(
    pos: dict[str, Any],
    ticket: int,
    symbol: str,
    direction: str,
    entry_price: float,
    price: float,
    tp_price: float,
    tp2_price: float,
) -> None:
    """Handle partial closes when take-profit levels are hit."""

    already_partialed: bool = pos.get("partial_closed", False)

    # TP2 hit -> close remaining
    if tp2_price and _tp_hit(price, tp2_price, direction):
        log.info(
            "tp2_hit_close_remaining",
            ticket=ticket,
            symbol=symbol,
            tp2=tp2_price,
            price=price,
        )
        close_position(ticket)
        return

    # TP1 hit -> close 50 %, move SL to breakeven (only once)
    if tp_price and not already_partialed and _tp_hit(price, tp_price, direction):
        log.info(
            "tp1_hit_partial_close",
            ticket=ticket,
            symbol=symbol,
            tp1=tp_price,
            price=price,
        )
        partial_close(ticket, percent=50.0)
        _move_to_breakeven(ticket, entry_price, direction, symbol)


def _tp_hit(price: float, tp: float, direction: str) -> bool:
    """Return ``True`` when *price* has reached or exceeded *tp*."""
    if direction == "buy":
        return price >= tp
    return price <= tp
