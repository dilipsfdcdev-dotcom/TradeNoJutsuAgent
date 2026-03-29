"""Ratchet stop-loss manager - locks in profit as trades move favorably.

Uses a 6-level ratchet table to progressively tighten the stop-loss
toward profit as the trade moves in the desired direction.  The key
invariant: the SL only ever moves *toward* profit, never backward.

Supports both paper-trading (synchronous ``check_and_update``) and
MT5 live trading (async ``run_forever`` loop).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tradenojutsu.data.models import Direction, Trade
from tradenojutsu.infra.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("execution.ratchet_sl")

# ── Ratchet table ──────────────────────────────────────────────────
# Each entry: (threshold in R-multiples, fraction of profit to lock)
RATCHET_TABLE: list[tuple[float, float]] = [
    (0.5, 0.00),   # 0.5R reached  → move SL to breakeven (lock 0%)
    (1.0, 0.30),   # 1.0R reached  → lock 30% of unrealised profit
    (1.5, 0.50),   # 1.5R reached  → lock 50%
    (2.0, 0.65),   # 2.0R reached  → lock 65%
    (2.5, 0.75),   # 2.5R reached  → lock 75%
    (3.0, 0.85),   # 3.0R reached  → lock 85%
]


# ── Pure computation (no side-effects) ─────────────────────────────

def compute_new_sl(
    trade: Trade,
    current_price: float,
    initial_risk: float | None = None,
) -> float | None:
    """Compute the ratcheted stop-loss for *trade* at *current_price*.

    Parameters
    ----------
    trade:
        A ``Trade`` dataclass instance.  Must have ``entry_price``,
        ``stop_loss``, and ``direction`` populated.
    current_price:
        The latest market price for the trade's symbol.
    initial_risk:
        The original |entry - stop_loss| distance at trade open.
        If ``None`` the function derives it from ``trade.entry_price``
        and ``trade.stop_loss`` (works when the SL has *already* been
        ratcheted, but only if the caller stored the original risk
        elsewhere; prefer passing it explicitly).

    Returns
    -------
    float | None
        The new stop-loss price if it should be tightened, or ``None``
        if no change is required.
    """
    if initial_risk is None:
        initial_risk = abs(trade.entry_price - trade.stop_loss)
    if initial_risk <= 0:
        return None

    # Signed profit in price terms
    if trade.direction == Direction.LONG:
        profit = current_price - trade.entry_price
    elif trade.direction == Direction.SHORT:
        profit = trade.entry_price - current_price
    else:
        return None

    if profit <= 0:
        return None

    # R-multiple achieved so far
    r_multiple = profit / initial_risk

    # Walk the table (highest qualifying level wins)
    best_lock_frac: float | None = None
    for threshold, lock_frac in RATCHET_TABLE:
        if r_multiple >= threshold:
            best_lock_frac = lock_frac

    if best_lock_frac is None:
        return None

    # Calculate the new SL price
    locked_profit = profit * best_lock_frac
    if trade.direction == Direction.LONG:
        new_sl = trade.entry_price + locked_profit
    else:
        new_sl = trade.entry_price - locked_profit

    new_sl = round(new_sl, 5)

    # Enforce the monotonicity invariant: SL only ever moves toward profit
    if trade.direction == Direction.LONG:
        if new_sl <= trade.stop_loss:
            return None  # would move SL backward
    else:
        if new_sl >= trade.stop_loss:
            return None  # would move SL backward

    return new_sl


# ── Paper-trading helper ───────────────────────────────────────────

def check_and_update(
    trade: Trade,
    current_price: float,
    initial_risk: float,
) -> float | None:
    """Convenience wrapper for paper trading (``PaperTrader``).

    Computes the ratcheted SL and, if it changed, mutates
    ``trade.stop_loss`` in-place and returns the new value.
    Returns ``None`` when no adjustment is needed.
    """
    new_sl = compute_new_sl(trade, current_price, initial_risk)
    if new_sl is None:
        return None

    old_sl = trade.stop_loss
    trade.stop_loss = new_sl
    logger.info(
        f"[RATCHET] {trade.symbol} {trade.direction.value} | "
        f"SL {old_sl:.5f} -> {new_sl:.5f} | price={current_price:.5f}"
    )
    return new_sl


# ── Live (MT5) ratchet manager ─────────────────────────────────────

@dataclass
class RatchetStopLossManager:
    """Continuously monitors open MT5 positions and ratchets their SLs.

    Parameters
    ----------
    mt5_executor:
        An ``MT5Executor`` instance (from ``tradenojutsu.execution.mt5_executor``).
    poll_seconds:
        How often (in seconds) the ``run_forever`` loop checks positions.
    initial_risks:
        Mapping of MT5 ticket -> original risk distance (|entry - SL|)
        at the time each trade was opened.  Callers must register every
        new trade here so the ratchet table can be evaluated correctly.
    """

    mt5_executor: object  # MT5Executor (avoid circular import at module level)
    poll_seconds: float = 5.0
    initial_risks: dict[int, float] = field(default_factory=dict)

    def register_trade(self, ticket: int, initial_risk: float) -> None:
        """Register the original risk for a newly opened trade."""
        self.initial_risks[ticket] = initial_risk
        logger.debug(f"Registered ticket {ticket} with initial risk {initial_risk:.5f}")

    def unregister_trade(self, ticket: int) -> None:
        """Remove tracking for a closed / cancelled trade."""
        self.initial_risks.pop(ticket, None)

    async def run_forever(self) -> None:
        """Main loop: poll open positions and ratchet SLs.

        Designed to be run as an ``asyncio.Task`` alongside the rest
        of the live trading pipeline.
        """
        logger.info("Ratchet SL manager started (live mode)")
        executor = self.mt5_executor  # type: ignore[attr-defined]

        while True:
            try:
                positions = executor.get_open_positions()
                for pos in positions:
                    ticket = pos.get("ticket")
                    if ticket is None:
                        continue

                    initial_risk = self.initial_risks.get(ticket)
                    if initial_risk is None or initial_risk <= 0:
                        continue

                    entry = pos.get("price_open", 0.0)
                    current_sl = pos.get("sl", 0.0)
                    current_price = pos.get("price_current", 0.0)
                    trade_type = pos.get("type", -1)  # 0 = buy, 1 = sell

                    if trade_type == 0:
                        direction = Direction.LONG
                    elif trade_type == 1:
                        direction = Direction.SHORT
                    else:
                        continue

                    # Build a lightweight Trade for compute_new_sl
                    trade = Trade(
                        id=ticket,
                        symbol=pos.get("symbol", ""),
                        direction=direction,
                        entry_price=entry,
                        stop_loss=current_sl,
                    )

                    new_sl = compute_new_sl(trade, current_price, initial_risk)
                    if new_sl is not None:
                        success = executor.modify_sl(ticket, new_sl)
                        if success:
                            logger.info(
                                f"[RATCHET-LIVE] ticket={ticket} {trade.symbol} "
                                f"SL {current_sl:.5f} -> {new_sl:.5f} "
                                f"(price={current_price:.5f})"
                            )
                        else:
                            logger.warning(
                                f"[RATCHET-LIVE] Failed to modify SL for ticket {ticket}"
                            )

            except Exception:
                logger.exception("Error in ratchet SL loop")

            await asyncio.sleep(self.poll_seconds)
