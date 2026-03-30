"""Validates trade decisions and produces executable order parameters."""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from agent.brain.analyst import TradeDecision
from agent.brain.risk_manager import (
    calculate_lot_size,
    get_adjusted_risk_pct,
    validate_rr_ratio,
)
from agent.config import settings
from agent.signals.market_structure import MarketContext

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Order dataclass
# ---------------------------------------------------------------------------


@dataclass
class Order:
    symbol: str
    direction: str  # "buy" or "sell"
    lot_size: float
    entry_type: str  # "market" or "limit"
    entry_price: float
    stop_loss: float
    take_profit: float
    magic_number: int
    comment: str
    confidence: int
    risk_pct: float
    rr_planned: float


# ---------------------------------------------------------------------------
# Trade planning
# ---------------------------------------------------------------------------


def plan_trade(
    decision: TradeDecision,
    context: MarketContext,
    account_info: dict,
    open_positions: list,
    consecutive_losses: int,
    consecutive_wins: int,
    daily_loss_remaining: float,
) -> Order | None:
    """Convert an AI trade decision into an executable :class:`Order`.

    Process
    -------
    1. Gate checks (max open trades, R:R validation).
    2. Calculate dynamically adjusted risk percentage.
    3. Calculate lot size via risk manager.
    4. Verify risk amount fits within the remaining daily loss budget.
    5. Build and validate the :class:`Order`.
    6. Return ``None`` if any validation step fails.
    """
    # -- Gate: action must be actionable --
    if decision.action not in ("buy", "sell"):
        logger.info(
            "trade_skipped",
            reason="action_is_wait",
            action=decision.action,
            symbol=decision.symbol,
        )
        return None

    # -- Gate: max open trades --
    if len(open_positions) >= settings.MAX_OPEN_TRADES:
        logger.warning(
            "trade_blocked",
            reason="max_open_trades_reached",
            open=len(open_positions),
            max=settings.MAX_OPEN_TRADES,
            symbol=decision.symbol,
        )
        return None

    # -- Gate: minimum R:R ratio --
    if not validate_rr_ratio(
        decision.entry_price, decision.stop_loss, decision.take_profit
    ):
        sl_dist = abs(decision.entry_price - decision.stop_loss)
        tp_dist = abs(decision.take_profit - decision.entry_price)
        actual_rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
        logger.warning(
            "trade_blocked",
            reason="rr_ratio_below_minimum",
            symbol=decision.symbol,
            actual_rr=round(actual_rr, 2),
            min_rr=settings.MIN_RR_RATIO,
        )
        return None

    # -- Dynamic risk percentage --
    risk_pct = get_adjusted_risk_pct(
        account_state=account_info,
        market_context=context,
        consecutive_losses=consecutive_losses,
        consecutive_wins=consecutive_wins,
        confidence=decision.confidence,
    )

    if risk_pct <= 0:
        logger.warning(
            "trade_blocked",
            reason="risk_pct_zero_trading_halted",
            symbol=decision.symbol,
        )
        return None

    # -- Lot size --
    lot_size = calculate_lot_size(
        symbol=decision.symbol,
        balance=account_info["balance"],
        entry=decision.entry_price,
        sl=decision.stop_loss,
        risk_pct=risk_pct,
    )

    if lot_size <= 0:
        logger.warning(
            "trade_blocked",
            reason="lot_size_zero",
            symbol=decision.symbol,
        )
        return None

    # -- Gate: daily loss budget --
    risk_amount = account_info["balance"] * (risk_pct / 100.0)
    if risk_amount > daily_loss_remaining:
        logger.warning(
            "trade_blocked",
            reason="daily_loss_limit_near",
            symbol=decision.symbol,
            risk_amount=round(risk_amount, 2),
            daily_loss_remaining=round(daily_loss_remaining, 2),
        )
        return None

    # -- Compute planned R:R --
    sl_dist = abs(decision.entry_price - decision.stop_loss)
    tp_dist = abs(decision.take_profit - decision.entry_price)
    rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

    # -- Magic number (unique-ish trade identifier from timestamp) --
    magic = int(time.time()) % 1_000_000

    # -- Entry type (scalping strategy defaults to market orders) --
    entry_type = "market"

    # -- Comment --
    comment = f"{decision.symbol}_{decision.action}_{decision.confidence}"

    order = Order(
        symbol=decision.symbol,
        direction=decision.action,
        lot_size=round(lot_size, 2),
        entry_type=entry_type,
        entry_price=decision.entry_price,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        magic_number=magic,
        comment=comment,
        confidence=decision.confidence,
        risk_pct=risk_pct,
        rr_planned=round(rr, 2),
    )

    # -- Final structural validation --
    valid, reason = validate_trade(order)
    if not valid:
        logger.warning(
            "trade_invalid",
            reason=reason,
            symbol=order.symbol,
            direction=order.direction,
            lot_size=order.lot_size,
            entry=order.entry_price,
            sl=order.stop_loss,
            tp=order.take_profit,
        )
        return None

    logger.info(
        "trade_planned",
        symbol=order.symbol,
        direction=order.direction,
        lot=order.lot_size,
        entry=order.entry_price,
        sl=order.stop_loss,
        tp=order.take_profit,
        rr=order.rr_planned,
        risk_pct=order.risk_pct,
        confidence=order.confidence,
        magic=order.magic_number,
    )
    return order


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_trade(order: Order) -> tuple[bool, str]:
    """Run structural sanity checks on the order.

    Returns ``(True, "")`` when valid, or ``(False, reason_code)`` on failure.
    """
    if order.lot_size <= 0:
        return False, "lot_size_zero_or_negative"

    if order.lot_size > 100:  # safety cap
        return False, "lot_size_exceeds_max"

    if order.stop_loss <= 0 or order.take_profit <= 0:
        return False, "invalid_sl_or_tp"

    if order.entry_price <= 0:
        return False, "invalid_entry_price"

    # Direction-specific price consistency
    if order.direction == "buy":
        if order.stop_loss >= order.entry_price:
            return False, "sl_above_entry_for_buy"
        if order.take_profit <= order.entry_price:
            return False, "tp_below_entry_for_buy"

    if order.direction == "sell":
        if order.stop_loss <= order.entry_price:
            return False, "sl_below_entry_for_sell"
        if order.take_profit >= order.entry_price:
            return False, "tp_above_entry_for_sell"

    return True, ""
