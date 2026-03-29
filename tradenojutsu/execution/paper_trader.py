"""Paper trading execution engine - simulates trades without real money."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tradenojutsu.data.models import Direction, Signal, Trade, TradeStatus
from tradenojutsu.infra.database import close_trade, insert_trade
from tradenojutsu.infra.logger import get_logger
from tradenojutsu.risk.manager import RiskManager, RiskParams

logger = get_logger("execution.paper")


class PaperTrader:
    """Simulates trade execution for paper trading mode.

    Tracks open positions, applies fills at current price,
    and records everything to the database.
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.open_positions: dict[int, Trade] = {}

    def execute_signal(
        self,
        signal: Signal,
        price: float,
        risk_params: RiskParams,
    ) -> Trade | None:
        """Execute a trading signal in paper mode.

        Returns the Trade object if executed, None if rejected.
        """
        # Risk check
        allowed, reason = self.risk_manager.can_trade(signal)
        if not allowed:
            logger.info(f"Trade rejected: {reason}")
            return None

        # Create trade record
        trade_id = insert_trade(
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_price=price,
            stop_loss=risk_params.stop_loss,
            take_profit=risk_params.take_profit,
            quantity=risk_params.position_size,
            strategy=signal.strategy,
            reasoning=signal.reasoning,
            metadata={
                "score": signal.score,
                "strength": signal.strength.value,
                "risk_pct": risk_params.risk_pct,
                "components": signal.components,
            },
        )

        trade = Trade(
            id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=price,
            stop_loss=risk_params.stop_loss,
            take_profit=risk_params.take_profit,
            quantity=risk_params.position_size,
            status=TradeStatus.OPEN,
            entry_time=datetime.now(timezone.utc),
            strategy=signal.strategy,
            reasoning=signal.reasoning,
        )

        self.open_positions[trade_id] = trade
        self.risk_manager.daily_trade_count += 1

        logger.info(
            f"[PAPER] Opened {signal.direction.value} {signal.symbol} @ {price:.4f} | "
            f"SL={risk_params.stop_loss:.4f} TP={risk_params.take_profit:.4f} | "
            f"Size={risk_params.position_size:.4f} Risk={risk_params.risk_pct:.1f}%"
        )
        return trade

    def update_positions(self, prices: dict[str, float], atrs: dict[str, float]) -> list[Trade]:
        """Update all open positions with current prices.

        Checks stop-loss, take-profit, and trailing stops.
        Returns list of closed trades.
        """
        closed = []

        for trade_id, trade in list(self.open_positions.items()):
            price = prices.get(trade.symbol)
            if price is None:
                continue

            # Check trailing stop
            atr = atrs.get(trade.symbol, 0)
            new_sl = self.risk_manager.check_trailing_stop(trade, price, atr)
            if new_sl is not None:
                trade.stop_loss = new_sl

            # Check exit conditions
            should_exit, reason = self.risk_manager.check_exit_conditions(trade, price)
            if should_exit:
                trade = self._close_position(trade, price, reason)
                closed.append(trade)

        return closed

    def force_close(self, trade_id: int, price: float, reason: str = "Manual close") -> Trade | None:
        """Force close a position."""
        trade = self.open_positions.get(trade_id)
        if trade is None:
            return None
        return self._close_position(trade, price, reason)

    def _close_position(self, trade: Trade, exit_price: float, reason: str) -> Trade:
        """Close a position and update records."""
        if trade.direction == Direction.LONG:
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity

        pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100 if trade.entry_price > 0 else 0

        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = TradeStatus.CLOSED
        trade.exit_time = datetime.now(timezone.utc)

        # Update database
        if trade.id:
            close_trade(trade.id, exit_price, pnl, pnl_pct)

        # Update risk manager
        self.risk_manager.on_trade_closed(pnl)

        # Remove from open positions
        if trade.id in self.open_positions:
            del self.open_positions[trade.id]

        emoji = "+" if pnl > 0 else ""
        logger.info(
            f"[PAPER] Closed {trade.direction.value} {trade.symbol} @ {exit_price:.4f} | "
            f"PnL: {emoji}{pnl:.2f} ({pnl_pct:+.1f}%) | Reason: {reason}"
        )
        return trade

    @property
    def open_count(self) -> int:
        return len(self.open_positions)
