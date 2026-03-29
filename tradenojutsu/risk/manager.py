"""Risk management system - position sizing, stop-loss, take-profit, portfolio limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tradenojutsu.data.models import Direction, MarketRegime, Signal, Trade
from tradenojutsu.infra.database import get_open_trades, get_recent_trades
from tradenojutsu.infra.logger import get_logger

logger = get_logger("risk.manager")


@dataclass
class RiskParams:
    """Risk parameters for a specific trade."""
    position_size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_pct: float


class RiskManager:
    """Manages position sizing, stop-losses, and portfolio-level risk.

    All parameters are tunable by the self-learning module.
    """

    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade_pct: float = 1.0,
        risk_per_trade_max: float = 2.0,
        max_concurrent: int = 3,
        max_daily_trades: int = 15,
        daily_drawdown_max_pct: float = 3.0,
        max_portfolio_risk_pct: float = 6.0,
        sl_atr_mult: float = 1.5,
        tp_rr_ratio: float = 2.0,
        trailing_stop_enabled: bool = True,
        trailing_stop_atr_mult: float = 1.0,
    ):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.risk_per_trade_max = risk_per_trade_max
        self.max_concurrent = max_concurrent
        self.max_daily_trades = max_daily_trades
        self.daily_drawdown_max_pct = daily_drawdown_max_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp_rr_ratio = tp_rr_ratio
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.daily_trade_count = 0
        self.daily_pnl = 0.0

    def can_trade(self, signal: Signal) -> tuple[bool, str]:
        """Check if a new trade is allowed given current risk constraints.

        Returns (allowed, reason).
        """
        # Max concurrent positions
        open_trades = get_open_trades()
        if len(open_trades) >= self.max_concurrent:
            return False, f"Max concurrent positions ({self.max_concurrent}) reached"

        # Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Daily trade limit ({self.max_daily_trades}) reached"

        # Daily drawdown check
        dd_pct = abs(self.daily_pnl / self.capital * 100) if self.daily_pnl < 0 else 0
        if dd_pct >= self.daily_drawdown_max_pct:
            return False, f"Daily drawdown limit ({self.daily_drawdown_max_pct}%) reached"

        # Portfolio risk check
        total_risk = sum(self._trade_risk(t) for t in open_trades)
        if total_risk / self.capital * 100 >= self.max_portfolio_risk_pct:
            return False, f"Portfolio risk limit ({self.max_portfolio_risk_pct}%) reached"

        return True, "OK"

    def calculate_risk_params(
        self,
        signal: Signal,
        price: float,
        atr: float,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> RiskParams:
        """Calculate position size, stop-loss, and take-profit for a trade."""
        # Adjust risk by regime
        regime_mult = {
            MarketRegime.HIGH_VOLATILITY: 0.75,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.RANGING: 0.9,
        }.get(regime, 1.0)

        risk_pct = min(
            self.risk_per_trade_pct * regime_mult,
            self.risk_per_trade_max,
        )
        risk_amount = self.capital * (risk_pct / 100)

        # Stop-loss distance based on ATR
        sl_distance = atr * self.sl_atr_mult
        if sl_distance <= 0:
            sl_distance = price * 0.02  # Fallback: 2% of price

        # Position size from risk
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0

        # Stop-loss and take-profit prices
        if signal.direction == Direction.LONG:
            stop_loss = price - sl_distance
            take_profit = price + (sl_distance * self.tp_rr_ratio)
        else:
            stop_loss = price + sl_distance
            take_profit = price - (sl_distance * self.tp_rr_ratio)

        return RiskParams(
            position_size=round(position_size, 6),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct, 2),
        )

    def check_trailing_stop(self, trade: Trade, current_price: float, atr: float) -> float | None:
        """Check and update trailing stop-loss.

        Returns new stop-loss level if it should be updated, None otherwise.
        """
        if not self.trailing_stop_enabled:
            return None

        trail_distance = atr * self.trailing_stop_atr_mult

        if trade.direction == Direction.LONG:
            new_sl = current_price - trail_distance
            if new_sl > trade.stop_loss:
                return round(new_sl, 4)
        else:
            new_sl = current_price + trail_distance
            if new_sl < trade.stop_loss:
                return round(new_sl, 4)

        return None

    def check_exit_conditions(self, trade: Trade, current_price: float) -> tuple[bool, str]:
        """Check if a trade should be exited.

        Returns (should_exit, reason).
        """
        if trade.direction == Direction.LONG:
            if current_price <= trade.stop_loss:
                return True, "Stop-loss hit"
            if current_price >= trade.take_profit:
                return True, "Take-profit hit"
        else:
            if current_price >= trade.stop_loss:
                return True, "Stop-loss hit"
            if current_price <= trade.take_profit:
                return True, "Take-profit hit"

        return False, ""

    def on_trade_closed(self, pnl: float) -> None:
        """Update daily accounting after a trade closes."""
        self.daily_pnl += pnl
        self.capital += pnl

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self.daily_trade_count = 0
        self.daily_pnl = 0.0

    def _trade_risk(self, trade_dict: dict) -> float:
        """Calculate the risk amount of an open trade."""
        entry = trade_dict.get("entry_price", 0)
        sl = trade_dict.get("stop_loss", 0)
        qty = trade_dict.get("quantity", 0)
        return abs(entry - sl) * qty
