"""Dynamic risk management — no hard limits, agent decides everything.

The AI brain and self-learning module control all risk parameters.
Position sizing, number of trades, drawdown tolerance — everything
is adaptive, just like a human trader would manage.
"""

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
    """Fully autonomous risk manager — no hard caps, no fixed limits.

    The agent dynamically decides:
    - How much to risk per trade (based on confidence + regime)
    - How many positions to hold (no cap)
    - When to scale up or down (based on performance)
    - Position sizing (ATR-adaptive)

    All parameters are tunable by the self-learning module.
    """

    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade_pct: float = 2.0,
        sl_atr_mult: float = 1.5,
        tp_rr_ratio: float = 2.0,
        trailing_stop_enabled: bool = True,
        trailing_stop_atr_mult: float = 1.0,
        # Legacy params accepted but ignored (no limits enforced)
        **kwargs,
    ):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp_rr_ratio = tp_rr_ratio
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.daily_trade_count = 0
        self.daily_pnl = 0.0

    def can_trade(self, signal: Signal) -> tuple[bool, str]:
        """Always allows trading — the agent decides, not hard limits.

        The AI brain's confidence score and signal strength already filter
        bad setups. No artificial caps on positions, trades, or drawdown.
        """
        # Only reject if capital is completely wiped out
        if self.capital <= 0:
            return False, "No capital remaining"

        return True, "OK"

    def calculate_risk_params(
        self,
        signal: Signal,
        price: float,
        atr: float,
        regime: MarketRegime = MarketRegime.RANGING,
    ) -> RiskParams:
        """Dynamically calculate position size based on signal confidence and market regime.

        Higher confidence signals get larger position sizes.
        Volatile regimes get tighter stops but same risk %.
        Trending regimes get wider stops to ride the move.
        """
        # Dynamic risk % based on signal confidence (0-100)
        # Weak signal (50) → 0.5% risk, Strong signal (100) → full risk_per_trade_pct
        confidence_factor = max(0.25, signal.score / 100.0)

        # Regime-adaptive multiplier
        regime_mult = {
            MarketRegime.HIGH_VOLATILITY: 0.7,   # Scale down in chaos
            MarketRegime.LOW_VOLATILITY: 1.3,     # Scale up in calm
            MarketRegime.TRENDING_UP: 1.2,        # Trend is your friend
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 0.8,            # Chop = smaller size
        }.get(regime, 1.0)

        # Performance-adaptive: scale up when winning, down when losing
        performance_mult = self._performance_multiplier()

        # Final risk percentage — fully dynamic, no cap
        risk_pct = self.risk_per_trade_pct * confidence_factor * regime_mult * performance_mult
        risk_amount = self.capital * (risk_pct / 100)

        # Stop-loss distance — regime-adaptive
        sl_mult = self.sl_atr_mult
        if regime == MarketRegime.HIGH_VOLATILITY:
            sl_mult *= 1.5  # Wider stops in volatile markets
        elif regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
            sl_mult *= 1.2  # Give trends room to breathe

        sl_distance = atr * sl_mult
        if sl_distance <= 0:
            sl_distance = price * 0.02

        # Position size from risk
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0

        # Take-profit — dynamic R:R based on regime
        tp_ratio = self.tp_rr_ratio
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            tp_ratio *= 1.5  # Let winners run in trends
        elif regime == MarketRegime.RANGING:
            tp_ratio *= 0.8  # Take profit faster in ranges

        if signal.direction == Direction.LONG:
            stop_loss = price - sl_distance
            take_profit = price + (sl_distance * tp_ratio)
        else:
            stop_loss = price + sl_distance
            take_profit = price - (sl_distance * tp_ratio)

        logger.info(
            f"Dynamic risk: {risk_pct:.2f}% (conf={confidence_factor:.2f} "
            f"regime={regime_mult:.1f} perf={performance_mult:.2f}) "
            f"size={position_size:.4f} SL={sl_distance:.4f} RR=1:{tp_ratio:.1f}"
        )

        return RiskParams(
            position_size=round(position_size, 6),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct, 2),
        )

    def _performance_multiplier(self) -> float:
        """Scale risk based on recent performance — like a human would.

        Winning streak → trade bigger (up to 2x).
        Losing streak → trade smaller (down to 0.3x).
        """
        recent = get_recent_trades(20)
        if not recent:
            return 1.0

        closed = [t for t in recent if t.get("pnl") is not None]
        if len(closed) < 3:
            return 1.0

        # Last 5 trades
        last_5 = closed[:5]
        wins = sum(1 for t in last_5 if t["pnl"] > 0)
        losses = len(last_5) - wins

        # Winning streak → scale up
        if wins >= 4:
            return 1.5
        if wins >= 3:
            return 1.2

        # Losing streak → scale down (but never stop)
        if losses >= 4:
            return 0.4
        if losses >= 3:
            return 0.6

        return 1.0

    def check_trailing_stop(self, trade: Trade, current_price: float, atr: float) -> float | None:
        """Check and update trailing stop-loss."""
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
        """Check if a trade should be exited."""
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
        """Update accounting after a trade closes."""
        self.daily_pnl += pnl
        self.capital += pnl

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_trade_count = 0
        self.daily_pnl = 0.0

    def _trade_risk(self, trade_dict: dict) -> float:
        """Calculate the risk amount of an open trade."""
        entry = trade_dict.get("entry_price", 0)
        sl = trade_dict.get("stop_loss", 0)
        qty = trade_dict.get("quantity", 0)
        return abs(entry - sl) * qty
