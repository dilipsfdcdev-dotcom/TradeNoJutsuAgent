"""Core data models used throughout the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Signal:
    """A trading signal produced by the analysis pipeline."""
    symbol: str
    direction: Direction
    score: float  # 0-100 confidence
    strength: SignalStrength
    strategy: str
    reasoning: str
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_actionable(self) -> bool:
        return self.direction != Direction.FLAT and self.strength != SignalStrength.NONE


@dataclass
class Trade:
    """Represents an executed or simulated trade."""
    id: int | None = None
    symbol: str = ""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    exit_price: float | None = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    quantity: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    strategy: str = ""
    reasoning: str = ""

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def risk_reward_achieved(self) -> float:
        if self.stop_loss == 0 or self.entry_price == 0:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        return self.pnl / (risk * self.quantity)


@dataclass
class MarketState:
    """Snapshot of current market conditions for a symbol."""
    symbol: str
    price: float
    regime: MarketRegime
    trend_direction: Direction
    volatility: float  # ATR as % of price
    volume_ratio: float  # current vs average volume
    key_levels: dict[str, float] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_prompt_context(self) -> str:
        """Format market state for LLM consumption."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Price: {self.price:.4f}",
            f"Regime: {self.regime.value}",
            f"Trend: {self.trend_direction.value}",
            f"Volatility (ATR%): {self.volatility:.2f}%",
            f"Volume ratio: {self.volume_ratio:.2f}x average",
        ]
        if self.key_levels:
            lines.append("Key Levels:")
            for name, level in self.key_levels.items():
                lines.append(f"  {name}: {level:.4f}")
        if self.indicators:
            lines.append("Indicators:")
            for name, val in self.indicators.items():
                lines.append(f"  {name}: {val:.4f}")
        return "\n".join(lines)


@dataclass
class PerformanceMetrics:
    """Portfolio/strategy performance summary."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    expectancy: float = 0.0

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Win Rate: {self.win_rate:.1%} | "
            f"PF: {self.profit_factor:.2f} | Sharpe: {self.sharpe_ratio:.2f} | "
            f"Max DD: {self.max_drawdown:.2%} | PnL: {self.total_pnl:.2f}"
        )
