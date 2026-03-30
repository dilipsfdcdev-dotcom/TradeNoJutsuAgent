"""Real-time performance metrics computed from the trade database."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from agent.db.models import Trade

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_duration_minutes: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    total_trades: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    total_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_closed_trades(
    session: AsyncSession,
    days: int,
    *,
    symbol: str | None = None,
    trading_session: str | None = None,
) -> list[Trade]:
    """Fetch closed trades with non-null pnl within the lookback window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    conditions = [
        Trade.status == "closed",
        Trade.pnl.isnot(None),
        Trade.exit_time >= cutoff,
    ]
    if symbol is not None:
        conditions.append(Trade.symbol == symbol)
    if trading_session is not None:
        conditions.append(Trade.session == trading_session)

    stmt = (
        select(Trade)
        .where(and_(*conditions))
        .order_by(Trade.exit_time.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _compute_metrics(trades: list[Trade]) -> MetricsSnapshot:
    """Pure computation of metrics from a list of Trade objects."""
    if not trades:
        return MetricsSnapshot()

    total = len(trades)
    pnls = [float(t.pnl) for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    # Win rate
    win_rate = len(winners) / total if total else 0.0

    # Profit factor
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Average R:R (from rr_actual where available)
    rr_values = [float(t.rr_actual) for t in trades if t.rr_actual is not None]
    avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else 0.0

    # Max drawdown from equity curve
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    max_drawdown = max_dd

    # Sharpe ratio (annualised from daily returns)
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        if t.exit_time is not None:
            day_key = t.exit_time.date().isoformat()
            daily_pnl[day_key] += float(t.pnl)
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        sharpe_ratio = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    elif len(daily_returns) == 1:
        sharpe_ratio = 0.0  # not meaningful with a single data point
    else:
        sharpe_ratio = 0.0

    # Avg duration
    durations: list[float] = []
    for t in trades:
        if t.entry_time is not None and t.exit_time is not None:
            delta = (t.exit_time - t.entry_time).total_seconds() / 60.0
            durations.append(delta)
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0

    # Best / worst
    best_pnl = max(pnls)
    worst_pnl = min(pnls)

    # Consecutive wins / losses (from most recent backwards)
    cons_wins = 0
    cons_losses = 0
    for p in reversed(pnls):
        if p > 0:
            cons_wins += 1
        else:
            break
    for p in reversed(pnls):
        if p < 0:
            cons_losses += 1
        else:
            break

    total_pnl = sum(pnls)

    return MetricsSnapshot(
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
        avg_rr=round(avg_rr, 4),
        max_drawdown=round(max_drawdown, 2),
        sharpe_ratio=round(sharpe_ratio, 4),
        avg_duration_minutes=round(avg_duration, 2),
        best_trade_pnl=round(best_pnl, 2),
        worst_trade_pnl=round(worst_pnl, 2),
        total_trades=total,
        consecutive_wins=cons_wins,
        consecutive_losses=cons_losses,
        total_pnl=round(total_pnl, 2),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_live_metrics(session: AsyncSession, days: int = 30) -> MetricsSnapshot:
    """Calculate all metrics from trades in the given period."""
    trades = await _fetch_closed_trades(session, days)
    metrics = _compute_metrics(trades)
    log.info("live_metrics_computed", total_trades=metrics.total_trades, days=days)
    return metrics


async def get_equity_curve(session: AsyncSession, days: int = 30) -> list[dict]:
    """Return equity curve data points: [{time, balance, equity}].

    Each point represents the cumulative P&L after a closed trade.  The
    ``balance`` and ``equity`` fields are identical here since we track
    realised P&L only (no floating equity from open positions).
    """
    trades = await _fetch_closed_trades(session, days)
    if not trades:
        return []

    curve: list[dict] = []
    cumulative = 0.0
    for t in trades:
        cumulative += float(t.pnl)
        curve.append({
            "time": t.exit_time.isoformat() if t.exit_time else None,
            "balance": round(cumulative, 2),
            "equity": round(cumulative, 2),
        })
    return curve


async def get_metrics_by_symbol(
    session: AsyncSession, symbol: str, days: int = 30
) -> MetricsSnapshot:
    """Metrics filtered by symbol."""
    trades = await _fetch_closed_trades(session, days, symbol=symbol)
    return _compute_metrics(trades)


async def get_metrics_by_session(
    session: AsyncSession, days: int = 30
) -> dict[str, MetricsSnapshot]:
    """Metrics grouped by trading session (asian/london/ny/overlap)."""
    sessions = ["asian", "london", "ny", "overlap"]
    result: dict[str, MetricsSnapshot] = {}
    for sess_name in sessions:
        trades = await _fetch_closed_trades(session, days, trading_session=sess_name)
        result[sess_name] = _compute_metrics(trades)
    return result
