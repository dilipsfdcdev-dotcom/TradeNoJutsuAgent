"""Trade journal – records every trade in PostgreSQL with full context."""

import uuid
from datetime import datetime, date, timedelta, timezone

import structlog
from sqlalchemy import select, update, and_, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from agent.db.models import Trade, DailySummary

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade_to_dict(trade: Trade) -> dict:
    """Convert a Trade ORM instance to a plain dict."""
    return {
        "id": str(trade.id),
        "symbol": trade.symbol,
        "direction": trade.direction,
        "entry_price": float(trade.entry_price) if trade.entry_price is not None else None,
        "exit_price": float(trade.exit_price) if trade.exit_price is not None else None,
        "stop_loss": float(trade.stop_loss) if trade.stop_loss is not None else None,
        "take_profit": float(trade.take_profit) if trade.take_profit is not None else None,
        "lot_size": float(trade.lot_size) if trade.lot_size is not None else None,
        "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
        "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
        "status": trade.status,
        "pnl": float(trade.pnl) if trade.pnl is not None else None,
        "pnl_pips": float(trade.pnl_pips) if trade.pnl_pips is not None else None,
        "rr_planned": float(trade.rr_planned) if trade.rr_planned is not None else None,
        "rr_actual": float(trade.rr_actual) if trade.rr_actual is not None else None,
        "confidence": trade.confidence,
        "risk_pct": float(trade.risk_pct) if trade.risk_pct is not None else None,
        "magic_number": trade.magic_number,
        "trend_3m": trade.trend_3m,
        "atr_at_entry": float(trade.atr_at_entry) if trade.atr_at_entry is not None else None,
        "session": trade.session,
        "sentiment_score": float(trade.sentiment_score) if trade.sentiment_score is not None else None,
        "patterns_detected": trade.patterns_detected,
        "indicators_snapshot": trade.indicators_snapshot,
        "ai_reasoning": trade.ai_reasoning,
        "ai_review": trade.ai_review,
        "entry_quality": trade.entry_quality,
        "trade_quality": trade.trade_quality,
        "lesson": trade.lesson,
        "created_at": trade.created_at.isoformat() if trade.created_at else None,
        "updated_at": trade.updated_at.isoformat() if trade.updated_at else None,
    }


def _summary_to_dict(s: DailySummary) -> dict:
    return {
        "date": s.date.isoformat() if s.date else None,
        "starting_balance": float(s.starting_balance) if s.starting_balance is not None else None,
        "ending_balance": float(s.ending_balance) if s.ending_balance is not None else None,
        "total_trades": s.total_trades,
        "winning_trades": s.winning_trades,
        "losing_trades": s.losing_trades,
        "gross_profit": float(s.gross_profit) if s.gross_profit is not None else None,
        "gross_loss": float(s.gross_loss) if s.gross_loss is not None else None,
        "net_pnl": float(s.net_pnl) if s.net_pnl is not None else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def record_trade(session: AsyncSession, trade_data: dict) -> str:
    """Record a new trade. Returns the trade UUID as string."""
    trade_id = uuid.uuid4()
    trade = Trade(id=trade_id, **trade_data)
    session.add(trade)
    await session.commit()
    log.info(
        "trade_recorded",
        trade_id=str(trade_id),
        symbol=trade_data.get("symbol"),
        direction=trade_data.get("direction"),
    )
    return str(trade_id)


async def update_trade(session: AsyncSession, trade_id: str, **fields) -> bool:
    """Update a trade record (e.g., when closed).

    Accepted fields: exit_price, exit_time, status, pnl, pnl_pips, rr_actual.
    """
    allowed = {"exit_price", "exit_time", "status", "pnl", "pnl_pips", "rr_actual"}
    filtered = {k: v for k, v in fields.items() if k in allowed}
    if not filtered:
        log.warning("update_trade_no_valid_fields", trade_id=trade_id)
        return False

    stmt = (
        update(Trade)
        .where(Trade.id == uuid.UUID(trade_id))
        .values(**filtered)
    )
    result = await session.execute(stmt)
    await session.commit()
    updated = result.rowcount > 0
    log.info("trade_updated", trade_id=trade_id, fields=list(filtered.keys()), updated=updated)
    return updated


async def add_review(session: AsyncSession, trade_id: str, review: dict) -> bool:
    """Add AI review to a closed trade.

    review should contain: ai_review (dict), entry_quality, trade_quality, lesson.
    """
    allowed = {"ai_review", "entry_quality", "trade_quality", "lesson"}
    filtered = {k: v for k, v in review.items() if k in allowed}
    if not filtered:
        log.warning("add_review_no_valid_fields", trade_id=trade_id)
        return False

    stmt = (
        update(Trade)
        .where(Trade.id == uuid.UUID(trade_id))
        .values(**filtered)
    )
    result = await session.execute(stmt)
    await session.commit()
    updated = result.rowcount > 0
    log.info("trade_review_added", trade_id=trade_id, updated=updated)
    return updated


async def get_trades(
    session: AsyncSession,
    symbol: str = None,
    status: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 100,
) -> list[dict]:
    """Query trades with optional filters. Returns list of dicts."""
    conditions = []
    if symbol is not None:
        conditions.append(Trade.symbol == symbol)
    if status is not None:
        conditions.append(Trade.status == status)
    if start_date is not None:
        conditions.append(Trade.entry_time >= start_date)
    if end_date is not None:
        conditions.append(Trade.entry_time <= end_date)

    stmt = select(Trade).order_by(Trade.entry_time.desc())
    if conditions:
        stmt = stmt.where(and_(*conditions))
    stmt = stmt.limit(limit)

    result = await session.execute(stmt)
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


async def get_daily_summary(session: AsyncSession, target_date: date) -> dict:
    """Get or compute daily summary for a given date.

    If a persisted DailySummary row exists it is returned directly.  Otherwise
    the summary is computed on the fly from Trade rows whose entry_time falls
    on *target_date*.
    """
    # Check for an existing persisted summary first.
    stmt = select(DailySummary).where(DailySummary.date == target_date)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        return _summary_to_dict(existing)

    # Compute from trades.
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    stmt = select(Trade).where(
        and_(Trade.entry_time >= day_start, Trade.entry_time < day_end)
    )
    result = await session.execute(stmt)
    trades = result.scalars().all()

    total_trades = len(trades)
    winning = [t for t in trades if t.pnl is not None and t.pnl > 0]
    losing = [t for t in trades if t.pnl is not None and t.pnl < 0]
    gross_profit = sum(float(t.pnl) for t in winning)
    gross_loss = sum(float(t.pnl) for t in losing)
    net_pnl = gross_profit + gross_loss

    return {
        "date": target_date.isoformat(),
        "starting_balance": None,
        "ending_balance": None,
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
    }


async def get_recent_trades_for_symbol(
    session: AsyncSession, symbol: str, limit: int = 10
) -> list[dict]:
    """Get last N trades for a symbol, used for AI context."""
    stmt = (
        select(Trade)
        .where(Trade.symbol == symbol)
        .order_by(Trade.entry_time.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


async def get_consecutive_losses(
    session: AsyncSession, symbol: str = None
) -> int:
    """Count consecutive losses from most recent trade backwards.

    Only considers closed trades with a non-null pnl.
    """
    conditions = [Trade.status == "closed", Trade.pnl.isnot(None)]
    if symbol is not None:
        conditions.append(Trade.symbol == symbol)

    stmt = (
        select(Trade.pnl)
        .where(and_(*conditions))
        .order_by(Trade.exit_time.desc())
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    count = 0
    for pnl in rows:
        if pnl < 0:
            count += 1
        else:
            break
    return count
