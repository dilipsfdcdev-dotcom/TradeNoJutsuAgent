"""SQLAlchemy models for the trading agent database."""

import uuid
from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import (
    String, Integer, BigInteger, Text, Date, Numeric,
    DateTime, JSON, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(4), nullable=False)  # buy/sell
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 5))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 5))
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(12, 5))
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(12, 5))
    lot_size: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(10), default="open")  # open/closed/cancelled
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    pnl_pips: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    rr_planned: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    rr_actual: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    confidence: Mapped[int | None] = mapped_column(Integer)
    risk_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    magic_number: Mapped[int | None] = mapped_column(Integer)

    # Context snapshot
    trend_3m: Mapped[str | None] = mapped_column(String(10))
    atr_at_entry: Mapped[Decimal | None] = mapped_column(Numeric(10, 5))
    session: Mapped[str | None] = mapped_column(String(10))
    sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    patterns_detected: Mapped[dict | None] = mapped_column(JSONB)
    indicators_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    ai_reasoning: Mapped[str | None] = mapped_column(Text)

    # Review
    ai_review: Mapped[dict | None] = mapped_column(JSONB)
    entry_quality: Mapped[int | None] = mapped_column(Integer)
    trade_quality: Mapped[int | None] = mapped_column(Integer)
    lesson: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_status", "status"),
        Index("idx_trades_entry_time", "entry_time"),
    )


class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    starting_balance: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    ending_balance: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    total_trades: Mapped[int | None] = mapped_column(Integer)
    winning_trades: Mapped[int | None] = mapped_column(Integer)
    losing_trades: Mapped[int | None] = mapped_column(Integer)
    gross_profit: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    gross_loss: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    net_pnl: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    ai_weekly_review: Mapped[dict | None] = mapped_column(JSONB)


class AgentLog(Base):
    __tablename__ = "agent_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    level: Mapped[str | None] = mapped_column(String(10))
    module: Mapped[str | None] = mapped_column(String(50))
    symbol: Mapped[str | None] = mapped_column(String(10))
    message: Mapped[str | None] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("idx_log_timestamp", "timestamp"),
    )


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
