"""Initial schema - create all tables

Revision ID: 001_initial
Revises:
Create Date: 2026-03-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # trades
    op.create_table(
        "trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("direction", sa.String(4), nullable=False),
        sa.Column("entry_price", sa.Numeric(12, 5)),
        sa.Column("exit_price", sa.Numeric(12, 5)),
        sa.Column("stop_loss", sa.Numeric(12, 5)),
        sa.Column("take_profit", sa.Numeric(12, 5)),
        sa.Column("lot_size", sa.Numeric(10, 4)),
        sa.Column("entry_time", sa.DateTime(timezone=True)),
        sa.Column("exit_time", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(10), default="open"),
        sa.Column("pnl", sa.Numeric(12, 2)),
        sa.Column("pnl_pips", sa.Numeric(10, 2)),
        sa.Column("rr_planned", sa.Numeric(4, 2)),
        sa.Column("rr_actual", sa.Numeric(4, 2)),
        sa.Column("confidence", sa.Integer),
        sa.Column("risk_pct", sa.Numeric(4, 2)),
        sa.Column("magic_number", sa.Integer),
        # Context snapshot
        sa.Column("trend_3m", sa.String(10)),
        sa.Column("atr_at_entry", sa.Numeric(10, 5)),
        sa.Column("session", sa.String(10)),
        sa.Column("sentiment_score", sa.Numeric(4, 2)),
        sa.Column("patterns_detected", postgresql.JSONB),
        sa.Column("indicators_snapshot", postgresql.JSONB),
        sa.Column("ai_reasoning", sa.Text),
        # Review
        sa.Column("ai_review", postgresql.JSONB),
        sa.Column("entry_quality", sa.Integer),
        sa.Column("trade_quality", sa.Integer),
        sa.Column("lesson", sa.Text),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_trades_symbol", "trades", ["symbol"])
    op.create_index("idx_trades_status", "trades", ["status"])
    op.create_index("idx_trades_entry_time", "trades", ["entry_time"])

    # daily_summaries
    op.create_table(
        "daily_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("date", sa.Date, unique=True, nullable=False),
        sa.Column("starting_balance", sa.Numeric(12, 2)),
        sa.Column("ending_balance", sa.Numeric(12, 2)),
        sa.Column("total_trades", sa.Integer),
        sa.Column("winning_trades", sa.Integer),
        sa.Column("losing_trades", sa.Integer),
        sa.Column("gross_profit", sa.Numeric(12, 2)),
        sa.Column("gross_loss", sa.Numeric(12, 2)),
        sa.Column("net_pnl", sa.Numeric(12, 2)),
        sa.Column("max_drawdown_pct", sa.Numeric(6, 2)),
        sa.Column("notes", sa.Text),
        sa.Column("ai_weekly_review", postgresql.JSONB),
    )

    # agent_log
    op.create_table(
        "agent_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("level", sa.String(10)),
        sa.Column("module", sa.String(50)),
        sa.Column("symbol", sa.String(10)),
        sa.Column("message", sa.Text),
        sa.Column("data", postgresql.JSONB),
    )
    op.create_index("idx_log_timestamp", "agent_log", ["timestamp"])

    # settings
    op.create_table(
        "settings",
        sa.Column("key", sa.String(50), primary_key=True),
        sa.Column("value", postgresql.JSONB),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("settings")
    op.drop_table("agent_log")
    op.drop_table("daily_summaries")
    op.drop_table("trades")
