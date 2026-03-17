"""
PostgreSQL database layer for the XAUUSD trading agent.

Uses *asyncpg* for high-performance async access with a managed connection
pool.  All DDL is applied idempotently via ``init_db``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import asyncpg

from xauusd_agent.infra.logger import get_logger

logger = get_logger("database")

# ---------------------------------------------------------------------------
# DSN from environment
# ---------------------------------------------------------------------------

DEFAULT_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/xauusd",
)
POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))

# ---------------------------------------------------------------------------
# DDL -- executed once via init_db
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(10) DEFAULT 'XAUUSD',
    action          VARCHAR(4) NOT NULL,
    buy_score       FLOAT,
    sell_score      FLOAT,
    entry_threshold FLOAT,
    ppo_action      INTEGER,
    ppo_confidence  FLOAT,
    lgbm_probability FLOAT,
    llm_action      VARCHAR(10),
    llm_reason      TEXT,
    htf_score       FLOAT,
    htf_direction   VARCHAR(20),
    d1_score        FLOAT,
    h4_score        FLOAT,
    h1_score        FLOAT,
    m15_at_support  BOOLEAN,
    regime          VARCHAR(30),
    is_counter_trend BOOLEAN DEFAULT FALSE,
    news_blackout   BOOLEAN DEFAULT FALSE,
    skip_reason     VARCHAR(100),
    was_executed    BOOLEAN DEFAULT FALSE,
    rsi_m3          FLOAT,
    rsi_m5          FLOAT,
    macd_m3         FLOAT,
    spread_pips     FLOAT,
    atr_m3          FLOAT,
    atr_m5          FLOAT
);

CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    ticket          BIGINT UNIQUE NOT NULL,
    signal_id       INTEGER REFERENCES signals(id),
    open_time       TIMESTAMPTZ NOT NULL,
    close_time      TIMESTAMPTZ,
    direction       VARCHAR(4) NOT NULL,
    lot_size        FLOAT NOT NULL,
    open_price      FLOAT NOT NULL,
    close_price     FLOAT,
    sl_price        FLOAT NOT NULL,
    tp_price        FLOAT NOT NULL,
    initial_sl_pips FLOAT,
    initial_risk_usd FLOAT,
    profit_usd      FLOAT,
    profit_pips     FLOAT,
    profit_R        FLOAT,
    status          VARCHAR(10) DEFAULT 'OPEN',
    close_reason    VARCHAR(50),
    ratchet_triggered BOOLEAN DEFAULT FALSE,
    ratchet_max_R   FLOAT,
    profit_locked_usd FLOAT,
    is_counter_trend BOOLEAN DEFAULT FALSE,
    htf_bias_score  FLOAT,
    regime          VARCHAR(30),
    balance_at_entry FLOAT,
    feature_snapshot JSONB
);

CREATE TABLE IF NOT EXISTS ratchet_events (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    ticket          BIGINT NOT NULL,
    old_sl          FLOAT,
    new_sl          FLOAT,
    current_price   FLOAT,
    profit_R        FLOAT,
    ratchet_level   VARCHAR(20),
    profit_locked_usd FLOAT
);

CREATE TABLE IF NOT EXISTS learning_log (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    trigger_type    VARCHAR(50),
    param_name      VARCHAR(60),
    old_value       FLOAT,
    new_value       FLOAT,
    reason          TEXT,
    trades_analysed INTEGER,
    win_rate        FLOAT
);

CREATE TABLE IF NOT EXISTS regime_log (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    regime          VARCHAR(30),
    htf_score       FLOAT,
    d1_score        FLOAT,
    h4_score        FLOAT,
    h1_score        FLOAT,
    atr_m5          FLOAT,
    atr_ratio       FLOAT
);

CREATE TABLE IF NOT EXISTS bot_commands (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    command         VARCHAR(30),
    value           VARCHAR(50),
    processed       BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS bot_state (
    key             VARCHAR(50) PRIMARY KEY,
    value           TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


async def init_db(dsn: str | None = None) -> asyncpg.Pool:
    """Create the connection pool and apply DDL idempotently.

    Parameters
    ----------
    dsn : str, optional
        PostgreSQL connection string.  Falls back to ``DATABASE_URL`` env var.

    Returns
    -------
    asyncpg.Pool
    """
    dsn = dsn or DEFAULT_DSN
    logger.info("Connecting to database", extra={"dsn": _sanitise_dsn(dsn)})

    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn,
        min_size=POOL_MIN,
        max_size=POOL_MAX,
    )

    async with pool.acquire() as conn:
        await conn.execute(_DDL)

    logger.info("Database initialised and tables ensured")
    return pool


def _sanitise_dsn(dsn: str) -> str:
    """Strip password from DSN for safe logging."""
    try:
        at = dsn.index("@")
        scheme_end = dsn.index("://") + 3
        return dsn[:scheme_end] + "***@" + dsn[at + 1 :]
    except ValueError:
        return dsn


# ---------------------------------------------------------------------------
# Helper: generic insert returning id
# ---------------------------------------------------------------------------


async def _insert_returning_id(
    pool: asyncpg.Pool,
    table: str,
    data: dict[str, Any],
) -> int:
    """Build an INSERT ... RETURNING id statement from a dict."""
    columns = list(data.keys())
    placeholders = [f"${i}" for i in range(1, len(columns) + 1)]
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) RETURNING id"
    )
    values = list(data.values())
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *values)
    return row["id"]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


async def insert_signal(pool: asyncpg.Pool, data: dict[str, Any]) -> int:
    """Insert a new signal row and return its id."""
    row_id = await _insert_returning_id(pool, "signals", data)
    logger.debug("Inserted signal", extra={"signal_id": row_id})
    return row_id


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


async def insert_trade(pool: asyncpg.Pool, data: dict[str, Any]) -> int:
    """Insert a new trade row and return its id."""
    row_id = await _insert_returning_id(pool, "trades", data)
    logger.debug("Inserted trade", extra={"trade_id": row_id, "ticket": data.get("ticket")})
    return row_id


async def update_trade(
    pool: asyncpg.Pool,
    ticket: int,
    data: dict[str, Any],
) -> None:
    """Update trade columns identified by *ticket*."""
    if not data:
        return
    set_clauses = [f"{col} = ${i}" for i, col in enumerate(data.keys(), start=1)]
    sql = f"UPDATE trades SET {', '.join(set_clauses)} WHERE ticket = ${len(data) + 1}"
    values = list(data.values()) + [ticket]
    async with pool.acquire() as conn:
        await conn.execute(sql, *values)
    logger.debug("Updated trade", extra={"ticket": ticket, "fields": list(data.keys())})


# ---------------------------------------------------------------------------
# Ratchet events
# ---------------------------------------------------------------------------


async def insert_ratchet_event(pool: asyncpg.Pool, data: dict[str, Any]) -> int:
    """Insert a ratchet SL-move event."""
    row_id = await _insert_returning_id(pool, "ratchet_events", data)
    logger.debug("Inserted ratchet event", extra={"id": row_id, "ticket": data.get("ticket")})
    return row_id


# ---------------------------------------------------------------------------
# Learning log
# ---------------------------------------------------------------------------


async def insert_learning_log(pool: asyncpg.Pool, data: dict[str, Any]) -> int:
    """Record a parameter-tuning event."""
    row_id = await _insert_returning_id(pool, "learning_log", data)
    logger.debug("Inserted learning log", extra={"id": row_id})
    return row_id


# ---------------------------------------------------------------------------
# Regime log
# ---------------------------------------------------------------------------


async def insert_regime_log(pool: asyncpg.Pool, data: dict[str, Any]) -> int:
    """Record a regime observation."""
    row_id = await _insert_returning_id(pool, "regime_log", data)
    logger.debug("Inserted regime log", extra={"id": row_id})
    return row_id


# ---------------------------------------------------------------------------
# Bot state (key-value store)
# ---------------------------------------------------------------------------


async def get_bot_state(pool: asyncpg.Pool, key: str) -> str | None:
    """Retrieve a value from ``bot_state`` or *None* if missing."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value FROM bot_state WHERE key = $1", key
        )
    return row["value"] if row else None


async def set_bot_state(pool: asyncpg.Pool, key: str, value: str) -> None:
    """Upsert a key-value pair in ``bot_state``."""
    sql = """
        INSERT INTO bot_state (key, value, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, key, value)
    logger.debug("Set bot_state", extra={"key": key})


# ---------------------------------------------------------------------------
# Bot commands
# ---------------------------------------------------------------------------


async def get_pending_commands(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Return all unprocessed bot commands ordered by timestamp."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, timestamp, command, value "
            "FROM bot_commands WHERE processed = FALSE ORDER BY timestamp"
        )
    return [dict(r) for r in rows]


async def mark_command_processed(pool: asyncpg.Pool, cmd_id: int) -> None:
    """Mark a bot command as processed."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE bot_commands SET processed = TRUE WHERE id = $1", cmd_id
        )
    logger.debug("Marked command processed", extra={"cmd_id": cmd_id})


# ---------------------------------------------------------------------------
# Trade queries
# ---------------------------------------------------------------------------


async def get_recent_trades(
    pool: asyncpg.Pool, n: int = 50
) -> list[dict[str, Any]]:
    """Return the *n* most recent trades (any status), newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM trades ORDER BY open_time DESC LIMIT $1", n
        )
    return [dict(r) for r in rows]


async def get_open_trades(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Return all currently open trades."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY open_time"
        )
    return [dict(r) for r in rows]


async def get_completed_trades_since(
    pool: asyncpg.Pool, since: datetime
) -> list[dict[str, Any]]:
    """Return trades closed after *since*."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM trades WHERE status != 'OPEN' AND close_time >= $1 "
            "ORDER BY close_time",
            since,
        )
    return [dict(r) for r in rows]


async def get_daily_pnl(pool: asyncpg.Pool) -> float:
    """Sum of ``profit_usd`` for trades closed today (UTC)."""
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COALESCE(SUM(profit_usd), 0) AS pnl "
            "FROM trades WHERE close_time >= $1",
            today,
        )
    return float(row["pnl"])
