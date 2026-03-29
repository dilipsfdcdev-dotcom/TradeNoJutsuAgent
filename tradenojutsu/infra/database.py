"""SQLite-based persistence for trades, signals, and learning state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path("data/tradenojutsu.db")

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    pnl REAL,
    pnl_pct REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    strategy TEXT,
    reasoning TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    score REAL NOT NULL,
    components TEXT,
    reasoning TEXT,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_num INTEGER NOT NULL,
    param_changed TEXT NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    reason TEXT,
    performance_before TEXT,
    performance_after TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thinking_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    market_state TEXT,
    chain_of_thought TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_trades INTEGER,
    win_rate REAL,
    profit_factor REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    total_pnl REAL,
    metadata TEXT,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_TABLES)
    return conn


def insert_trade(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: float,
    strategy: str = "",
    reasoning: str = "",
    metadata: dict | None = None,
) -> int:
    """Insert a new trade record."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO trades
        (symbol, direction, entry_price, stop_loss, take_profit, quantity,
         strategy, reasoning, metadata, entry_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            symbol, direction, entry_price, stop_loss, take_profit, quantity,
            strategy, reasoning, json.dumps(metadata or {}),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()
    return trade_id


def close_trade(trade_id: int, exit_price: float, pnl: float, pnl_pct: float) -> None:
    """Close an open trade with exit details."""
    conn = get_connection()
    conn.execute(
        """UPDATE trades SET exit_price=?, pnl=?, pnl_pct=?, status='closed',
        exit_time=? WHERE id=?""",
        (exit_price, pnl, pnl_pct, datetime.now(timezone.utc).isoformat(), trade_id),
    )
    conn.commit()
    conn.close()


def get_open_trades(symbol: str | None = None) -> list[dict[str, Any]]:
    """Get all open trades, optionally filtered by symbol."""
    conn = get_connection()
    query = "SELECT * FROM trades WHERE status='open'"
    params: list = []
    if symbol:
        query += " AND symbol=?"
        params.append(symbol)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_trades(n: int = 100) -> list[dict[str, Any]]:
    """Get the N most recent closed trades."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status='closed' ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_signal(
    symbol: str, direction: str, score: float, components: dict, reasoning: str, accepted: bool
) -> int:
    """Log a signal evaluation."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO signals (symbol, direction, score, components, reasoning, accepted, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (symbol, direction, score, json.dumps(components), reasoning, int(accepted),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    signal_id = cursor.lastrowid
    conn.close()
    return signal_id


def insert_thinking_log(
    symbol: str, market_state: dict, chain_of_thought: str, decision: str, confidence: float
) -> int:
    """Log an AI thinking/reasoning session."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO thinking_log (symbol, market_state, chain_of_thought, decision, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (symbol, json.dumps(market_state), chain_of_thought, decision, confidence,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def insert_learning_log(
    cycle_num: int, param: str, old_val: str, new_val: str,
    reason: str, perf_before: dict, perf_after: dict
) -> None:
    """Log a parameter change from the learning module."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO learning_log
        (cycle_num, param_changed, old_value, new_value, reason,
         performance_before, performance_after, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (cycle_num, param, old_val, new_val, reason,
         json.dumps(perf_before), json.dumps(perf_after),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def insert_performance_snapshot(metrics: dict[str, Any]) -> None:
    """Save a performance snapshot."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO performance_snapshots
        (total_trades, win_rate, profit_factor, sharpe_ratio, max_drawdown, total_pnl, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            metrics.get("total_trades", 0),
            metrics.get("win_rate", 0),
            metrics.get("profit_factor", 0),
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0),
            metrics.get("total_pnl", 0),
            json.dumps(metrics),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
