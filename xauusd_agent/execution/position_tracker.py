"""
Position reconciliation between the local database and MetaTrader 5.

Detects positions that were closed externally (e.g. by SL/TP or manual
intervention) and ensures the DB stays in sync.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5

from xauusd_agent.execution.mt5_executor import MT5Executor
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class PositionTracker:
    """Reconcile local DB state with live MT5 positions."""

    def __init__(self, db_pool: Any, executor: MT5Executor) -> None:
        self.db = db_pool
        self.executor = executor

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    async def sync_positions(self) -> list[dict[str, Any]]:
        """Compare DB open trades with MT5 positions and reconcile.

        Returns:
            A list of dicts for trades that were discovered to be closed,
            each containing P&L information.
        """
        # 1. Open trades according to the DB.
        db_open = await self._get_db_open_trades()
        db_tickets = {row["ticket"] for row in db_open}

        # 2. Live positions from MT5.
        mt5_positions = self.executor.get_open_positions(symbol="XAUUSD")
        mt5_tickets = {pos["ticket"] for pos in mt5_positions}

        newly_closed: list[dict[str, Any]] = []

        # 3. Trades in DB marked OPEN but absent from MT5 → closed.
        closed_tickets = db_tickets - mt5_tickets
        for ticket in closed_tickets:
            close_info = self._fetch_close_info(ticket)
            await self._mark_trade_closed(ticket, close_info)
            newly_closed.append(close_info)
            logger.info(
                "Trade %d detected as closed: profit=%.2f",
                ticket,
                close_info.get("profit", 0.0),
                extra=close_info,
            )

        # 4. Positions in MT5 not present in DB → warn (manual trade?).
        orphan_tickets = mt5_tickets - db_tickets
        for ticket in orphan_tickets:
            logger.warning(
                "MT5 position %d has no matching DB record (manual trade?)",
                ticket,
            )

        if newly_closed:
            logger.info(
                "sync_positions: %d trades reconciled as closed",
                len(newly_closed),
            )

        return newly_closed

    # ------------------------------------------------------------------
    # Enriched positions
    # ------------------------------------------------------------------

    async def get_enriched_positions(self) -> list[dict[str, Any]]:
        """Return live MT5 positions enriched with DB metadata.

        The returned dicts include all fields from
        ``MT5Executor.get_open_positions`` plus ``initial_risk_usd``,
        ``pip_value``, ``pip_size``, and ``current_price``.
        """
        mt5_positions = self.executor.get_open_positions(symbol="XAUUSD")
        if not mt5_positions:
            return []

        db_open = await self._get_db_open_trades()
        db_map: dict[int, dict[str, Any]] = {
            row["ticket"]: row for row in db_open
        }

        # Fetch symbol info once for pip metadata.
        sym_info = self.executor.connector.get_symbol_info("XAUUSD")
        pip_size = 0.1  # XAUUSD default
        pip_value = 1.0
        if sym_info:
            pip_size = sym_info.get("trade_tick_size", pip_size)
            tick_value = sym_info.get("trade_tick_value", 1.0)
            tick_size = sym_info.get("trade_tick_size", pip_size)
            if tick_size > 0:
                pip_value = tick_value / tick_size * pip_size

        # Current tick for live price.
        tick = mt5.symbol_info_tick("XAUUSD")

        enriched: list[dict[str, Any]] = []
        for pos in mt5_positions:
            db_row = db_map.get(pos["ticket"], {})
            pos["initial_risk_usd"] = db_row.get("initial_risk_usd", 0.0)
            pos["pip_value"] = pip_value
            pos["pip_size"] = pip_size
            pos["symbol"] = "XAUUSD"

            if tick is not None:
                if pos["direction"] == "BUY":
                    pos["current_price"] = tick.bid
                else:
                    pos["current_price"] = tick.ask
            else:
                pos["current_price"] = pos["open_price"]

            enriched.append(pos)

        return enriched

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _get_db_open_trades(self) -> list[dict[str, Any]]:
        """Query trades marked as OPEN in the database."""
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT ticket, direction, lot_size, open_price, sl, tp, "
                    "       initial_risk_usd, comment "
                    "FROM trades WHERE status = 'OPEN'"
                )
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to query open trades from DB")
            return []

    async def _mark_trade_closed(
        self, ticket: int, close_info: dict[str, Any]
    ) -> None:
        """Update a DB trade record to CLOSED with final P&L data."""
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    "UPDATE trades "
                    "SET status = 'CLOSED', "
                    "    close_price = $1, "
                    "    profit = $2, "
                    "    close_time = $3 "
                    "WHERE ticket = $4",
                    close_info.get("close_price", 0.0),
                    close_info.get("profit", 0.0),
                    close_info.get("close_time", datetime.now(timezone.utc)),
                    ticket,
                )
        except Exception:
            logger.exception("Failed to mark trade %d as closed in DB", ticket)

    @staticmethod
    def _fetch_close_info(ticket: int) -> dict[str, Any]:
        """Retrieve closing details for a position from MT5 deal history."""
        # Look at deals for the past 30 days to find the closing deal.
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(
            now - timedelta(days=30),
            now,
            position=ticket,
        )

        if deals is None or len(deals) == 0:
            logger.warning("No history deals found for ticket %d", ticket)
            return {
                "ticket": ticket,
                "close_price": 0.0,
                "profit": 0.0,
                "close_time": now,
            }

        # The last deal in the sequence is typically the close.
        close_deal = deals[-1]
        return {
            "ticket": ticket,
            "close_price": close_deal.price,
            "profit": close_deal.profit,
            "close_time": datetime.fromtimestamp(
                close_deal.time, tz=timezone.utc
            ),
            "commission": getattr(close_deal, "commission", 0.0),
            "swap": getattr(close_deal, "swap", 0.0),
        }
