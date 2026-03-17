"""
DB-mediated command bridge between the Telegram bot and the main trading loop.

Telegram command handlers call :meth:`CommandBridge.push_command` to write rows
into ``bot_commands``.  The main loop calls :meth:`CommandBridge.pull_commands`
each cycle to fetch and mark pending commands, then acts on them.

This decoupled design means the bot and the trading loop never share mutable
state directly -- they communicate exclusively through PostgreSQL rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from xauusd_agent.infra.logger import get_logger

if TYPE_CHECKING:
    import asyncpg

logger = get_logger(__name__)


class CommandBridge:
    """Read, write, and execute commands via the ``bot_commands`` table.

    Parameters
    ----------
    db_pool : asyncpg.Pool
        Shared database connection pool.
    """

    def __init__(self, db_pool: asyncpg.Pool) -> None:
        self.db: asyncpg.Pool = db_pool

    # ------------------------------------------------------------------
    # Push (called from Telegram handlers)
    # ------------------------------------------------------------------

    async def push_command(self, command: str, value: str = "") -> int:
        """Insert a command into ``bot_commands`` and return its id.

        Parameters
        ----------
        command : str
            Command name, e.g. ``"stop"``, ``"resume"``, ``"halt"``,
            ``"risk"``.
        value : str
            Optional payload (e.g. ``"0.8"`` for a risk command).

        Returns
        -------
        int
            The auto-generated command id.
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO bot_commands (command, value) "
                "VALUES ($1, $2) RETURNING id",
                command,
                value,
            )
        cmd_id = row["id"]
        logger.info(
            "Command pushed: %s %s (id=%d)",
            command,
            value,
            cmd_id,
            extra={"command": command, "value": value, "cmd_id": cmd_id},
        )
        return cmd_id

    # ------------------------------------------------------------------
    # Pull (called from main loop)
    # ------------------------------------------------------------------

    async def pull_commands(self) -> list[dict[str, Any]]:
        """Fetch all unprocessed commands and mark them as processed.

        Uses ``FOR UPDATE SKIP LOCKED`` inside a transaction so that
        concurrent callers never process the same command twice.

        Each returned dict contains: ``id``, ``timestamp``, ``command``,
        ``value``.
        """
        async with self.db.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    "SELECT id, timestamp, command, value "
                    "FROM bot_commands "
                    "WHERE processed = FALSE "
                    "ORDER BY timestamp "
                    "FOR UPDATE SKIP LOCKED"
                )

                if rows:
                    ids = [r["id"] for r in rows]
                    await conn.execute(
                        "UPDATE bot_commands SET processed = TRUE "
                        "WHERE id = ANY($1::int[])",
                        ids,
                    )

        commands = [dict(r) for r in rows]
        if commands:
            logger.info(
                "Pulled %d command(s)",
                len(commands),
                extra={"commands": [c["command"] for c in commands]},
            )
        return commands

    # ------------------------------------------------------------------
    # Emergency halt
    # ------------------------------------------------------------------

    async def process_halt(self, executor: Any) -> str:
        """Emergency: close all positions, set status to HALTED.

        Parameters
        ----------
        executor : object
            Trade executor with a ``close_all_positions()`` method.

        Returns
        -------
        str
            Human-readable result string (e.g. ``"HALTED (closed 3/3)"``).
        """
        logger.warning("Processing EMERGENCY HALT")
        close_results: list[dict[str, Any]] = []

        # 1. Close all positions via the executor
        if executor is not None:
            try:
                close_results = executor.close_all_positions("XAUUSD")
                closed = sum(1 for r in close_results if r.get("success"))
                logger.warning(
                    "HALT: closed %d/%d positions",
                    closed,
                    len(close_results),
                )
            except Exception:
                logger.exception("HALT: error closing positions via executor")
        else:
            logger.warning("No executor available; cannot close positions")

        # 2. Set bot state to HALTED
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    "INSERT INTO bot_state (key, value, updated_at) "
                    "VALUES ('status', 'HALTED', NOW()) "
                    "ON CONFLICT (key) DO UPDATE "
                    "SET value = 'HALTED', updated_at = NOW()"
                )

                # Mark all open trades as closed by halt
                await conn.execute(
                    "UPDATE trades SET status = 'CLOSED', "
                    "close_reason = 'EMERGENCY_HALT', "
                    "close_time = $1 "
                    "WHERE status = 'OPEN'",
                    datetime.now(timezone.utc),
                )
            logger.info("Bot state set to HALTED; open trades marked closed")
        except Exception:
            logger.exception("Failed to update DB during halt")

        total = len(close_results)
        closed = sum(1 for r in close_results if r.get("success"))
        return f"HALTED (closed {closed}/{total} positions)"

    # ------------------------------------------------------------------
    # Full command processing loop (convenience for main loop)
    # ------------------------------------------------------------------

    async def process_commands(
        self,
        executor: Any,
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch and execute all pending commands in one call.

        Supported commands:

        * **stop**   -- Set ``bot_state.status = PAUSED``.
        * **resume** -- Set ``bot_state.status = RUNNING``.
        * **halt**   -- Close ALL positions + set ``HALTED``.
        * **risk**   -- Update ``risk_pct`` in DB and *settings*.

        Parameters
        ----------
        executor : object
            Trade executor (used by ``halt``).
        settings : dict
            Live settings dict (mutated in-place for ``risk`` commands).

        Returns
        -------
        list[dict]
            Processed command dicts with an added ``"result"`` key.
        """
        pending = await self.pull_commands()
        if not pending:
            return []

        results: list[dict[str, Any]] = []

        for cmd in pending:
            cmd_name = cmd["command"].lower().strip()
            value = cmd.get("value", "")

            try:
                result = await self._execute_one(
                    cmd_name, value, executor, settings
                )
                cmd["result"] = result
                logger.info(
                    "Processed command: %s -> %s",
                    cmd_name,
                    result,
                    extra={"cmd_id": cmd["id"], "command": cmd_name},
                )
            except Exception:
                cmd["result"] = "ERROR"
                logger.exception(
                    "Failed to process command %s (id=%d)",
                    cmd_name,
                    cmd["id"],
                )

            results.append(cmd)

        return results

    # ------------------------------------------------------------------
    # Individual command executors
    # ------------------------------------------------------------------

    async def _execute_one(
        self,
        command: str,
        value: str,
        executor: Any,
        settings: dict[str, Any],
    ) -> str:
        """Dispatch and execute a single command.  Returns a status string."""
        if command == "stop":
            await self._set_state("status", "PAUSED")
            logger.info("Bot status set to PAUSED via command bridge")
            return "PAUSED"

        if command == "resume":
            await self._set_state("status", "RUNNING")
            logger.info("Bot status set to RUNNING via command bridge")
            return "RUNNING"

        if command == "halt":
            return await self.process_halt(executor)

        if command == "risk":
            return await self._apply_risk(value, settings)

        logger.warning("Unknown command: %s", command)
        return f"UNKNOWN_COMMAND:{command}"

    async def _apply_risk(
        self, value: str, settings: dict[str, Any]
    ) -> str:
        """Validate and apply a risk-percentage change."""
        try:
            risk_pct = float(value)
        except (ValueError, TypeError):
            return f"INVALID_RISK_VALUE:{value}"

        if not 0.1 <= risk_pct <= 2.0:
            return f"RISK_OUT_OF_RANGE:{risk_pct}"

        await self._set_state("risk_pct", str(risk_pct))
        settings["risk_pct"] = risk_pct

        logger.info(
            "Risk updated to %.1f%% via command bridge",
            risk_pct,
            extra={"risk_pct": risk_pct},
        )
        return f"RISK_SET:{risk_pct}"

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    async def _set_state(self, key: str, value: str) -> None:
        """Upsert a key-value pair in ``bot_state``."""
        async with self.db.acquire() as conn:
            await conn.execute(
                "INSERT INTO bot_state (key, value, updated_at) "
                "VALUES ($1, $2, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()",
                key,
                value,
            )

    async def _get_state(self, key: str) -> str | None:
        """Retrieve a value from ``bot_state``."""
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM bot_state WHERE key = $1", key
            )
        return row["value"] if row else None
