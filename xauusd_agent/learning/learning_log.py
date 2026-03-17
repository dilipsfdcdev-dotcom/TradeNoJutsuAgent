"""
Persistent learning log backed by the ``learning_log`` database table.

Every adaptive change made by the memory agent, retrainer, or parameter
optimizer is recorded here for auditability and Telegram alerts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class LearningLog:
    """Read/write interface to the ``learning_log`` table."""

    def __init__(self, db_pool: Any) -> None:
        self.db = db_pool

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def log_change(
        self,
        trigger_type: str,
        param_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        trades_analysed: int,
        win_rate: float,
    ) -> None:
        """Insert a single learning-log entry.

        Parameters
        ----------
        trigger_type:
            Origin of the change, e.g. ``"memory_agent"``, ``"lgbm_retrain"``,
            ``"ppo_finetune"``, ``"param_optimizer"``.
        param_name:
            Dot-separated settings key that was changed.
        old_value / new_value:
            Previous and updated values.
        reason:
            Human-readable explanation.
        trades_analysed:
            Number of trades in the analysis window.
        win_rate:
            Overall win rate at the time of the change.
        """
        query = """
            INSERT INTO learning_log
                (trigger_type, param_name, old_value, new_value,
                 reason, trades_analysed, win_rate, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        now = datetime.now(timezone.utc)
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    query,
                    trigger_type,
                    param_name,
                    old_value,
                    new_value,
                    reason,
                    trades_analysed,
                    win_rate,
                    now,
                )
            logger.debug(
                "Logged learning change: %s %s %.4f→%.4f",
                trigger_type, param_name, old_value, new_value,
            )
        except Exception:
            logger.exception(
                "Failed to log learning change: %s %s", trigger_type, param_name
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_recent_changes(self, n: int = 10) -> list[dict]:
        """Return the last *n* learning-log entries, newest first."""
        query = """
            SELECT trigger_type, param_name, old_value, new_value,
                   reason, trades_analysed, win_rate, created_at
            FROM   learning_log
            ORDER  BY created_at DESC
            LIMIT  $1
        """
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(query, n)
            return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to fetch recent learning changes")
            return []

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_learning_alert(self, changes: list[dict]) -> str:
        """Format a list of learning changes into a Telegram-friendly string.

        Parameters
        ----------
        changes:
            List of change dicts as returned by :meth:`get_recent_changes`
            or produced by :class:`MemoryAgent.run_cycle`.
        """
        if not changes:
            return "No learning changes to report."

        lines = ["LEARNING UPDATE"]
        lines.append("=" * 30)

        for i, c in enumerate(changes, 1):
            param = c.get("param_name") or c.get("param", "?")
            old = c.get("old_value") if "old_value" in c else c.get("old", "?")
            new = c.get("new_value") if "new_value" in c else c.get("new", "?")
            reason = c.get("reason", "")
            trigger = c.get("trigger_type", "")

            old_str = f"{old:.4f}" if isinstance(old, float) else str(old)
            new_str = f"{new:.4f}" if isinstance(new, float) else str(new)

            lines.append(f"{i}. {param}: {old_str} -> {new_str}")
            if reason:
                lines.append(f"   Reason: {reason}")
            if trigger:
                lines.append(f"   Trigger: {trigger}")

        return "\n".join(lines)
