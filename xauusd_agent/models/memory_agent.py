"""
Adaptive memory agent that watches trade outcomes and tunes system parameters.

Runs after every 10 completed trades.  All mutations are bounded by
``SAFETY_LIMITS`` and logged to both ``settings.yaml`` and the
``learning_log`` DB table so every change is auditable.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# Session time windows (UTC hours)
_SESSIONS: dict[str, tuple[int, int]] = {
    "london": (8, 10),
    "ny": (13, 15),
    "overlap": (13, 16),
    "asian": (0, 7),
}


class MemoryAgent:
    """Watches trade outcomes and updates system parameters within safety limits.

    All changes are written to ``settings.yaml`` **and** persisted in the
    ``learning_log`` database table.
    """

    SAFETY_LIMITS: dict[str, tuple[float, float]] = {
        "entry_threshold": (52, 78),
        "risk_pct": (0.3, 2.0),
        "sl_min_pips": (8, 8),  # immutable
        "sl_max_pips": (35, 40),
    }

    MAX_CHANGES_PER_CYCLE = 3
    MIN_TRADES_BETWEEN_SAME_CHANGE = 20

    def __init__(self, settings_path: str, db_pool: Any) -> None:
        self.settings_path = settings_path
        self.db = db_pool
        self._last_changes: dict[str, int] = {}  # param → trade_count_at_change
        self._trade_count: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_cycle(self) -> list[dict]:
        """Execute a full learning cycle.

        1. Pull last 50 completed trades with feature snapshots.
        2. Run all analysis functions.
        3. Determine which parameters to update (max 3).
        4. Write changes to ``settings.yaml`` + DB ``learning_log``.
        5. Return a summary list suitable for a Telegram alert.
        """
        trades = await self._fetch_recent_trades(limit=50)
        if len(trades) < 10:
            logger.info("Only %d completed trades; skipping learning cycle", len(trades))
            return []

        analysis = self.analyse_win_rate(trades)

        # Collect candidate parameter updates
        candidates: list[tuple[str, float, float, str] | None] = [
            self.update_entry_threshold(analysis),
            self.update_atr_multipliers(analysis),
            self.update_news_blackout(analysis),
            self.update_counter_trend_flag(analysis),
        ]
        proposals = [c for c in candidates if c is not None]

        # Enforce per-cycle cap
        changes_applied: list[dict] = []
        for param_name, old_val, new_val, reason in proposals[: self.MAX_CHANGES_PER_CYCLE]:
            if not self._can_change_param(param_name):
                logger.info(
                    "Skipping %s — too soon since last change", param_name
                )
                continue

            self._write_setting(param_name, new_val)
            self._last_changes[param_name] = self._trade_count
            await self._log_to_db(param_name, old_val, new_val, reason, len(trades), analysis)

            entry = {
                "param": param_name,
                "old": old_val,
                "new": new_val,
                "reason": reason,
            }
            changes_applied.append(entry)
            logger.info(
                "MemoryAgent updated %s: %.4f → %.4f (%s)",
                param_name, old_val, new_val, reason,
            )

        self._trade_count += len(trades)
        return changes_applied

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse_win_rate(self, trades: list[dict]) -> dict:
        """Compute win rates sliced by multiple dimensions.

        Returns a dict with keys like ``"by_htf_bias"``, ``"by_session"``,
        ``"by_regime"``, ``"by_direction"``, ``"by_counter_trend"``,
        ``"by_score_bucket"``, ``"overall"``, ``"rolling_30"``,
        ``"sl_hit_rate"``, ``"avg_winner_r"``, ``"news_trades"``.
        """
        analysis: dict[str, Any] = {}

        # --- Overall ---
        wins = [t for t in trades if t.get("profit_r", 0) > 0]
        analysis["overall"] = len(wins) / len(trades) if trades else 0.0

        # --- Rolling 30-trade window ---
        last_30 = trades[-30:]
        wins_30 = [t for t in last_30 if t.get("profit_r", 0) > 0]
        analysis["rolling_30"] = len(wins_30) / len(last_30) if last_30 else 0.0

        # --- By HTF bias ---
        analysis["by_htf_bias"] = self._bucket_win_rate(trades, "htf_bias")

        # --- By session ---
        analysis["by_session"] = self._session_win_rate(trades)

        # --- By regime ---
        analysis["by_regime"] = self._bucket_win_rate(trades, "regime")

        # --- By direction ---
        analysis["by_direction"] = self._bucket_win_rate(trades, "direction")

        # --- Counter-trend vs with-trend ---
        ct_trades = [t for t in trades if t.get("counter_trend")]
        wt_trades = [t for t in trades if not t.get("counter_trend")]
        analysis["by_counter_trend"] = {
            "counter_trend": self._win_rate(ct_trades),
            "with_trend": self._win_rate(wt_trades),
            "counter_trend_net_r": sum(t.get("profit_r", 0) for t in ct_trades),
        }

        # --- Signal-score buckets ---
        analysis["by_score_bucket"] = {
            "62_70": self._win_rate([t for t in trades if 62 <= t.get("signal_score", 0) < 70]),
            "70_80": self._win_rate([t for t in trades if 70 <= t.get("signal_score", 0) < 80]),
            "80_100": self._win_rate([t for t in trades if 80 <= t.get("signal_score", 0) <= 100]),
        }

        # --- SL hit rate ---
        sl_hits = [t for t in trades if t.get("exit_reason") == "sl"]
        analysis["sl_hit_rate"] = len(sl_hits) / len(trades) if trades else 0.0

        # --- Average winner R ---
        winner_rs = [t["profit_r"] for t in wins if "profit_r" in t]
        analysis["avg_winner_r"] = (
            sum(winner_rs) / len(winner_rs) if winner_rs else 0.0
        )

        # --- News-adjacent trades ---
        news_trades = [
            t for t in trades if t.get("minutes_to_news", 999) <= 90
        ]
        analysis["news_trades"] = {
            "count": len(news_trades),
            "win_rate": self._win_rate(news_trades),
        }

        return analysis

    # ------------------------------------------------------------------
    # Parameter update rules
    # ------------------------------------------------------------------

    def update_entry_threshold(
        self, analysis: dict
    ) -> tuple[str, float, float, str] | None:
        """Adjust ``entry_threshold`` based on rolling 30-trade win rate."""
        wr = analysis.get("rolling_30", 0.55)
        settings = self._load_settings()
        old_val = float(settings.get("entry_threshold", 65))

        if wr < 0.48:
            delta, reason = 3, f"win_rate={wr:.2%} < 48%"
        elif wr < 0.52:
            delta, reason = 1, f"win_rate={wr:.2%} in 48-52%"
        elif wr <= 0.60:
            return None  # no change
        elif wr <= 0.66:
            delta, reason = -1, f"win_rate={wr:.2%} in 60-66%"
        else:
            delta, reason = -2, f"win_rate={wr:.2%} > 66%"

        lo, hi = self.SAFETY_LIMITS["entry_threshold"]
        new_val = float(max(lo, min(hi, old_val + delta)))
        if new_val == old_val:
            return None
        return "entry_threshold", old_val, new_val, reason

    def update_atr_multipliers(
        self, analysis: dict
    ) -> tuple[str, float, float, str] | None:
        """Adjust SL/TP ATR multipliers based on SL-hit rate and winner quality."""
        settings = self._load_settings()
        sl_mult = float(settings.get("sl_multiplier", 1.5))
        tp_ratio = float(settings.get("tp_ratio", 2.0))

        sl_hit = analysis.get("sl_hit_rate", 0.45)
        avg_r = analysis.get("avg_winner_r", 1.5)

        # SL adjustment
        if sl_hit > 0.60:
            new_sl = round(sl_mult + 0.1, 2)
            return (
                "sl_multiplier",
                sl_mult,
                new_sl,
                f"SL hit rate {sl_hit:.0%} > 60%; widening SL",
            )
        if sl_hit < 0.30:
            new_sl = round(sl_mult - 0.05, 2)
            return (
                "sl_multiplier",
                sl_mult,
                new_sl,
                f"SL hit rate {sl_hit:.0%} < 30%; tightening SL",
            )

        # TP adjustment
        if avg_r < 1.3:
            new_tp = round(tp_ratio + 0.1, 2)
            return (
                "tp_ratio",
                tp_ratio,
                new_tp,
                f"avg winner R={avg_r:.2f} < 1.3; extending TP target",
            )

        return None

    def update_news_blackout(
        self, analysis: dict
    ) -> tuple[str, float, float, str] | None:
        """Widen or narrow the news-blackout window."""
        news = analysis.get("news_trades", {})
        wr = news.get("win_rate", 0.5)
        count = news.get("count", 0)

        if count < 5:
            return None  # insufficient data

        settings = self._load_settings()
        old_val = float(settings.get("news_blackout_minutes", 30))

        if wr < 0.42:
            new_val = min(60.0, old_val + 5)
            reason = f"news-window WR {wr:.0%} < 42%; extending blackout"
        elif wr > 0.58:
            new_val = max(15.0, old_val - 5)
            reason = f"news-window WR {wr:.0%} > 58%; reducing blackout"
        else:
            return None

        if new_val == old_val:
            return None
        return "news_blackout_minutes", old_val, new_val, reason

    def update_counter_trend_flag(
        self, analysis: dict
    ) -> tuple[str, float, float, str] | None:
        """Enable or disable counter-trend trading."""
        ct = analysis.get("by_counter_trend", {})
        net_r = ct.get("counter_trend_net_r", 0.0)

        settings = self._load_settings()
        current = bool(settings.get("allow_counter_trend", True))

        if net_r < 0 and current:
            return (
                "allow_counter_trend",
                1.0,
                0.0,
                f"counter-trend net R={net_r:.2f}; disabling",
            )
        if net_r > 0 and not current:
            return (
                "allow_counter_trend",
                0.0,
                1.0,
                f"counter-trend net R={net_r:.2f}; re-enabling",
            )
        return None

    # ------------------------------------------------------------------
    # Settings I/O
    # ------------------------------------------------------------------

    def _load_settings(self) -> dict:
        """Read the full settings YAML into a dict."""
        try:
            with open(self.settings_path, "r") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Settings file not found at %s", self.settings_path)
            return {}

    def _write_setting(self, param_path: str, value: Any) -> None:
        """Update a single dot-separated key in ``settings.yaml``.

        Example: ``param_path="risk.sl_multiplier"`` navigates to
        ``settings["risk"]["sl_multiplier"]``.
        """
        settings = self._load_settings()
        keys = param_path.split(".")
        node = settings
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

        tmp_path = Path(self.settings_path).with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as fh:
            yaml.safe_dump(settings, fh, default_flow_style=False)
        tmp_path.replace(self.settings_path)
        logger.debug("Settings written: %s = %s", param_path, value)

    # ------------------------------------------------------------------
    # Guard-rails
    # ------------------------------------------------------------------

    def _can_change_param(self, param_name: str) -> bool:
        """Return ``True`` if enough trades have elapsed since the last
        change to *param_name*."""
        last = self._last_changes.get(param_name)
        if last is None:
            return True
        return (self._trade_count - last) >= self.MIN_TRADES_BETWEEN_SAME_CHANGE

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    async def _fetch_recent_trades(self, limit: int = 50) -> list[dict]:
        """Pull the most recent completed trades from the DB."""
        query = """
            SELECT *
            FROM   trades
            WHERE  status = 'closed'
            ORDER  BY closed_at DESC
            LIMIT  $1
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, limit)
        # Convert to list of dicts and reverse so oldest first
        return [dict(r) for r in reversed(rows)]

    async def _log_to_db(
        self,
        param_name: str,
        old_val: float,
        new_val: float,
        reason: str,
        trades_analysed: int,
        analysis: dict,
    ) -> None:
        """Persist a learning-log entry."""
        query = """
            INSERT INTO learning_log
                (trigger_type, param_name, old_value, new_value,
                 reason, trades_analysed, win_rate)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    query,
                    "memory_agent",
                    param_name,
                    old_val,
                    new_val,
                    reason,
                    trades_analysed,
                    analysis.get("overall", 0.0),
                )
        except Exception:
            logger.exception("Failed to log learning change for %s", param_name)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _win_rate(trades: list[dict]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("profit_r", 0) > 0)
        return wins / len(trades)

    @staticmethod
    def _bucket_win_rate(trades: list[dict], key: str) -> dict[str, float]:
        buckets: dict[str, list[dict]] = {}
        for t in trades:
            val = str(t.get(key, "unknown"))
            buckets.setdefault(val, []).append(t)
        return {
            k: (
                sum(1 for t in v if t.get("profit_r", 0) > 0) / len(v)
                if v
                else 0.0
            )
            for k, v in buckets.items()
        }

    @staticmethod
    def _session_win_rate(trades: list[dict]) -> dict[str, float]:
        results: dict[str, list[dict]] = {s: [] for s in _SESSIONS}
        for t in trades:
            hour = t.get("entry_hour")
            if hour is None:
                continue
            for session, (start, end) in _SESSIONS.items():
                if start <= hour < end:
                    results[session].append(t)
        return {
            s: (
                sum(1 for t in tl if t.get("profit_r", 0) > 0) / len(tl)
                if tl
                else 0.0
            )
            for s, tl in results.items()
        }
