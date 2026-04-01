"""Safety mechanisms to protect the trading account.

Runs a series of pre-trade checks (daily loss, consecutive losses, drawdown,
spread, time restrictions) and exposes an emergency shutdown procedure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from agent.config import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Eastern-time helper (UTC-5 / UTC-4 during DST).  We keep it simple and use
# the stdlib only so there is no hard dependency on ``pytz`` / ``zoneinfo``
# at import time.  ``zoneinfo`` is available on Python 3.9+.
# ---------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo

    _EST = ZoneInfo("America/New_York")
except ImportError:  # pragma: no cover – fallback for older runtimes
    _EST = timezone(timedelta(hours=-5))


class CircuitBreaker:
    """Stateful pre-trade safety gate.

    Instantiate once at agent start-up and call :meth:`can_trade` before every
    new trade.  Call :meth:`record_trade_result` after each trade closes.
    """

    # ----- configurable thresholds (pulled from settings) ------------------
    MAX_DAILY_LOSS_PCT: float = settings.MAX_DAILY_LOSS_PCT
    MAX_DRAWDOWN_PCT: float = settings.MAX_DRAWDOWN_PCT
    SPREAD_FILTER_MULTIPLIER: float = settings.SPREAD_FILTER_MULTIPLIER

    SYMBOL_LOSS_STREAK_LIMIT: int = 3
    SYMBOL_COOLDOWN_HOURS: int = 1
    TOTAL_LOSS_STREAK_LIMIT: int = 5
    TOTAL_COOLDOWN_HOURS: int = 2

    def __init__(self) -> None:
        self.peak_balance: float = 0.0
        self.symbol_loss_streaks: dict[str, int] = {}
        self.total_loss_streak: int = 0
        self.symbol_cooldowns: dict[str, datetime] = {}
        self.trading_paused_until: Optional[datetime] = None
        self.agent_disabled: bool = False

    # ---- public entry point -----------------------------------------------

    def can_trade(
        self,
        symbol: str,
        current_spread: float,
        avg_spread: float,
        daily_pnl: float,
        account_info: dict,
        upcoming_events: list | None = None,
    ) -> tuple[bool, str]:
        """Run **all** pre-trade checks.

        Returns ``(True, "")`` when the trade is allowed, or
        ``(False, reason)`` when it must be blocked.
        """
        if self.agent_disabled:
            return False, "Agent is disabled after emergency shutdown"

        # 0. Global pause
        if self.trading_paused_until is not None:
            now = datetime.now(tz=timezone.utc)
            if now < self.trading_paused_until:
                remaining = (self.trading_paused_until - now).total_seconds()
                return False, (
                    f"Trading paused for {remaining:.0f}s more "
                    f"after {self.TOTAL_LOSS_STREAK_LIMIT} consecutive losses"
                )
            # cooldown expired – clear it
            self.trading_paused_until = None
            self.total_loss_streak = 0
            log.info("circuit_breaker.global_pause_expired")

        checks: list[tuple[bool, str]] = [
            self.check_daily_loss(daily_pnl, account_info.get("balance", 0.0)),
            self.check_consecutive_losses(symbol),
            self.check_drawdown(account_info),
            self.check_spread(symbol, current_spread, avg_spread),
            self.check_time_restrictions(upcoming_events),
        ]

        for allowed, reason in checks:
            if not allowed:
                log.warning(
                    "circuit_breaker.blocked",
                    symbol=symbol,
                    reason=reason,
                )
                return False, reason

        return True, ""

    # ---- individual checks -------------------------------------------------

    def check_daily_loss(
        self, daily_pnl: float, balance: float
    ) -> tuple[bool, str]:
        """Block if today's P&L exceeds the maximum daily loss threshold."""
        if balance <= 0:
            return False, "Invalid balance"

        loss_pct = abs(min(daily_pnl, 0.0)) / balance * 100.0
        if loss_pct >= self.MAX_DAILY_LOSS_PCT:
            log.warning(
                "circuit_breaker.daily_loss_limit",
                loss_pct=round(loss_pct, 2),
                max_pct=self.MAX_DAILY_LOSS_PCT,
            )
            return False, (
                f"Daily loss {loss_pct:.2f}% exceeds limit "
                f"{self.MAX_DAILY_LOSS_PCT}%"
            )
        return True, ""

    def check_consecutive_losses(self, symbol: str) -> tuple[bool, str]:
        """Block if the symbol or total loss streak has tripped a cooldown."""
        now = datetime.now(tz=timezone.utc)

        # Per-symbol cooldown
        cooldown_until = self.symbol_cooldowns.get(symbol)
        if cooldown_until is not None:
            if now < cooldown_until:
                remaining = (cooldown_until - now).total_seconds()
                return False, (
                    f"{symbol} on cooldown for {remaining:.0f}s "
                    f"after {self.SYMBOL_LOSS_STREAK_LIMIT} consecutive losses"
                )
            # cooldown expired – clear it
            del self.symbol_cooldowns[symbol]
            self.symbol_loss_streaks[symbol] = 0
            log.info(
                "circuit_breaker.symbol_cooldown_expired", symbol=symbol
            )

        return True, ""

    def check_drawdown(self, account_info: dict) -> tuple[bool, str]:
        """Trigger emergency shutdown if drawdown from peak exceeds limit."""
        balance = account_info.get("balance", 0.0)
        equity = account_info.get("equity", balance)

        # Update peak with the higher of balance and equity
        self.update_peak_balance(balance)

        if self.peak_balance <= 0:
            return True, ""

        drawdown_pct = (self.peak_balance - equity) / self.peak_balance * 100.0

        if drawdown_pct >= self.MAX_DRAWDOWN_PCT:
            log.critical(
                "circuit_breaker.max_drawdown_breached",
                drawdown_pct=round(drawdown_pct, 2),
                peak=self.peak_balance,
                equity=equity,
                max_pct=self.MAX_DRAWDOWN_PCT,
            )
            self.emergency_shutdown()
            return False, (
                f"EMERGENCY: Drawdown {drawdown_pct:.2f}% exceeds "
                f"max {self.MAX_DRAWDOWN_PCT}%. All positions closed, "
                "agent disabled."
            )
        return True, ""

    def check_spread(
        self,
        symbol: str,
        current_spread: float,
        avg_spread: float,
    ) -> tuple[bool, str]:
        """Skip if current spread is too wide relative to the average."""
        if avg_spread <= 0:
            return True, ""

        if current_spread > self.SPREAD_FILTER_MULTIPLIER * avg_spread:
            log.info(
                "circuit_breaker.spread_too_wide",
                symbol=symbol,
                current=current_spread,
                avg=avg_spread,
                multiplier=self.SPREAD_FILTER_MULTIPLIER,
            )
            return False, (
                f"{symbol} spread {current_spread:.2f} > "
                f"{self.SPREAD_FILTER_MULTIPLIER}x avg {avg_spread:.2f}"
            )
        return True, ""

    def check_time_restrictions(
        self, upcoming_events: list | None = None
    ) -> tuple[bool, str]:
        """Log high-impact news windows but allow trading to continue.

        Previously this method blocked all trades during high-impact event
        windows.  Now it only logs a warning so the agent can still take
        high-confidence setups.
        """
        if upcoming_events:
            for event in upcoming_events:
                if self._is_high_impact_window(event):
                    log.warning(
                        "circuit_breaker.high_impact_event_nearby",
                        event=event.get("title", "unknown"),
                        hint="Trading allowed — reduce size if needed",
                    )

        return True, ""

    # ---- state mutation helpers --------------------------------------------

    def record_trade_result(self, symbol: str, is_win: bool) -> None:
        """Update streak counters after a trade closes."""
        if is_win:
            self.symbol_loss_streaks[symbol] = 0
            self.total_loss_streak = 0
            log.info(
                "circuit_breaker.win_recorded",
                symbol=symbol,
            )
            return

        # Loss
        self.symbol_loss_streaks[symbol] = (
            self.symbol_loss_streaks.get(symbol, 0) + 1
        )
        self.total_loss_streak += 1

        log.warning(
            "circuit_breaker.loss_recorded",
            symbol=symbol,
            symbol_streak=self.symbol_loss_streaks[symbol],
            total_streak=self.total_loss_streak,
        )

        # Per-symbol cooldown
        if self.symbol_loss_streaks[symbol] >= self.SYMBOL_LOSS_STREAK_LIMIT:
            cooldown_until = datetime.now(tz=timezone.utc) + timedelta(
                hours=self.SYMBOL_COOLDOWN_HOURS
            )
            self.symbol_cooldowns[symbol] = cooldown_until
            log.warning(
                "circuit_breaker.symbol_cooldown_activated",
                symbol=symbol,
                until=cooldown_until.isoformat(),
            )

        # Global pause
        if self.total_loss_streak >= self.TOTAL_LOSS_STREAK_LIMIT:
            self.trading_paused_until = datetime.now(
                tz=timezone.utc
            ) + timedelta(hours=self.TOTAL_COOLDOWN_HOURS)
            log.warning(
                "circuit_breaker.global_pause_activated",
                until=self.trading_paused_until.isoformat(),
            )

    def emergency_shutdown(self) -> None:
        """Close all positions, disable the agent, and log a critical alert."""
        log.critical("circuit_breaker.EMERGENCY_SHUTDOWN")

        try:
            from agent.execution.mt5_executor import close_all_positions

            close_all_positions()
            log.critical(
                "circuit_breaker.all_positions_closed_by_emergency"
            )
        except Exception:
            log.exception(
                "circuit_breaker.failed_to_close_positions_in_emergency"
            )

        self.agent_disabled = True

    def update_peak_balance(self, balance: float) -> None:
        """Track the highest balance seen for drawdown calculation."""
        if balance > self.peak_balance:
            self.peak_balance = balance

    def is_friday_close(self) -> bool:
        """Return True if it is Friday 3 PM or later Eastern Time."""
        now_est = datetime.now(tz=_EST)
        return now_est.weekday() == 4 and now_est.hour >= 15

    def reset_daily_state(self) -> None:
        """Reset state that should not carry across trading days.

        Called at the start of each trading day.
        """
        self.symbol_loss_streaks.clear()
        self.total_loss_streak = 0
        self.symbol_cooldowns.clear()
        self.trading_paused_until = None
        # peak_balance and agent_disabled intentionally persist
        log.info("circuit_breaker.daily_state_reset")

    # ---- private helpers ---------------------------------------------------

    @staticmethod
    def _is_high_impact_window(event: dict) -> bool:
        """Return True if *now* falls inside a high-impact event window.

        Expected ``event`` keys:
        - ``impact``: ``"high"`` to be considered.
        - ``datetime``: ISO-8601 string **or** :class:`datetime` of the event.
        - ``pre_minutes`` (optional): minutes before the event to block
          (default 15).
        - ``post_minutes`` (optional): minutes after the event to block
          (default 15).
        """
        if str(event.get("impact", "")).lower() != "high":
            return False

        event_dt = event.get("datetime")
        if event_dt is None:
            return False

        if isinstance(event_dt, str):
            # Parse ISO-8601, tolerate missing tz → assume UTC
            try:
                event_dt = datetime.fromisoformat(event_dt)
            except ValueError:
                return False

        if event_dt.tzinfo is None:
            event_dt = event_dt.replace(tzinfo=timezone.utc)

        pre = timedelta(minutes=event.get("pre_minutes", 15))
        post = timedelta(minutes=event.get("post_minutes", 15))
        now = datetime.now(tz=timezone.utc)

        return (event_dt - pre) <= now <= (event_dt + post)


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------
circuit_breaker = CircuitBreaker()
