"""Tests for agent.execution.circuit_breaker."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agent.execution.circuit_breaker import CircuitBreaker


@pytest.fixture
def cb():
    """Return a fresh CircuitBreaker instance for each test."""
    return CircuitBreaker()


@pytest.fixture
def normal_account():
    return {"balance": 10000.0, "equity": 10000.0}


class TestCanTradeNormal:
    def test_allows_trade_under_normal_conditions(self, cb, normal_account):
        cb.update_peak_balance(10000.0)
        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, reason = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is True
        assert reason == ""


class TestDailyLossExceeded:
    def test_blocks_on_daily_loss_exceeded(self, cb, normal_account):
        """Daily loss exceeding MAX_DAILY_LOSS_PCT should block trading."""
        cb.update_peak_balance(10000.0)
        # 3% default limit; -350 on 10k balance = 3.5%
        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, reason = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=-350.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is False
        assert "Daily loss" in reason


class TestConsecutiveLosses:
    def test_blocks_after_symbol_loss_streak(self, cb, normal_account):
        """After 3 consecutive losses on a symbol, should block that symbol."""
        cb.update_peak_balance(10000.0)
        # Record 3 losses to trigger per-symbol cooldown
        cb.record_trade_result("XAUUSD", is_win=False)
        cb.record_trade_result("XAUUSD", is_win=False)
        cb.record_trade_result("XAUUSD", is_win=False)

        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, reason = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is False
        assert "cooldown" in reason.lower() or "consecutive losses" in reason.lower()

    def test_other_symbol_not_blocked(self, cb, normal_account):
        """Losses on XAUUSD should not block BTCUSD."""
        cb.update_peak_balance(10000.0)
        cb.record_trade_result("XAUUSD", is_win=False)
        cb.record_trade_result("XAUUSD", is_win=False)
        cb.record_trade_result("XAUUSD", is_win=False)

        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, _ = cb.can_trade(
                symbol="BTCUSD",
                current_spread=50.0,
                avg_spread=50.0,
                daily_pnl=0.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is True


class TestDrawdownEmergency:
    def test_triggers_shutdown_on_max_drawdown(self, cb):
        """Drawdown >= MAX_DRAWDOWN_PCT should trigger emergency shutdown."""
        cb.update_peak_balance(10000.0)
        # Equity at 9100 = 9% drawdown (limit is 8%)
        account = {"balance": 10000.0, "equity": 9100.0}

        with patch.object(cb, "is_friday_close", return_value=False), \
             patch("agent.execution.circuit_breaker.CircuitBreaker.emergency_shutdown") as mock_shutdown:
            allowed, reason = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=account,
                upcoming_events=[],
            )
        assert allowed is False
        assert "EMERGENCY" in reason
        mock_shutdown.assert_called_once()

    def test_no_shutdown_on_normal_drawdown(self, cb):
        cb.update_peak_balance(10000.0)
        account = {"balance": 10000.0, "equity": 9500.0}  # 5% drawdown, under 8%

        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, _ = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=account,
                upcoming_events=[],
            )
        assert allowed is True


class TestSpreadCheck:
    def test_blocks_on_high_spread(self, cb, normal_account):
        """Spread exceeding SPREAD_FILTER_MULTIPLIER * avg should block."""
        cb.update_peak_balance(10000.0)
        # Default multiplier is 2.0; current_spread 5.0 > 2.0 * 2.0 = 4.0
        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, reason = cb.can_trade(
                symbol="XAUUSD",
                current_spread=5.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is False
        assert "spread" in reason.lower()

    def test_allows_normal_spread(self, cb, normal_account):
        cb.update_peak_balance(10000.0)
        with patch.object(cb, "is_friday_close", return_value=False):
            allowed, _ = cb.can_trade(
                symbol="XAUUSD",
                current_spread=2.0,
                avg_spread=2.0,
                daily_pnl=0.0,
                account_info=normal_account,
                upcoming_events=[],
            )
        assert allowed is True


class TestRecordTradeResult:
    def test_win_resets_streaks(self, cb):
        cb.record_trade_result("XAUUSD", is_win=False)
        cb.record_trade_result("XAUUSD", is_win=False)
        assert cb.symbol_loss_streaks["XAUUSD"] == 2
        assert cb.total_loss_streak == 2

        cb.record_trade_result("XAUUSD", is_win=True)
        assert cb.symbol_loss_streaks["XAUUSD"] == 0
        assert cb.total_loss_streak == 0

    def test_loss_increments_streaks(self, cb):
        cb.record_trade_result("XAUUSD", is_win=False)
        assert cb.symbol_loss_streaks["XAUUSD"] == 1
        assert cb.total_loss_streak == 1

        cb.record_trade_result("XAUUSD", is_win=False)
        assert cb.symbol_loss_streaks["XAUUSD"] == 2
        assert cb.total_loss_streak == 2

    def test_global_pause_after_total_streak(self, cb):
        """5 total consecutive losses should trigger a global pause."""
        for _ in range(5):
            cb.record_trade_result("XAUUSD", is_win=False)
        assert cb.trading_paused_until is not None
        assert cb.trading_paused_until > datetime.now(tz=timezone.utc)
