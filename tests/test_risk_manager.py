"""Tests for agent.brain.risk_manager."""

import pytest
from unittest.mock import MagicMock

from agent.brain.risk_manager import (
    calculate_lot_size,
    get_adjusted_risk_pct,
    validate_rr_ratio,
)


class TestCalculateLotSizeXauusd:
    def test_basic_lot_size(self):
        """XAUUSD: 10k balance, 1% risk, 5$ SL distance -> reasonable lot."""
        lot = calculate_lot_size(
            symbol="XAUUSD",
            balance=10000.0,
            entry=2050.0,
            sl=2045.0,
            risk_pct=1.0,
        )
        assert lot > 0
        # risk_amount = 100, sl_dist = 5.0
        # pip_value = 100 * 0.01 = 1.0
        # lot = 100 / (5.0 / 0.01 * 1.0) = 100 / 500 = 0.20
        assert lot == 0.20

    def test_zero_sl_distance_returns_zero(self):
        lot = calculate_lot_size("XAUUSD", 10000.0, 2050.0, 2050.0, 1.0)
        assert lot == 0.0

    def test_negative_balance_returns_zero(self):
        lot = calculate_lot_size("XAUUSD", -100.0, 2050.0, 2045.0, 1.0)
        assert lot == 0.0


class TestCalculateLotSizeBtcusd:
    def test_basic_lot_size(self):
        """BTCUSD uses simplified formula: risk_amount / sl_distance."""
        lot = calculate_lot_size(
            symbol="BTCUSD",
            balance=10000.0,
            entry=42000.0,
            sl=41500.0,
            risk_pct=1.0,
        )
        assert lot > 0
        # risk_amount = 100, sl_dist = 500
        # lot = 100 / 500 = 0.20
        assert lot == 0.20

    def test_unknown_non_crypto_symbol_returns_zero(self):
        lot = calculate_lot_size("UNKNOWN", 10000.0, 100.0, 95.0, 1.0)
        assert lot == 0.0


class TestAdjustedRiskLosingStreak:
    def test_losing_streak_reduces_risk(self):
        """3+ consecutive losses should reduce risk to 0.5%."""
        account = {"balance": 10000.0, "peak_balance": 10000.0}
        ctx = MagicMock(atr=None, avg_atr=None, session=None, symbol=None)
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=3, consecutive_wins=0, confidence=50)
        assert risk == 0.5

    def test_no_losing_streak_uses_base(self):
        account = {"balance": 10000.0, "peak_balance": 10000.0}
        ctx = MagicMock(atr=None, avg_atr=None, session=None, symbol=None)
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 1.0  # base risk from settings


class TestAdjustedRiskHighVolatility:
    def test_high_volatility_reduces_risk(self):
        """ATR > 1.5x avg_atr should reduce risk to 0.5%."""
        account = {"balance": 10000.0, "peak_balance": 10000.0}
        ctx = MagicMock(atr=3.0, avg_atr=1.5, session="london", symbol="XAUUSD")
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 0.5

    def test_normal_volatility_no_reduction(self):
        account = {"balance": 10000.0, "peak_balance": 10000.0}
        ctx = MagicMock(atr=1.5, avg_atr=1.5, session="london", symbol="XAUUSD")
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 1.0


class TestAdjustedRiskDrawdownStop:
    def test_drawdown_above_8_pct_stops_trading(self):
        """Drawdown >= 8% should return 0.0 (stop trading)."""
        account = {"balance": 9100.0, "peak_balance": 10000.0}  # 9% drawdown
        ctx = MagicMock(atr=None, avg_atr=None, session=None, symbol=None)
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 0.0

    def test_drawdown_above_5_pct_reduces_risk(self):
        """Drawdown > 5% but < 8% should reduce risk to 0.5%."""
        account = {"balance": 9400.0, "peak_balance": 10000.0}  # 6% drawdown
        ctx = MagicMock(atr=None, avg_atr=None, session=None, symbol=None)
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 0.5

    def test_no_drawdown_uses_base(self):
        account = {"balance": 10000.0, "peak_balance": 10000.0}
        ctx = MagicMock(atr=None, avg_atr=None, session=None, symbol=None)
        risk = get_adjusted_risk_pct(account, ctx, consecutive_losses=0, consecutive_wins=0, confidence=50)
        assert risk == 1.0


class TestValidateRrRatio:
    def test_valid_rr(self):
        # Buy: entry 2050, sl 2045 (risk 5), tp 2060 (reward 10) -> R:R = 2.0
        assert validate_rr_ratio(entry=2050.0, sl=2045.0, tp=2060.0) is True

    def test_invalid_rr(self):
        # Buy: entry 2050, sl 2045 (risk 5), tp 2052 (reward 2) -> R:R = 0.4
        assert validate_rr_ratio(entry=2050.0, sl=2045.0, tp=2052.0) is False

    def test_zero_risk_returns_false(self):
        assert validate_rr_ratio(entry=2050.0, sl=2050.0, tp=2060.0) is False

    def test_custom_min_rr(self):
        # R:R = 1.0, min_rr = 1.0 -> should pass
        assert validate_rr_ratio(entry=2050.0, sl=2045.0, tp=2055.0, min_rr=1.0) is True
        # R:R = 1.0, min_rr = 1.5 -> should fail
        assert validate_rr_ratio(entry=2050.0, sl=2045.0, tp=2055.0, min_rr=1.5) is False
