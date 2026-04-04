"""Tests for agent.backtest.engine."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from agent.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade


@pytest.fixture
def backtest_candles():
    """Generate 500 1-minute XAUUSD candles with enough bars for warm-up and signals."""
    np.random.seed(123)
    n = 500
    dates = [datetime(2024, 1, 15, 8, 0) + timedelta(minutes=i) for i in range(n)]
    base_price = 2050.0

    prices = [base_price]
    for i in range(n - 1):
        # Add slight trend and mean-reversion to increase signal variety
        drift = 0.01 * np.sin(i / 50.0)
        prices.append(prices[-1] + drift + np.random.randn() * 1.5)

    data = []
    for dt, price in zip(dates, prices):
        o = price
        h = price + abs(np.random.randn()) * 2.0
        l = price - abs(np.random.randn()) * 2.0
        c = price + np.random.randn() * 1.0
        v = int(np.random.uniform(100, 1000))
        data.append({
            "time": dt,
            "open": o,
            "high": max(o, h, c),
            "low": min(o, l, c),
            "close": c,
            "volume": v,
            "spread": 20,
        })

    return pd.DataFrame(data)


@pytest.fixture
def engine():
    return BacktestEngine(
        symbol="XAUUSD",
        initial_balance=10_000.0,
        risk_pct=1.0,
        slippage_pips=0.5,
        spread_cost_pips=0.5,
        preferred_rr=2.0,
    )


class TestBacktestRuns:
    def test_backtest_completes_without_error(self, engine, backtest_candles):
        """The backtest engine should run to completion without raising."""
        result = engine.run(backtest_candles, mode="rules")
        assert result is not None

    def test_backtest_with_short_data(self, engine):
        """Backtest should handle data shorter than warm-up period gracefully."""
        np.random.seed(99)
        n = 30  # fewer than _WARMUP_BARS (60)
        dates = [datetime(2024, 1, 15, 10, 0) + timedelta(minutes=i) for i in range(n)]
        data = []
        for dt in dates:
            p = 2050.0 + np.random.randn()
            data.append({
                "time": dt,
                "open": p,
                "high": p + 1,
                "low": p - 1,
                "close": p + 0.5,
                "volume": 500,
                "spread": 20,
            })
        df = pd.DataFrame(data)
        result = engine.run(df, mode="rules")
        assert result is not None
        # With only 30 bars no trades should be opened (warm-up is 60)
        assert result.metrics["total_trades"] == 0


class TestBacktestResultStructure:
    def test_result_has_required_fields(self, engine, backtest_candles):
        """BacktestResult should contain all expected fields."""
        result = engine.run(backtest_candles, mode="rules")

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.equity_curve, list)
        assert result.symbol == "XAUUSD"
        assert isinstance(result.start_date, datetime)
        assert isinstance(result.end_date, datetime)
        assert result.mode == "rules"
        assert result.initial_balance == 10_000.0
        assert isinstance(result.final_balance, float)

    def test_equity_curve_structure(self, engine, backtest_candles):
        """Each equity curve point should have time, balance, equity, open_trades."""
        result = engine.run(backtest_candles, mode="rules")

        assert len(result.equity_curve) == len(backtest_candles)
        point = result.equity_curve[0]
        assert "time" in point
        assert "balance" in point
        assert "equity" in point
        assert "open_trades" in point

    def test_trades_are_backtest_trade_instances(self, engine, backtest_candles):
        """All trades in the result should be BacktestTrade dataclass instances."""
        result = engine.run(backtest_candles, mode="rules")

        for trade in result.trades:
            assert isinstance(trade, BacktestTrade)
            assert trade.symbol == "XAUUSD"
            assert trade.direction in ("buy", "sell")
            assert trade.entry_price > 0
            assert trade.lot_size > 0
            assert trade.exit_price is not None
            assert trade.pnl is not None
            assert trade.exit_reason in ("tp", "sl", "time", "signal")


class TestBacktestMetrics:
    def test_metrics_keys_present(self, engine, backtest_candles):
        """Metrics dict should contain all expected performance keys."""
        result = engine.run(backtest_candles, mode="rules")
        expected_keys = [
            "total_trades",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "sharpe_ratio",
            "avg_rr",
            "total_pnl",
            "avg_pnl",
            "max_win",
            "max_loss",
            "avg_trade_duration_minutes",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing metric: {key}"

    def test_metrics_types(self, engine, backtest_candles):
        """Metric values should be numeric."""
        result = engine.run(backtest_candles, mode="rules")
        m = result.metrics

        assert isinstance(m["total_trades"], int)
        assert isinstance(m["win_rate"], float)
        assert isinstance(m["profit_factor"], float)
        assert isinstance(m["max_drawdown_pct"], float)
        assert isinstance(m["sharpe_ratio"], float)
        assert isinstance(m["total_pnl"], float)

    def test_win_rate_in_range(self, engine, backtest_candles):
        """Win rate should be between 0 and 100."""
        result = engine.run(backtest_candles, mode="rules")
        assert 0 <= result.metrics["win_rate"] <= 100

    def test_max_drawdown_non_negative(self, engine, backtest_candles):
        """Max drawdown should be >= 0."""
        result = engine.run(backtest_candles, mode="rules")
        assert result.metrics["max_drawdown_pct"] >= 0

    def test_total_pnl_matches_balance_change(self, engine, backtest_candles):
        """Total P&L from metrics should approximately match final - initial balance."""
        result = engine.run(backtest_candles, mode="rules")
        balance_change = result.final_balance - result.initial_balance
        # Allow small floating-point tolerance
        assert abs(result.metrics["total_pnl"] - balance_change) < 0.1

    def test_no_trades_metrics(self):
        """When no trades occur, metrics should have sensible defaults."""
        eng = BacktestEngine(symbol="XAUUSD", initial_balance=10_000.0)
        # Only 10 bars -- not enough for warm-up
        dates = [datetime(2024, 1, 15, 10, 0) + timedelta(minutes=i) for i in range(10)]
        data = [{
            "time": dt,
            "open": 2050.0,
            "high": 2051.0,
            "low": 2049.0,
            "close": 2050.0,
            "volume": 500,
            "spread": 20,
        } for dt in dates]
        df = pd.DataFrame(data)

        result = eng.run(df, mode="rules")
        assert result.metrics["total_trades"] == 0
        assert result.metrics["win_rate"] == 0.0
        assert result.metrics["profit_factor"] == 0.0
        assert result.final_balance == 10_000.0
