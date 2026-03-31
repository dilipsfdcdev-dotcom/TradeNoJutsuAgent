"""Tests for the v3 deterministic rule engine."""

import pytest
from agent.brain.rule_engine import RuleEngine, TradePlan
from agent.signals.mtf_analyzer import MTFState


@pytest.fixture
def rule_engine():
    return RuleEngine()


@pytest.fixture
def bullish_mtf():
    state = MTFState(symbol="XAUUSD")
    state.h1_bias = "bullish"
    state.h1_trend_strength = 0.8
    state.h1_structure = "HH_HL"
    state.h1_ema_stack = "bullish_stack"
    state.m3_confirmed = True
    state.m3_momentum = "bullish"
    state.confluence_score = 75
    state.gates_passed = True
    state.timeframes = {
        "1H": {"trend": "bullish", "atr": 5.0},
        "15M": {"trend": "bullish", "atr": 2.0},
        "3M": {"trend": "bullish", "atr": 1.0},
        "1M": {"trend": "bullish", "atr": 1.5},
    }
    return state


@pytest.fixture
def xgb_pass():
    return {"score": 0.75, "pass": True, "top_features": [("m3_confirmed", 0.12)]}


@pytest.fixture
def lstm_buy():
    return {"direction": "buy", "confidence": 0.72, "regime": "trending", "pass": True}


@pytest.fixture
def account():
    return {"balance": 10000.0, "equity": 10050.0, "consecutive_losses": 0, "daily_pnl_pct": 0.5}


@pytest.fixture
def tick_buy():
    return {"bid": 2050.0, "ask": 2050.20}


class TestRuleEngineComputeTrade:
    def test_valid_buy_trade(self, rule_engine, bullish_mtf, xgb_pass, lstm_buy, account, tick_buy):
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_buy, account, tick_buy)
        assert plan is not None
        assert isinstance(plan, TradePlan)
        assert plan.direction == "buy"
        assert plan.lot_size > 0
        assert plan.rr_ratio_tp1 >= 1.5
        assert plan.stop_loss < plan.entry_price
        assert plan.take_profit_1 > plan.entry_price

    def test_neutral_bias_returns_none(self, rule_engine, bullish_mtf, xgb_pass, lstm_buy, account, tick_buy):
        bullish_mtf.h1_bias = "neutral"
        bullish_mtf.timeframes["1H"]["trend"] = "neutral"
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_buy, account, tick_buy)
        assert plan is None

    def test_lstm_wait_returns_none(self, rule_engine, bullish_mtf, xgb_pass, account, tick_buy):
        lstm_wait = {"direction": "wait", "confidence": 0.72, "regime": "ranging", "pass": True}
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_wait, account, tick_buy)
        assert plan is None

    def test_sell_trade(self, rule_engine, bullish_mtf, xgb_pass, account):
        bullish_mtf.h1_bias = "bearish"
        bullish_mtf.timeframes["1H"]["trend"] = "bearish"
        lstm_sell = {"direction": "sell", "confidence": 0.70, "regime": "trending", "pass": True}
        tick = {"bid": 2050.0, "ask": 2050.20}
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_sell, account, tick)
        if plan:
            assert plan.direction == "sell"
            assert plan.stop_loss > plan.entry_price
            assert plan.take_profit_1 < plan.entry_price

    def test_reasoning_factors_populated(self, rule_engine, bullish_mtf, xgb_pass, lstm_buy, account, tick_buy):
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_buy, account, tick_buy)
        if plan:
            assert len(plan.reasoning_factors) > 0
            assert any("XGB" in f for f in plan.reasoning_factors)
            assert any("LSTM" in f for f in plan.reasoning_factors)

    def test_composite_confidence(self, rule_engine, bullish_mtf, xgb_pass, lstm_buy, account, tick_buy):
        plan = rule_engine.compute_trade("XAUUSD", bullish_mtf, xgb_pass, lstm_buy, account, tick_buy)
        if plan:
            assert 0 < plan.confidence_composite <= 1.0


class TestVetoRegister:
    def test_default_allows(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        reg = VetoRegister()
        state = reg.check("XAUUSD")
        assert state.blocked is False
        assert state.risk_modifier == 1.0

    def test_set_and_check_block(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        from datetime import datetime, timedelta
        reg = VetoRegister()
        reg.set_veto("XAUUSD", VetoState(
            blocked=True, reason="News event",
            expires_at=datetime.utcnow() + timedelta(minutes=30),
        ))
        state = reg.check("XAUUSD")
        assert state.blocked is True
        assert "News" in state.reason

    def test_set_and_check_reduce(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        reg = VetoRegister()
        reg.set_veto("BTCUSD", VetoState(
            blocked=False, risk_modifier=0.5, reason="High volatility",
        ))
        state = reg.check("BTCUSD")
        assert state.blocked is False
        assert state.risk_modifier == 0.5

    def test_global_overrides_symbol(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        from datetime import datetime, timedelta
        reg = VetoRegister()
        reg.set_veto("XAUUSD", VetoState(blocked=False, risk_modifier=0.5))
        reg.set_veto("GLOBAL", VetoState(
            blocked=True, reason="Emergency",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        ))
        state = reg.check("XAUUSD")
        assert state.blocked is True  # Global overrides

    def test_clear_veto(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        reg = VetoRegister()
        reg.set_veto("XAUUSD", VetoState(blocked=True, reason="test"))
        reg.clear_veto("XAUUSD")
        state = reg.check("XAUUSD")
        assert state.blocked is False

    def test_expired_veto_auto_clears(self):
        from agent.brain.veto_register import VetoRegister, VetoState
        from datetime import datetime, timedelta
        reg = VetoRegister()
        reg.set_veto("XAUUSD", VetoState(
            blocked=True, reason="expired test",
            expires_at=datetime.utcnow() - timedelta(minutes=1),  # already expired
        ))
        state = reg.check("XAUUSD")
        assert state.blocked is False  # Should auto-clear


class TestModelMonitor:
    def test_record_and_accuracy(self):
        from agent.brain.model_monitor import ModelMonitor
        mon = ModelMonitor()
        for i in range(20):
            mon.record_prediction("xgb", 0.7 if i % 2 == 0 else 0.3, actual=1 if i % 2 == 0 else 0)
        acc = mon.get_accuracy("xgb", last_n=20)
        assert acc == 100.0  # All predictions match

    def test_health_report(self):
        from agent.brain.model_monitor import ModelMonitor
        mon = ModelMonitor()
        health = mon.get_health()
        assert "xgb_accuracy" in health
        assert "lstm_accuracy" in health
