"""Tests for agent.brain.trade_planner."""

import pytest
from unittest.mock import MagicMock

from agent.brain.trade_planner import plan_trade, validate_trade, Order
from agent.brain.analyst import TradeDecision
from agent.signals.market_structure import MarketContext


def _make_decision(
    action="buy",
    confidence=80,
    entry_price=2050.0,
    stop_loss=2045.0,
    take_profit=2060.0,
    symbol="XAUUSD",
) -> TradeDecision:
    return TradeDecision(
        action=action,
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning="test setup",
        risk_score=5,
        symbol=symbol,
        timeframe="M1",
    )


def _make_context(
    trend="bullish",
    session="london",
    atr=2.0,
    volatility_rank="normal",
) -> MarketContext:
    return MarketContext(
        trend=trend,
        key_levels=[2040.0, 2060.0],
        session=session,
        atr=atr,
        volatility_rank=volatility_rank,
    )


def _make_account(balance=10000.0, peak_balance=10000.0):
    return {
        "balance": balance,
        "equity": balance,
        "peak_balance": peak_balance,
    }


class TestPlanValidTrade:
    def test_plan_valid_trade_returns_order(self):
        """A valid buy setup with good R:R should produce an Order."""
        decision = _make_decision(
            action="buy",
            confidence=80,
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2060.0,
        )
        context = _make_context()
        account = _make_account()

        order = plan_trade(
            decision=decision,
            context=context,
            account_info=account,
            open_positions=[],
            consecutive_losses=0,
            consecutive_wins=0,
            daily_loss_remaining=300.0,
        )

        assert order is not None
        assert isinstance(order, Order)
        assert order.symbol == "XAUUSD"
        assert order.direction == "buy"
        assert order.lot_size > 0
        assert order.entry_price == 2050.0
        assert order.stop_loss == 2045.0
        assert order.take_profit == 2060.0
        assert order.rr_planned == 2.0

    def test_plan_valid_sell_trade(self):
        """A valid sell setup should produce an Order."""
        decision = _make_decision(
            action="sell",
            entry_price=2050.0,
            stop_loss=2055.0,
            take_profit=2040.0,
        )
        context = _make_context(trend="bearish")
        account = _make_account()

        order = plan_trade(
            decision=decision,
            context=context,
            account_info=account,
            open_positions=[],
            consecutive_losses=0,
            consecutive_wins=0,
            daily_loss_remaining=300.0,
        )

        assert order is not None
        assert order.direction == "sell"
        assert order.stop_loss == 2055.0
        assert order.take_profit == 2040.0


class TestPlanInvalidRR:
    def test_plan_invalid_rr_returns_none(self):
        """A trade with R:R below minimum (1.5) should return None."""
        decision = _make_decision(
            action="buy",
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2052.0,  # reward=2, risk=5 -> R:R=0.4
        )
        context = _make_context()
        account = _make_account()

        order = plan_trade(
            decision=decision,
            context=context,
            account_info=account,
            open_positions=[],
            consecutive_losses=0,
            consecutive_wins=0,
            daily_loss_remaining=300.0,
        )

        assert order is None


class TestPlanMaxPositions:
    def test_plan_max_positions_returns_none(self):
        """When max open trades is reached, plan_trade should return None."""
        decision = _make_decision()
        context = _make_context()
        account = _make_account()

        # Create enough mock positions to hit the MAX_OPEN_TRADES limit (default 3)
        mock_positions = [MagicMock() for _ in range(3)]

        order = plan_trade(
            decision=decision,
            context=context,
            account_info=account,
            open_positions=mock_positions,
            consecutive_losses=0,
            consecutive_wins=0,
            daily_loss_remaining=300.0,
        )

        assert order is None

    def test_plan_wait_action_returns_none(self):
        """A 'wait' action should return None."""
        decision = _make_decision(action="wait")
        context = _make_context()
        account = _make_account()

        order = plan_trade(
            decision=decision,
            context=context,
            account_info=account,
            open_positions=[],
            consecutive_losses=0,
            consecutive_wins=0,
            daily_loss_remaining=300.0,
        )

        assert order is None


class TestValidateTradeBuy:
    def test_valid_buy_order(self):
        order = Order(
            symbol="XAUUSD",
            direction="buy",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2060.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is True
        assert reason == ""

    def test_buy_sl_above_entry_invalid(self):
        """For a buy, stop loss must be below entry."""
        order = Order(
            symbol="XAUUSD",
            direction="buy",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2055.0,  # SL above entry for buy = invalid
            take_profit=2060.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "sl_above_entry_for_buy"

    def test_buy_tp_below_entry_invalid(self):
        """For a buy, take profit must be above entry."""
        order = Order(
            symbol="XAUUSD",
            direction="buy",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2040.0,  # TP below entry for buy = invalid
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "tp_below_entry_for_buy"


class TestValidateTradeSell:
    def test_valid_sell_order(self):
        order = Order(
            symbol="XAUUSD",
            direction="sell",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2055.0,
            take_profit=2040.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is True
        assert reason == ""

    def test_sell_sl_below_entry_invalid(self):
        """For a sell, stop loss must be above entry."""
        order = Order(
            symbol="XAUUSD",
            direction="sell",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2045.0,  # SL below entry for sell = invalid
            take_profit=2040.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "sl_below_entry_for_sell"

    def test_sell_tp_above_entry_invalid(self):
        """For a sell, take profit must be below entry."""
        order = Order(
            symbol="XAUUSD",
            direction="sell",
            lot_size=0.20,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2055.0,
            take_profit=2060.0,  # TP above entry for sell = invalid
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "tp_above_entry_for_sell"

    def test_zero_lot_size_invalid(self):
        order = Order(
            symbol="XAUUSD",
            direction="buy",
            lot_size=0.0,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2060.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "lot_size_zero_or_negative"

    def test_excessive_lot_size_invalid(self):
        order = Order(
            symbol="XAUUSD",
            direction="buy",
            lot_size=150.0,
            entry_type="market",
            entry_price=2050.0,
            stop_loss=2045.0,
            take_profit=2060.0,
            magic_number=123456,
            comment="test",
            confidence=80,
            risk_pct=1.0,
            rr_planned=2.0,
        )
        valid, reason = validate_trade(order)
        assert valid is False
        assert reason == "lot_size_exceeds_max"
