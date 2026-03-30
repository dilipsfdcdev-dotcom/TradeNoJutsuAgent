"""Event-driven backtesting engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from agent.signals.indicators import compute_all_indicators
from agent.signals.patterns import detect_all_patterns, PatternSignal
from agent.signals.market_structure import detect_trend

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    direction: str  # "buy" or "sell"
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    lot_size: float
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float | None = None
    exit_reason: str | None = None  # "tp", "sl", "time", "signal"


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    metrics: dict
    equity_curve: list[dict]
    symbol: str
    start_date: datetime
    end_date: datetime
    mode: str
    initial_balance: float
    final_balance: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Minimum number of candles required before we start generating signals
# (need enough history for EMA-50 + RSI-14 warm-up).
_WARMUP_BARS = 60

# Minimum bars between consecutive entries to avoid over-trading.
_MIN_BARS_BETWEEN_TRADES = 5


class BacktestEngine:
    """Walk-forward, event-driven backtester.

    Parameters
    ----------
    symbol : str
        Instrument being tested (e.g. ``"XAUUSD"``).
    initial_balance : float
        Starting account equity.
    risk_pct : float
        Percentage of balance risked per trade.
    slippage_pips : float
        Maximum random slippage added on entry (0 to 2 * slippage_pips).
    spread_cost_pips : float
        Simulated spread cost added to entry.
    preferred_rr : float
        Reward-to-risk ratio used for take-profit placement.
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        initial_balance: float = 10_000.0,
        risk_pct: float = 1.0,
        slippage_pips: float = 1.0,
        spread_cost_pips: float = 1.0,
        preferred_rr: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pips = slippage_pips
        self.spread_cost_pips = spread_cost_pips
        self.preferred_rr = preferred_rr

        # pip size: gold is 0.01, most FX is 0.0001
        upper = symbol.upper()
        if "XAU" in upper or "GOLD" in upper:
            self._pip = 0.01
        elif "XAG" in upper or "SILVER" in upper:
            self._pip = 0.001
        elif "BTC" in upper:
            self._pip = 0.01
        elif "JPY" in upper:
            self._pip = 0.01
        else:
            self._pip = 0.0001

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df_1m: pd.DataFrame,
        df_3m: pd.DataFrame | None = None,
        mode: str = "rules",
    ) -> BacktestResult:
        """Run the back-test.

        Parameters
        ----------
        df_1m : pd.DataFrame
            1-minute OHLCV candles (the execution timeframe).
        df_3m : pd.DataFrame | None
            3-minute OHLCV candles used for higher-TF trend context.
            If *None*, a synthetic 3-minute frame is resampled from *df_1m*.
        mode : str
            ``"rules"`` for pure indicator/pattern logic (no API calls).
            ``"ai"`` to call Claude for each candidate signal (expensive).
        """
        if df_3m is None:
            df_3m = self._resample_3m(df_1m)

        # Pre-compute indicators on the full frames (we will slice later).
        df_1m = df_1m.copy().reset_index(drop=True)
        df_3m = df_3m.copy().reset_index(drop=True)

        balance = self.initial_balance
        open_trades: list[BacktestTrade] = []
        closed_trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        last_entry_bar = -_MIN_BARS_BETWEEN_TRADES  # allow immediate first trade

        logger.info(
            "backtest_start",
            symbol=self.symbol,
            mode=mode,
            bars=len(df_1m),
            balance=balance,
        )

        for i in range(len(df_1m)):
            candle = df_1m.iloc[i]

            # --- 1. Check exits on this candle's high/low ---
            still_open: list[BacktestTrade] = []
            for trade in open_trades:
                closed = self._check_exits(trade, candle)
                if closed:
                    balance += closed.pnl  # type: ignore[operator]
                    closed_trades.append(closed)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # --- 2. Record equity ---
            unrealised = self._unrealised_pnl(open_trades, candle)
            equity_curve.append({
                "time": str(candle["time"]),
                "balance": round(balance, 2),
                "equity": round(balance + unrealised, 2),
                "open_trades": len(open_trades),
            })

            # --- 3. Generate entry signals (skip warm-up) ---
            if i < _WARMUP_BARS:
                continue
            if i - last_entry_bar < _MIN_BARS_BETWEEN_TRADES:
                continue
            if len(open_trades) >= 2:
                continue  # max 2 concurrent positions in backtest

            window_1m = df_1m.iloc[max(0, i - 99) : i + 1].copy().reset_index(drop=True)
            window_1m = compute_all_indicators(window_1m)

            # Higher-TF trend from 3-min data up to approximately this point in time
            candle_time = candle["time"]
            mask_3m = df_3m["time"] <= candle_time
            window_3m = df_3m.loc[mask_3m].tail(100).copy().reset_index(drop=True)

            trend = detect_trend(window_3m) if len(window_3m) > 20 else "ranging"
            patterns = detect_all_patterns(window_1m)
            recent_patterns = [p for p in patterns if p.candle_index >= len(window_1m) - 3]

            last_row = window_1m.iloc[-1]
            indicators = {
                "ema_9": last_row.get("ema_9"),
                "ema_21": last_row.get("ema_21"),
                "ema_50": last_row.get("ema_50"),
                "rsi": last_row.get("rsi"),
                "atr": last_row.get("atr"),
                "close": last_row["close"],
            }

            signal: dict | None = None
            if mode == "rules":
                signal = self._generate_rule_signal(indicators, recent_patterns, trend)
            elif mode == "ai":
                signal = self._generate_ai_signal(indicators, recent_patterns, trend)

            if signal is not None:
                trade = self._open_trade(signal, candle, balance)
                if trade is not None:
                    open_trades.append(trade)
                    last_entry_bar = i

        # --- Close remaining open trades at last candle ---
        if len(df_1m) > 0:
            last_candle = df_1m.iloc[-1]
            for trade in open_trades:
                trade.exit_price = float(last_candle["close"])
                trade.exit_time = last_candle["time"]
                trade.exit_reason = "time"
                trade.pnl = self._compute_pnl(trade)
                balance += trade.pnl
                closed_trades.append(trade)

        metrics = self._calculate_metrics(closed_trades, equity_curve)

        result = BacktestResult(
            trades=closed_trades,
            metrics=metrics,
            equity_curve=equity_curve,
            symbol=self.symbol,
            start_date=df_1m.iloc[0]["time"] if len(df_1m) > 0 else datetime.min,
            end_date=df_1m.iloc[-1]["time"] if len(df_1m) > 0 else datetime.min,
            mode=mode,
            initial_balance=self.initial_balance,
            final_balance=round(balance, 2),
        )

        logger.info(
            "backtest_complete",
            trades=len(closed_trades),
            final_balance=result.final_balance,
            net_pnl=round(result.final_balance - self.initial_balance, 2),
        )
        return result

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exits(self, trade: BacktestTrade, candle: pd.Series) -> BacktestTrade | None:
        """Check whether *candle* triggers SL or TP for *trade*.

        Returns the trade with exit fields populated, or *None* if still open.
        """
        high = float(candle["high"])
        low = float(candle["low"])
        time = candle["time"]

        if trade.direction == "buy":
            # Stop-loss hit
            if low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_time = time
                trade.exit_reason = "sl"
                trade.pnl = self._compute_pnl(trade)
                return trade
            # Take-profit hit
            if high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = time
                trade.exit_reason = "tp"
                trade.pnl = self._compute_pnl(trade)
                return trade
        else:  # sell
            if high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_time = time
                trade.exit_reason = "sl"
                trade.pnl = self._compute_pnl(trade)
                return trade
            if low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = time
                trade.exit_reason = "tp"
                trade.pnl = self._compute_pnl(trade)
                return trade

        return None

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_rule_signal(
        self,
        indicators: dict,
        patterns: list[PatternSignal],
        trend: str,
    ) -> dict | None:
        """Pure rule-based signal using EMA crossover, RSI, and pattern confirmation.

        Rules
        -----
        BUY when:
          - EMA-9 > EMA-21  (short-term momentum)
          - EMA-21 > EMA-50  (trend confirmation) OR trend == "bullish"
          - RSI < 70  (not overbought)
          - At least one: RSI < 30  OR  bullish pattern present
        SELL (mirror logic).

        SL = 1.5 * ATR from entry; TP = preferred_rr * SL distance.
        """
        ema9 = indicators.get("ema_9")
        ema21 = indicators.get("ema_21")
        ema50 = indicators.get("ema_50")
        rsi = indicators.get("rsi")
        atr = indicators.get("atr")
        close = indicators.get("close")

        # Need all indicators to be computed
        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [ema9, ema21, ema50, rsi, atr, close]):
            return None
        if atr <= 0:
            return None

        bullish_patterns = [
            p for p in patterns
            if p.type in ("bullish_engulfing", "bullish_pin_bar", "bullish_bos",
                          "bullish_order_block", "bullish_liquidity_sweep", "bullish_fvg")
        ]
        bearish_patterns = [
            p for p in patterns
            if p.type in ("bearish_engulfing", "bearish_pin_bar", "bearish_bos",
                          "bearish_order_block", "bearish_liquidity_sweep", "bearish_fvg")
        ]

        # --- BUY ---
        if ema9 > ema21 and (ema21 > ema50 or trend == "bullish") and rsi < 70:
            has_trigger = rsi < 30 or len(bullish_patterns) > 0
            if has_trigger:
                sl_dist = 1.5 * atr
                tp_dist = self.preferred_rr * sl_dist
                return {
                    "direction": "buy",
                    "sl_distance": sl_dist,
                    "tp_distance": tp_dist,
                    "reason": "ema_cross_buy",
                }

        # --- SELL ---
        if ema9 < ema21 and (ema21 < ema50 or trend == "bearish") and rsi > 30:
            has_trigger = rsi > 70 or len(bearish_patterns) > 0
            if has_trigger:
                sl_dist = 1.5 * atr
                tp_dist = self.preferred_rr * sl_dist
                return {
                    "direction": "sell",
                    "sl_distance": sl_dist,
                    "tp_distance": tp_dist,
                    "reason": "ema_cross_sell",
                }

        return None

    def _generate_ai_signal(
        self,
        indicators: dict,
        patterns: list[PatternSignal],
        trend: str,
    ) -> dict | None:
        """Call Claude to evaluate the setup. Expensive -- use sparingly."""
        try:
            import anthropic
            from agent.config import settings

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            pattern_strs = [f"{p.type} (strength={p.strength})" for p in patterns[:5]]
            prompt = (
                f"You are a professional scalping analyst.\n"
                f"Symbol: {self.symbol}\n"
                f"Trend (3m): {trend}\n"
                f"Indicators: {indicators}\n"
                f"Recent patterns: {pattern_strs}\n\n"
                f"Should we enter a trade? Respond with exactly one JSON object:\n"
                f'{{"action": "buy"|"sell"|"none", "confidence": 0.0-1.0, "reason": "..."}}\n'
                f"Only recommend buy/sell if confidence >= 0.7."
            )

            response = client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            import json
            text = response.content[0].text.strip()
            # Extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                if result.get("action") in ("buy", "sell") and result.get("confidence", 0) >= 0.7:
                    atr = indicators.get("atr", 1.0)
                    sl_dist = 1.5 * atr
                    tp_dist = self.preferred_rr * sl_dist
                    return {
                        "direction": result["action"],
                        "sl_distance": sl_dist,
                        "tp_distance": tp_dist,
                        "reason": f"ai: {result.get('reason', '')}",
                    }
        except Exception:
            logger.exception("ai_signal_error")
        return None

    # ------------------------------------------------------------------
    # Trade management helpers
    # ------------------------------------------------------------------

    def _open_trade(
        self, signal: dict, candle: pd.Series, balance: float
    ) -> BacktestTrade | None:
        """Create a BacktestTrade from a signal dict, applying slippage and spread."""
        direction = signal["direction"]
        sl_dist = signal["sl_distance"]
        tp_dist = signal["tp_distance"]
        close = float(candle["close"])

        # Random slippage 0-2 pips
        slippage = random.uniform(0, 2 * self.slippage_pips) * self._pip
        spread = self.spread_cost_pips * self._pip

        if direction == "buy":
            entry = close + spread / 2 + slippage
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = close - spread / 2 - slippage
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Position sizing: risk_pct of balance / SL distance
        risk_amount = balance * (self.risk_pct / 100.0)
        if sl_dist <= 0:
            return None
        lot_size = round(risk_amount / (sl_dist / self._pip), 2)
        if lot_size <= 0:
            return None

        return BacktestTrade(
            symbol=self.symbol,
            direction=direction,
            entry_price=round(entry, 5),
            entry_time=candle["time"],
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            lot_size=lot_size,
        )

    def _compute_pnl(self, trade: BacktestTrade) -> float:
        """Compute P&L in account currency for a closed trade."""
        if trade.exit_price is None:
            return 0.0
        if trade.direction == "buy":
            pip_delta = (trade.exit_price - trade.entry_price) / self._pip
        else:
            pip_delta = (trade.entry_price - trade.exit_price) / self._pip
        return round(pip_delta * trade.lot_size, 2)

    def _unrealised_pnl(
        self, open_trades: list[BacktestTrade], candle: pd.Series
    ) -> float:
        """Sum unrealised P&L for open trades at the current candle close."""
        total = 0.0
        close = float(candle["close"])
        for t in open_trades:
            if t.direction == "buy":
                delta = (close - t.entry_price) / self._pip
            else:
                delta = (t.entry_price - close) / self._pip
            total += delta * t.lot_size
        return round(total, 2)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[dict],
    ) -> dict:
        """Compute performance metrics from closed trades and equity curve."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_rr": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "avg_trade_duration_minutes": 0.0,
            }

        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown from equity curve
        max_dd_pct = 0.0
        peak = self.initial_balance
        for point in equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

        # Sharpe ratio (annualised, assuming 1-min bars ~252 trading days * 1440 min)
        if len(pnls) > 1:
            arr = np.array(pnls, dtype=float)
            mean_ret = np.mean(arr)
            std_ret = np.std(arr, ddof=1)
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # Average realised R:R
        rr_ratios: list[float] = []
        for t in trades:
            if t.pnl is not None and t.pnl != 0:
                sl_dist = abs(t.entry_price - t.stop_loss)
                if sl_dist > 0:
                    rr_ratios.append(t.pnl / (sl_dist / self._pip * t.lot_size))
        avg_rr = float(np.mean(rr_ratios)) if rr_ratios else 0.0

        # Average trade duration
        durations: list[float] = []
        for t in trades:
            if t.entry_time is not None and t.exit_time is not None:
                delta = pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)
                durations.append(delta.total_seconds() / 60.0)
        avg_duration = float(np.mean(durations)) if durations else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "avg_rr": round(avg_rr, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2),
            "max_win": round(max(pnls), 2) if pnls else 0.0,
            "max_loss": round(min(pnls), 2) if pnls else 0.0,
            "avg_trade_duration_minutes": round(avg_duration, 1),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_3m(df_1m: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-minute candles to 3-minute candles."""
        df = df_1m.copy()
        df = df.set_index("time")
        ohlcv = df.resample("3min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "max",
        }).dropna(subset=["open"])
        ohlcv = ohlcv.reset_index()
        return ohlcv
