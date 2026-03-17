"""
Event-driven backtester for the XAUUSD trading agent.

Simulates the full signal-to-execution pipeline bar by bar on M3 data,
applying HTF bias, M3/M5 signal scoring, ATR-based SL/TP, ratchet
stop-loss management, and position sizing.  Produces a comprehensive
results dictionary consumed by the report generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from xauusd_agent.brain.htf_analyzer import HTFBiasEngine
from xauusd_agent.brain.signal_engine import M3M5SignalEngine
from xauusd_agent.execution.lot_calculator import compute_sl_tp
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# XAUUSD constants
# ---------------------------------------------------------------------------

_PIP_SIZE: float = 0.1      # 1 pip = $0.10 price movement
_PIP_VALUE: float = 10.0    # USD per pip per standard lot
_DEFAULT_RISK_PCT: float = 1.0


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """Record for a single backtest trade."""

    ticket: int
    direction: str  # "BUY" or "SELL"
    open_time: pd.Timestamp
    open_price: float
    lot_size: float
    sl_price: float
    tp_price: float
    initial_risk_usd: float

    # Filled on close
    close_time: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    profit_usd: float = 0.0
    profit_pips: float = 0.0
    profit_R: float = 0.0
    close_reason: str = ""

    # Ratchet bookkeeping
    ratchet_triggered: bool = False
    ratchet_max_R: float = 0.0
    profit_locked_usd: float = 0.0
    max_profit_usd: float = 0.0

    # Context at entry
    htf_bias_score: float = 0.0
    regime: str = ""
    is_counter_trend: bool = False
    signal_score: float = 0.0


# ---------------------------------------------------------------------------
# ATR helpers (standalone, no MT5 dependency)
# ---------------------------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int) -> float | None:
    """Compute the latest ATR value from OHLC data."""
    if df is None or len(df) < period + 1:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(span=period, adjust=False).mean()
    val = atr_series.iloc[-1]
    return None if pd.isna(val) else float(val)


def _compute_sl_tp_backtest(
    m3: pd.DataFrame,
    m5: pd.DataFrame,
    direction: str,
    is_counter_trend: bool,
    regime: str,
    settings: dict | None = None,
) -> tuple[float, float, float] | None:
    """Backtest-local SL/TP computation (no MT5 dependency).

    Mirrors the logic in ``lot_calculator.compute_sl_tp``.
    Returns ``(sl_pips, tp_pips, atr_value)`` or ``None``.
    """
    defaults = {
        "sl_mult_trend": 1.2,
        "sl_mult_counter": 0.8,
        "sl_mult_volatile": 1.5,
        "tp_ratio_trend": 2.2,
        "tp_ratio_counter": 1.5,
        "tp_ratio_volatile": 2.5,
        "sl_min_pips": 8.0,
        "sl_max_pips": 35.0,
    }

    def _s(key: str):
        if settings and key in settings:
            return settings[key]
        return defaults[key]

    atr7_m3 = _compute_atr(m3, 7)
    atr14_m5 = _compute_atr(m5, 14)

    if atr7_m3 is None or atr14_m5 is None:
        return None

    atr_value = (atr7_m3 + atr14_m5) / 2.0
    atr_pips = atr_value / _PIP_SIZE

    regime_upper = regime.upper()
    if regime_upper == "HIGH_VOLATILE":
        sl_mult = _s("sl_mult_volatile")
    elif is_counter_trend:
        sl_mult = _s("sl_mult_counter")
    else:
        sl_mult = _s("sl_mult_trend")

    sl_pips = atr_pips * sl_mult

    sl_max = _s("sl_max_pips")
    if sl_pips > sl_max:
        return None

    sl_pips = max(sl_pips, _s("sl_min_pips"))

    if regime_upper == "HIGH_VOLATILE":
        tp_ratio = _s("tp_ratio_volatile")
    elif is_counter_trend:
        tp_ratio = _s("tp_ratio_counter")
    else:
        tp_ratio = _s("tp_ratio_trend")

    tp_pips = sl_pips * tp_ratio
    return sl_pips, tp_pips, atr_value


def _calculate_lot_backtest(
    balance: float,
    risk_pct: float,
    sl_pips: float,
    is_counter_trend: bool,
    regime: str,
    settings: dict | None = None,
) -> tuple[float, float]:
    """Backtest-local lot calculator (no MT5 dependency).

    Returns ``(lot_size, risk_usd)``.
    """
    defaults = {
        "counter_trend_lot_mult": 0.50,
        "high_volatile_lot_mult": 0.75,
        "low_volatile_lot_mult": 1.10,
        "max_lot_cap": 10.0,
    }

    def _s(key: str):
        if settings and key in settings:
            return settings[key]
        return defaults[key]

    risk_usd = balance * (risk_pct / 100.0)
    if sl_pips <= 0 or _PIP_VALUE <= 0:
        return 0.0, 0.0

    lot = risk_usd / (sl_pips * _PIP_VALUE)

    if is_counter_trend:
        lot *= _s("counter_trend_lot_mult")

    regime_upper = regime.upper()
    if regime_upper == "HIGH_VOLATILE":
        lot *= _s("high_volatile_lot_mult")
    elif regime_upper == "LOW_VOLATILE":
        lot *= _s("low_volatile_lot_mult")

    max_cap = _s("max_lot_cap")
    lot = max(0.01, min(lot, max_cap))
    lot = round(lot, 2)

    return lot, risk_usd


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class XAUUSDBacktester:
    """Simulates the XAUUSD agent strategy on historical data."""

    # Ratchet schedule: (R-multiple reached, lock specification)
    RATCHET: list[tuple[float, str | float]] = [
        (0.50, "breakeven"),
        (1.00, 0.30),
        (1.50, 0.50),
        (2.00, 0.65),
        (2.50, 0.75),
        (3.00, 0.85),
    ]

    def __init__(
        self,
        settings: dict,
        initial_balance: float = 10_000.0,
    ) -> None:
        self.settings = settings
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve: list[dict] = []
        self.trades: list[BacktestTrade] = []
        self.open_trades: list[BacktestTrade] = []
        self.trade_counter: int = 0

        self.htf_engine = HTFBiasEngine(settings.get("htf_bias", {}))
        self.signal_engine = M3M5SignalEngine(settings.get("signals", {}))

        self._risk_pct: float = settings.get("risk_pct", _DEFAULT_RISK_PCT)
        self._max_open: int = settings.get("max_open_trades", 3)
        self._min_bar_gap: int = settings.get("min_bar_gap", 5)
        self._bars_since_trade: int = 0

    # ------------------------------------------------------------------ #
    #  Main backtest loop                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        all_tf_data: dict[str, pd.DataFrame],
        spread_pips: float = 2.5,
    ) -> dict:
        """Execute the backtest over all M3 bars.

        For each M3 bar the loop:

        1. Checks open trades for SL/TP hits and applies ratchet updates.
        2. Builds the feature snapshot visible at this point in time.
        3. Recomputes HTF bias when a new H1 bar closes.
        4. Scores BUY/SELL via the M3/M5 signal engine.
        5. Opens a trade if the signal exceeds the entry threshold.
        6. Records the equity snapshot.

        Parameters
        ----------
        all_tf_data:
            Dict of ``{tf_name: DataFrame}`` for M3, M5, M15, H1, H4, D1.
        spread_pips:
            Simulated spread in pips applied to entry prices.

        Returns
        -------
        dict
            Full results dictionary (see :meth:`get_results`).
        """
        m3 = all_tf_data.get("M3", pd.DataFrame())
        m5 = all_tf_data.get("M5", pd.DataFrame())
        m15 = all_tf_data.get("M15", pd.DataFrame())
        h1 = all_tf_data.get("H1", pd.DataFrame())
        h4 = all_tf_data.get("H4", pd.DataFrame())
        d1 = all_tf_data.get("D1", pd.DataFrame())

        if m3.empty:
            logger.error("M3 data is empty; cannot run backtest")
            return self.get_results()

        # Minimum lookback bars before trading begins
        warmup = max(200, len(m3) // 100)
        if len(m3) <= warmup:
            logger.error(
                "Not enough M3 bars (need > %d, got %d)", warmup, len(m3),
            )
            return self.get_results()

        # Current HTF bias (refreshed on H1 close)
        htf_bias: dict = {
            "score": 0.0,
            "direction": "NEUTRAL",
            "trade_longs": True,
            "trade_shorts": True,
            "d1": 0.0,
            "h4": 0.0,
            "h1": 0.0,
            "m15_at_support": False,
            "m15_at_resist": False,
        }

        regime = "NORMAL"
        last_h1_refresh: pd.Timestamp | None = None

        logger.info(
            "Starting backtest: %d M3 bars, warmup=%d, balance=%.2f",
            len(m3), warmup, self.balance,
        )

        for i in range(warmup, len(m3)):
            current_time = m3.index[i]
            bar = m3.iloc[i]

            # --- 1. Update open trades (ratchet + SL/TP) -----------------
            self._update_open_trades(bar, current_time)

            # --- 2. Slice data visible at this point ---------------------
            m3_slice = m3.iloc[max(0, i - 500) : i + 1]
            m5_slice = self._slice_up_to(m5, current_time, 500)
            m15_slice = self._slice_up_to(m15, current_time, 300)
            h1_slice = self._slice_up_to(h1, current_time, 300)
            h4_slice = self._slice_up_to(h4, current_time, 200)
            d1_slice = self._slice_up_to(d1, current_time, 250)

            # --- 3. Refresh HTF bias on new H1 close ---------------------
            most_recent_h1 = self._most_recent_before(h1, current_time)
            if most_recent_h1 is not None and most_recent_h1 != last_h1_refresh:
                last_h1_refresh = most_recent_h1
                if (
                    not d1_slice.empty
                    and not h4_slice.empty
                    and not h1_slice.empty
                ):
                    try:
                        htf_bias = self.htf_engine.get_bias({
                            "D1": d1_slice,
                            "H4": h4_slice,
                            "H1": h1_slice,
                            "M15": m15_slice,
                        })
                    except Exception:
                        logger.debug(
                            "HTF bias computation failed at %s",
                            current_time, exc_info=True,
                        )

                # Derive regime from ATR ratio
                regime = self._detect_regime(m5_slice)

            # --- 4. Signal scoring ---------------------------------------
            self._bars_since_trade += 1

            if (
                len(self.open_trades) >= self._max_open
                or self._bars_since_trade < self._min_bar_gap
                or m3_slice.empty
                or m5_slice.empty
                or m15_slice.empty
                or len(m3_slice) < 50
                or len(m5_slice) < 30
            ):
                self._record_equity(current_time)
                continue

            try:
                signal = self.signal_engine.get_signal(
                    {"M3": m3_slice, "M5": m5_slice, "M15": m15_slice},
                    htf_bias,
                    spread_pips,
                )
            except Exception:
                logger.debug(
                    "Signal scoring failed at %s", current_time, exc_info=True,
                )
                self._record_equity(current_time)
                continue

            # --- 5. Open trade if signal fires ---------------------------
            action = signal.get("action", "HOLD")
            is_ct = signal.get("is_counter_trend", False)

            if action in ("BUY", "SELL"):
                # Gate: respect HTF directional filter
                if action == "BUY" and not htf_bias.get("trade_longs", True):
                    action = "HOLD"
                elif action == "SELL" and not htf_bias.get("trade_shorts", True):
                    action = "HOLD"

            if action in ("BUY", "SELL"):
                sl_tp = _compute_sl_tp_backtest(
                    m3_slice, m5_slice,
                    direction=action,
                    is_counter_trend=is_ct,
                    regime=regime,
                    settings=self.settings.get("lot_calc", {}),
                )
                if sl_tp is not None:
                    sl_pips, tp_pips, _ = sl_tp
                    self._open_trade(
                        direction=action,
                        price=float(bar["close"]),
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        signal=signal,
                        htf_bias=htf_bias,
                        regime=regime,
                        spread_pips=spread_pips,
                        bar_time=current_time,
                    )

            # --- 6. Record equity ----------------------------------------
            self._record_equity(current_time)

        # Close any remaining open trades at last bar's close
        if not m3.empty:
            last_bar = m3.iloc[-1]
            last_time = m3.index[-1]
            for trade in list(self.open_trades):
                self._close_trade(
                    trade, float(last_bar["close"]),
                    "end_of_backtest", bar_time=last_time,
                )

        logger.info(
            "Backtest complete: %d trades, final balance=%.2f",
            len(self.trades), self.balance,
        )
        return self.get_results()

    # ------------------------------------------------------------------ #
    #  SL / TP checking                                                    #
    # ------------------------------------------------------------------ #

    def _check_sl_tp(
        self,
        trade: BacktestTrade,
        bar: pd.Series,
        bar_time: pd.Timestamp,
        pip_size: float = _PIP_SIZE,
    ) -> bool:
        """Check if SL or TP was hit on this bar.

        When both SL and TP are within the bar's range, proximity to the
        bar open price determines which was hit first.

        Returns ``True`` if the trade was closed.
        """
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_open = float(bar["open"])

        if trade.direction == "BUY":
            tp_hit = bar_high >= trade.tp_price
            sl_hit = bar_low <= trade.sl_price
        else:  # SELL
            tp_hit = bar_low <= trade.tp_price
            sl_hit = bar_high >= trade.sl_price

        if tp_hit and sl_hit:
            # Both hit on same bar -- determine order via open proximity
            dist_to_tp = abs(trade.tp_price - bar_open)
            dist_to_sl = abs(trade.sl_price - bar_open)

            if dist_to_sl <= dist_to_tp:
                self._close_trade(
                    trade, trade.sl_price, "stop_loss", bar_time=bar_time,
                )
            else:
                self._close_trade(
                    trade, trade.tp_price, "take_profit", bar_time=bar_time,
                )
            return True

        if tp_hit:
            self._close_trade(
                trade, trade.tp_price, "take_profit", bar_time=bar_time,
            )
            return True

        if sl_hit:
            self._close_trade(
                trade, trade.sl_price, "stop_loss", bar_time=bar_time,
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Ratchet logic                                                       #
    # ------------------------------------------------------------------ #

    def _apply_ratchet(
        self,
        trade: BacktestTrade,
        current_price: float,
        pip_size: float = _PIP_SIZE,
        pip_value: float = _PIP_VALUE,
    ) -> None:
        """Apply ratchet SL logic to a trade.

        Mirrors ``RatchetSLManager.compute_new_sl`` but operates directly
        on :class:`BacktestTrade` fields.  The stop-loss is only ever
        moved toward profit (monotonic).
        """
        if trade.direction == "BUY":
            profit_pips = (current_price - trade.open_price) / pip_size
        else:
            profit_pips = (trade.open_price - current_price) / pip_size

        profit_usd = profit_pips * pip_value * trade.lot_size
        if trade.initial_risk_usd <= 0:
            return
        profit_r = profit_usd / trade.initial_risk_usd

        # Track maximum profit seen
        if profit_usd > trade.max_profit_usd:
            trade.max_profit_usd = profit_usd
        if profit_r > trade.ratchet_max_R:
            trade.ratchet_max_R = profit_r

        # Find highest ratchet level reached
        active_level: tuple[float, str | float] | None = None
        for r_threshold, lock_spec in self.RATCHET:
            if profit_r >= r_threshold:
                active_level = (r_threshold, lock_spec)

        if active_level is None:
            return

        _, lock_spec = active_level

        # Compute target SL
        if lock_spec == "breakeven":
            target_sl = trade.open_price
        else:
            locked_pips = profit_pips * float(lock_spec)
            if trade.direction == "BUY":
                target_sl = trade.open_price + locked_pips * pip_size
            else:
                target_sl = trade.open_price - locked_pips * pip_size

        # Enforce monotonic direction (SL only moves toward profit)
        if trade.direction == "BUY":
            if target_sl <= trade.sl_price:
                return
        else:
            if target_sl >= trade.sl_price:
                return

        trade.sl_price = round(target_sl, 5)
        trade.ratchet_triggered = True

        # Compute locked profit in USD
        if trade.direction == "BUY":
            locked_profit_pips = (trade.sl_price - trade.open_price) / pip_size
        else:
            locked_profit_pips = (trade.open_price - trade.sl_price) / pip_size
        trade.profit_locked_usd = max(
            0.0, locked_profit_pips * pip_value * trade.lot_size,
        )

    # ------------------------------------------------------------------ #
    #  Trade management                                                    #
    # ------------------------------------------------------------------ #

    def _open_trade(
        self,
        direction: str,
        price: float,
        sl_pips: float,
        tp_pips: float,
        signal: dict,
        htf_bias: dict,
        regime: str,
        spread_pips: float,
        bar_time: pd.Timestamp | None = None,
    ) -> None:
        """Open a new backtest trade.

        Position size is computed from the current balance and risk
        percentage.  Spread is applied to the entry price (added for
        BUY, subtracted for SELL).
        """
        # Apply spread to entry
        spread_price = spread_pips * _PIP_SIZE
        if direction == "BUY":
            entry_price = price + spread_price
        else:
            entry_price = price - spread_price

        is_ct = signal.get("is_counter_trend", False)

        # Lot sizing
        lot, risk_usd = _calculate_lot_backtest(
            self.balance,
            self._risk_pct,
            sl_pips,
            is_ct,
            regime,
            self.settings.get("lot_calc", {}),
        )
        if lot <= 0:
            return

        # Compute SL / TP prices
        sl_dist = sl_pips * _PIP_SIZE
        tp_dist = tp_pips * _PIP_SIZE

        if direction == "BUY":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        self.trade_counter += 1
        score = (
            signal.get("buy_score", 0.0)
            if direction == "BUY"
            else signal.get("sell_score", 0.0)
        )

        trade = BacktestTrade(
            ticket=self.trade_counter,
            direction=direction,
            open_time=bar_time or pd.Timestamp.now(tz="UTC"),
            open_price=entry_price,
            lot_size=lot,
            sl_price=round(sl_price, 5),
            tp_price=round(tp_price, 5),
            initial_risk_usd=risk_usd,
            htf_bias_score=htf_bias.get("score", 0.0),
            regime=regime,
            is_counter_trend=is_ct,
            signal_score=score,
        )

        self.open_trades.append(trade)
        self._bars_since_trade = 0

        logger.debug(
            "Opened %s #%d @ %.5f  SL=%.5f  TP=%.5f  lot=%.2f",
            direction, trade.ticket, entry_price,
            trade.sl_price, trade.tp_price, lot,
        )

    def _close_trade(
        self,
        trade: BacktestTrade,
        price: float,
        reason: str,
        bar_time: pd.Timestamp | None = None,
        pip_size: float = _PIP_SIZE,
        pip_value: float = _PIP_VALUE,
    ) -> None:
        """Close a trade and update the account balance."""
        trade.close_price = price
        trade.close_time = bar_time or pd.Timestamp.now(tz="UTC")
        trade.close_reason = reason

        if trade.direction == "BUY":
            trade.profit_pips = (price - trade.open_price) / pip_size
        else:
            trade.profit_pips = (trade.open_price - price) / pip_size

        trade.profit_usd = trade.profit_pips * pip_value * trade.lot_size

        if trade.initial_risk_usd > 0:
            trade.profit_R = trade.profit_usd / trade.initial_risk_usd
        else:
            trade.profit_R = 0.0

        self.balance += trade.profit_usd

        # Move from open to closed
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.trades.append(trade)

        logger.debug(
            "Closed %s #%d @ %.5f  P&L=%.2f (%.2fR)  reason=%s",
            trade.direction, trade.ticket, price,
            trade.profit_usd, trade.profit_R, reason,
        )

    # ------------------------------------------------------------------ #
    #  Per-bar update of open trades                                       #
    # ------------------------------------------------------------------ #

    def _update_open_trades(
        self, bar: pd.Series, current_time: pd.Timestamp,
    ) -> None:
        """Check SL/TP and apply ratchet for every open trade."""
        current_price = float(bar["close"])

        for trade in list(self.open_trades):
            # Apply ratchet first (may tighten SL)
            self._apply_ratchet(trade, current_price)

            # Update max unrealised profit
            if trade.direction == "BUY":
                unrealised_pips = (current_price - trade.open_price) / _PIP_SIZE
            else:
                unrealised_pips = (trade.open_price - current_price) / _PIP_SIZE
            unrealised_usd = unrealised_pips * _PIP_VALUE * trade.lot_size
            if unrealised_usd > trade.max_profit_usd:
                trade.max_profit_usd = unrealised_usd

            # Check SL / TP
            self._check_sl_tp(trade, bar, current_time)

    # ------------------------------------------------------------------ #
    #  Results computation                                                 #
    # ------------------------------------------------------------------ #

    def get_results(self) -> dict:
        """Compute comprehensive backtest metrics.

        Returns
        -------
        dict
            Keys include:

            - ``total_trades``, ``winners``, ``losers``, ``win_rate``
            - ``profit_factor``
            - ``total_pnl``, ``max_drawdown_pct``
            - ``sharpe_ratio`` (annualized on daily returns)
            - ``avg_winner_R``, ``avg_loser_R``
            - ``ratchet_saves``, ``ratchet_saves_pct``
            - ``avg_locked_profit``, ``max_profit_given_back``
            - ``win_rate_by_htf_bias``, ``win_rate_by_regime``
            - ``signal_score_vs_outcome``
            - ``equity_curve``
            - ``trades``
        """
        total = len(self.trades)
        winners = [t for t in self.trades if t.profit_usd > 0]
        losers = [t for t in self.trades if t.profit_usd <= 0]

        win_rate = len(winners) / total * 100 if total > 0 else 0.0

        gross_profit = sum(t.profit_usd for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.profit_usd for t in losers)) if losers else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        total_pnl = sum(t.profit_usd for t in self.trades)

        # Max drawdown from equity curve
        max_dd_pct = self._compute_max_drawdown()

        # Sharpe ratio (annualized)
        sharpe = self._compute_sharpe()

        # Average R
        avg_winner_R = (
            float(np.mean([t.profit_R for t in winners])) if winners else 0.0
        )
        avg_loser_R = (
            float(np.mean([t.profit_R for t in losers])) if losers else 0.0
        )

        # Ratchet statistics
        ratchet_trades = [t for t in self.trades if t.ratchet_triggered]
        ratchet_saves = len([
            t for t in ratchet_trades
            if t.profit_usd >= 0 and t.close_reason == "stop_loss"
        ])
        ratchet_saves_pct = (
            ratchet_saves / total * 100 if total > 0 else 0.0
        )
        avg_locked = (
            float(np.mean([t.profit_locked_usd for t in ratchet_trades]))
            if ratchet_trades else 0.0
        )
        max_given_back = float(max(
            (t.max_profit_usd - t.profit_usd for t in self.trades),
            default=0.0,
        ))

        # Win rate by HTF bias bucket
        win_rate_by_htf = self._win_rate_by_field("htf_bias_score", buckets=[
            (-1.0, -0.5, "STRONG_BEAR"),
            (-0.5, -0.15, "BEAR"),
            (-0.15, 0.15, "NEUTRAL"),
            (0.15, 0.5, "BULL"),
            (0.5, 1.01, "STRONG_BULL"),
        ])

        # Win rate by regime
        win_rate_by_regime = self._win_rate_grouped("regime")

        # Signal score vs outcome (10-point buckets)
        signal_buckets = self._signal_score_vs_outcome()

        # Monthly P&L
        monthly_pnl: dict[str, float] = {}
        for t in self.trades:
            if t.close_time is not None:
                month_key = str(t.close_time)[:7]  # YYYY-MM
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + t.profit_usd

        return {
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 3),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "avg_winner_R": round(avg_winner_R, 3),
            "avg_loser_R": round(avg_loser_R, 3),
            "ratchet_saves": ratchet_saves,
            "ratchet_saves_pct": round(ratchet_saves_pct, 2),
            "avg_locked_profit": round(avg_locked, 2),
            "max_profit_given_back": round(max_given_back, 2),
            "win_rate_by_htf_bias": win_rate_by_htf,
            "win_rate_by_regime": win_rate_by_regime,
            "signal_score_vs_outcome": signal_buckets,
            "equity_curve": self.equity_curve,
            "trades": [self._trade_to_dict(t) for t in self.trades],
            "initial_balance": self.initial_balance,
            "final_balance": round(self.balance, 2),
            "monthly_pnl": monthly_pnl,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _record_equity(self, timestamp: pd.Timestamp) -> None:
        """Append a point to the equity curve."""
        # Mark-to-market for open trades (approximate using current balance)
        self.equity_curve.append({
            "time": timestamp,
            "balance": round(self.balance, 2),
            "equity": round(self.balance, 2),
        })

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown percentage from the equity curve."""
        if not self.equity_curve:
            return 0.0
        equities = [e["equity"] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _compute_sharpe(self, periods_per_year: int = 252) -> float:
        """Annualized Sharpe ratio from daily equity returns."""
        if len(self.equity_curve) < 2:
            return 0.0

        eq_df = pd.DataFrame(self.equity_curve)
        eq_df["time"] = pd.to_datetime(eq_df["time"])
        eq_df = eq_df.set_index("time")

        # Resample to daily for a meaningful Sharpe
        daily = eq_df["equity"].resample("1D").last().dropna()
        if len(daily) < 2:
            return 0.0

        returns = daily.pct_change().dropna()
        if returns.std() == 0:
            return 0.0

        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    def _win_rate_by_field(
        self, field: str, buckets: list[tuple],
    ) -> dict[str, dict]:
        """Compute win rate within score buckets."""
        result: dict[str, dict] = {}
        for lo, hi, label in buckets:
            subset = [
                t for t in self.trades
                if lo <= getattr(t, field, 0.0) < hi
            ]
            wins = sum(1 for t in subset if t.profit_usd > 0)
            total = len(subset)
            result[label] = {
                "trades": total,
                "win_rate": round(wins / total * 100, 2) if total > 0 else 0.0,
            }
        return result

    def _win_rate_grouped(self, field: str) -> dict[str, dict]:
        """Win rate grouped by a categorical trade attribute."""
        groups: dict[str, list[BacktestTrade]] = {}
        for t in self.trades:
            key = getattr(t, field, "UNKNOWN")
            groups.setdefault(key, []).append(t)

        result: dict[str, dict] = {}
        for key, trades_list in groups.items():
            wins = sum(1 for t in trades_list if t.profit_usd > 0)
            result[key] = {
                "trades": len(trades_list),
                "win_rate": round(
                    wins / len(trades_list) * 100, 2,
                ) if trades_list else 0.0,
            }
        return result

    def _signal_score_vs_outcome(self) -> list[dict]:
        """Bucket signal scores into 10-point bands and compute avg outcome."""
        buckets: list[dict] = []
        for lo in range(0, 100, 10):
            hi = lo + 10
            subset = [
                t for t in self.trades
                if lo <= t.signal_score < hi
            ]
            if not subset:
                buckets.append({
                    "range": f"{lo}-{hi}",
                    "trades": 0,
                    "avg_R": 0.0,
                    "win_rate": 0.0,
                })
                continue
            avg_r = float(np.mean([t.profit_R for t in subset]))
            wins = sum(1 for t in subset if t.profit_usd > 0)
            buckets.append({
                "range": f"{lo}-{hi}",
                "trades": len(subset),
                "avg_R": round(avg_r, 3),
                "win_rate": round(wins / len(subset) * 100, 2),
            })
        return buckets

    @staticmethod
    def _trade_to_dict(trade: BacktestTrade) -> dict:
        """Serialise a BacktestTrade to a plain dict."""
        return {
            "ticket": trade.ticket,
            "direction": trade.direction,
            "open_time": str(trade.open_time) if trade.open_time else None,
            "open_price": trade.open_price,
            "lot_size": trade.lot_size,
            "sl_price": trade.sl_price,
            "tp_price": trade.tp_price,
            "close_time": str(trade.close_time) if trade.close_time else None,
            "close_price": trade.close_price,
            "profit_usd": round(trade.profit_usd, 2),
            "profit_pips": round(trade.profit_pips, 2),
            "profit_R": round(trade.profit_R, 3),
            "close_reason": trade.close_reason,
            "ratchet_triggered": trade.ratchet_triggered,
            "ratchet_max_R": round(trade.ratchet_max_R, 3),
            "profit_locked_usd": round(trade.profit_locked_usd, 2),
            "max_profit_usd": round(trade.max_profit_usd, 2),
            "htf_bias_score": trade.htf_bias_score,
            "regime": trade.regime,
            "is_counter_trend": trade.is_counter_trend,
            "signal_score": trade.signal_score,
        }

    @staticmethod
    def _slice_up_to(
        df: pd.DataFrame, up_to: pd.Timestamp, max_rows: int,
    ) -> pd.DataFrame:
        """Return the last ``max_rows`` rows of *df* up to *up_to*."""
        if df.empty:
            return df
        mask = df.index <= up_to
        sliced = df.loc[mask]
        if len(sliced) > max_rows:
            sliced = sliced.iloc[-max_rows:]
        return sliced

    @staticmethod
    def _most_recent_before(
        df: pd.DataFrame, ts: pd.Timestamp,
    ) -> pd.Timestamp | None:
        """Return the index of the most recent row <= *ts*."""
        if df.empty:
            return None
        mask = df.index <= ts
        if not mask.any():
            return None
        return df.index[mask][-1]

    @staticmethod
    def _detect_regime(m5: pd.DataFrame) -> str:
        """Simple ATR-based regime detection on M5 data."""
        if m5.empty or len(m5) < 21:
            return "NORMAL"

        high = m5["high"]
        low = m5["low"]
        close = m5["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_7 = float(tr.iloc[-7:].mean())
        atr_20 = float(tr.iloc[-20:].mean())

        ratio = atr_7 / atr_20 if atr_20 > 0 else 1.0

        if ratio > 1.5:
            return "HIGH_VOLATILE"
        elif ratio < 0.65:
            return "LOW_VOLATILE"
        return "NORMAL"
