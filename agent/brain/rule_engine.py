"""Rule Engine — makes ALL trade decisions deterministically.

No API calls. No randomness. Same inputs = same outputs.
Takes ML scores + MTF state -> produces exact trade plan.
"""

from dataclasses import dataclass, field
from typing import Optional
import structlog

from agent.config import settings
from agent.brain.risk_manager import calculate_lot_size, get_adjusted_risk_pct

logger = structlog.get_logger()


@dataclass
class TradePlan:
    symbol: str
    direction: str            # "buy" | "sell"
    entry_price: float
    entry_type: str           # "market" | "limit"
    stop_loss: float
    take_profit_1: float      # 15M target (50% close)
    take_profit_2: float | None  # 1H target (remaining)
    lot_size: float
    rr_ratio_tp1: float
    rr_ratio_tp2: float | None
    confidence_composite: float
    reasoning_factors: list[str] = field(default_factory=list)


class RuleEngine:
    """Pure math. No opinions. No LLM calls."""

    def __init__(self):
        self.min_rr_tp1 = settings.MIN_RR_RATIO  # 1.5
        self.xgb_weight = 0.4
        self.lstm_weight = 0.4
        self.mtf_weight = 0.2

    def compute_trade(self, symbol: str, mtf_state, xgb: dict, lstm: dict,
                       account: dict, current_tick: dict) -> TradePlan | None:
        """
        Produce exact trade plan or None. Pure deterministic logic.
        """
        # Step 1: Direction — from MTF state, cross-checked with LSTM
        h1_bias = getattr(mtf_state, "h1_bias", None)
        if not h1_bias or h1_bias == "neutral":
            # Try timeframes dict fallback
            tf = getattr(mtf_state, "timeframes", {})
            h1_bias = tf.get("1H", {}).get("trend", "neutral")

        if h1_bias == "neutral":
            return None

        lstm_dir = lstm.get("direction", "wait")
        if lstm_dir == "wait":
            return None
        if lstm_dir != h1_bias.replace("ish", ""):
            # LSTM says buy but MTF says bearish, or vice versa
            if h1_bias == "bullish" and lstm_dir != "buy":
                return None
            if h1_bias == "bearish" and lstm_dir != "sell":
                return None

        direction = "buy" if h1_bias == "bullish" else "sell"

        # Step 2: Entry price
        m1_signal = getattr(mtf_state, "m1_entry_signal", None)
        if m1_signal and hasattr(m1_signal, "entry_price"):
            entry = m1_signal.entry_price
        else:
            entry = current_tick.get("ask" if direction == "buy" else "bid", 0)

        if entry <= 0:
            return None

        # Step 3: Stop loss from structure
        m1_atr = 0
        tf = getattr(mtf_state, "timeframes", {})
        m1_atr = tf.get("1M", {}).get("atr", 0) or 0
        if m1_atr <= 0:
            # Fallback: use a default based on symbol
            m1_atr = {"XAUUSD": 2.0, "BTCUSD": 50.0, "XAGUSD": 0.05}.get(symbol, 2.0)

        sl_candidates = []
        if m1_signal and hasattr(m1_signal, "sl_price") and m1_signal.sl_price > 0:
            sl_candidates.append(m1_signal.sl_price)

        # Default SL at 1.5x ATR
        if direction == "buy":
            sl_default = entry - m1_atr * 1.5
            sl_candidates.append(sl_default)
            sl = max(sl_candidates) if sl_candidates else sl_default  # tightest

            # Validate SL distance
            sl_dist = entry - sl
            if sl_dist < m1_atr * 0.5:
                sl = entry - m1_atr  # minimum distance
            if sl_dist > m1_atr * 3.0:
                return None  # too wide for scalp
        else:
            sl_default = entry + m1_atr * 1.5
            sl_candidates.append(sl_default)
            sl = min(sl_candidates) if sl_candidates else sl_default  # tightest

            sl_dist = sl - entry
            if sl_dist < m1_atr * 0.5:
                sl = entry + m1_atr
            if sl_dist > m1_atr * 3.0:
                return None

        # Step 4: Take profits from structure levels
        h1_levels = getattr(mtf_state, "h1_key_levels", [])
        m15_pools = getattr(mtf_state, "m15_liquidity_pools", [])

        sl_distance = abs(entry - sl)

        if direction == "buy":
            # TP1: nearest target above entry
            tp1 = entry + sl_distance * self.min_rr_tp1  # minimum R:R
            # Check for 15M/1H levels
            for level in sorted(_extract_prices(list(m15_pools) + list(h1_levels))):
                if level > entry + sl_distance:
                    tp1 = level
                    break
            tp2 = entry + sl_distance * 3.0  # default TP2 at 3:1
            for level in sorted(_extract_prices(h1_levels)):
                if level > tp1:
                    tp2 = level
                    break
        else:
            tp1 = entry - sl_distance * self.min_rr_tp1
            for level in sorted(_extract_prices(list(m15_pools) + list(h1_levels)), reverse=True):
                if level < entry - sl_distance:
                    tp1 = level
                    break
            tp2 = entry - sl_distance * 3.0
            for level in sorted(_extract_prices(h1_levels), reverse=True):
                if level < tp1:
                    tp2 = level
                    break

        # Step 5: Validate R:R
        rr1 = abs(tp1 - entry) / sl_distance if sl_distance > 0 else 0
        rr2 = abs(tp2 - entry) / sl_distance if sl_distance > 0 else 0

        if rr1 < self.min_rr_tp1:
            return None
        if rr2 < 2.5:
            tp2 = None
            rr2 = None

        # Step 6: Compute lot size
        composite = (xgb["score"] * self.xgb_weight +
                     lstm["confidence"] * self.lstm_weight +
                     (getattr(mtf_state, "confluence_score", 50) / 100) * self.mtf_weight)

        base_risk = settings.MAX_RISK_PER_TRADE_PCT
        if composite > 0.80:
            risk_pct = base_risk * 1.2
        elif composite > 0.70:
            risk_pct = base_risk
        else:
            risk_pct = base_risk * 0.7

        # Streak/drawdown adjustments
        consec_losses = account.get("consecutive_losses", 0)
        if consec_losses >= 3:
            risk_pct *= 0.5
        daily_loss_pct = abs(account.get("daily_pnl_pct", 0))
        if daily_loss_pct > settings.MAX_DAILY_LOSS_PCT * 0.6:
            risk_pct *= 0.5

        balance = account.get("balance", 0)
        lot_size = calculate_lot_size(symbol, balance, entry, sl, risk_pct)

        if lot_size <= 0:
            return None

        # Build reasoning factors
        factors = self._list_factors(mtf_state, xgb, lstm, rr1, rr2)

        return TradePlan(
            symbol=symbol, direction=direction,
            entry_price=round(entry, 5), entry_type="market",
            stop_loss=round(sl, 5),
            take_profit_1=round(tp1, 5),
            take_profit_2=round(tp2, 5) if tp2 else None,
            lot_size=round(lot_size, 2),
            rr_ratio_tp1=round(rr1, 2),
            rr_ratio_tp2=round(rr2, 2) if rr2 else None,
            confidence_composite=round(composite, 4),
            reasoning_factors=factors,
        )

    def _list_factors(self, mtf, xgb, lstm, rr1, rr2) -> list[str]:
        """Deterministic list of why this trade was taken."""
        factors = []
        h1_bias = getattr(mtf, "h1_bias", "")
        h1_struct = getattr(mtf, "h1_structure", "")
        h1_ema = getattr(mtf, "h1_ema_stack", "")
        factors.append(f"1H {h1_bias} ({h1_struct}, {h1_ema})")

        m15_poi = getattr(mtf, "m15_poi", None)
        if m15_poi:
            factors.append(f"15M {getattr(m15_poi, 'zone_type', 'zone')} at {getattr(m15_poi, 'low', '?')}-{getattr(m15_poi, 'high', '?')}")

        m3_conf = getattr(mtf, "m3_confirmed", False)
        m3_mom = getattr(mtf, "m3_momentum", "flat")
        factors.append(f"3M {'confirmed' if m3_conf else 'unconfirmed'}: {m3_mom}")

        m1_sig = getattr(mtf, "m1_entry_signal", None)
        if m1_sig:
            factors.append(f"1M trigger: {getattr(m1_sig, 'trigger_pattern', 'unknown')}")

        factors.append(f"XGB: {xgb.get('score', 0):.2f}")
        factors.append(f"LSTM: {lstm.get('confidence', 0):.2f} {lstm.get('direction', '?')}, {lstm.get('regime', '?')}")
        factors.append(f"Confluence: {getattr(mtf, 'confluence_score', 0)}/100")
        factors.append(f"R:R {rr1:.1f}:1" + (f" / {rr2:.1f}:1" if rr2 else ""))
        return factors


def _extract_prices(levels: list) -> list[float]:
    """Safely extract numeric prices from a mixed list of levels."""
    prices = []
    for level in levels:
        if level is None:
            continue
        try:
            if hasattr(level, "price"):
                prices.append(float(level.price))
            else:
                prices.append(float(level))
        except (TypeError, ValueError):
            continue
    return prices
