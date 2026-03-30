"""
Dynamic position sizing based on account balance and volatility.

CORE FORMULA:
  risk_amount = balance * (risk_pct / 100)
  sl_distance = abs(entry - stop_loss)

  For forex (XAUUSD, XAGUSD):
    pip_value = contract_size * tick_size
    lot_size = risk_amount / (sl_distance / tick_size * pip_value)

  For crypto (BTCUSD):
    lot_size = risk_amount / sl_distance

DYNAMIC RISK ADJUSTMENT:
  Base risk: 1% per trade (from settings.MAX_RISK_PER_TRADE_PCT)

  Adjustments:
  - Winning streak (3+): keep at base (don't increase on euphoria)
  - Losing streak (3+): reduce to 0.5%
  - High volatility (ATR > 1.5x average): reduce to 0.5%
  - Low volatility session (Asian for gold): reduce to 0.5%
  - Account drawdown > 5%: reduce to 0.5%
  - Account drawdown > 8%: STOP TRADING
  - High confidence trade (>85): allow up to 1.5%

R:R RULES:
  Minimum 1.5:1 always (from settings.MIN_RR_RATIO)
  Preferred by volatility:
  - Low vol: 2:1
  - Normal: 1.5:1
  - High vol: 2.5:1
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from agent.config import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Symbol configuration for lot-size calculation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSpec:
    """Per-symbol parameters needed for position sizing."""

    contract_size: float
    tick_size: float
    tick_value: float


SYMBOL_SPECS: dict[str, SymbolSpec] = {
    "XAUUSD": SymbolSpec(contract_size=100, tick_size=0.01, tick_value=1.0),
    "XAGUSD": SymbolSpec(contract_size=5000, tick_size=0.001, tick_value=5.0),
    "BTCUSD": SymbolSpec(contract_size=1, tick_size=0.01, tick_value=0.01),
}

# Crypto symbols use a simplified lot-size formula.
_CRYPTO_SYMBOLS = {"BTCUSD"}

# Volatility-rank to preferred R:R mapping.
_PREFERRED_RR: dict[str, float] = {
    "low": 2.0,
    "normal": 1.5,
    "high": 2.5,
}

# Drawdown thresholds
_DRAWDOWN_REDUCE_PCT = 5.0
_DRAWDOWN_STOP_PCT = 8.0

# Reduced risk percentage applied under adverse conditions.
_REDUCED_RISK_PCT = 0.5

# Elevated risk percentage for high-confidence setups.
_HIGH_CONFIDENCE_RISK_PCT = 1.5
_HIGH_CONFIDENCE_THRESHOLD = 85

# Streak threshold for risk reduction.
_LOSING_STREAK_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Core position-sizing
# ---------------------------------------------------------------------------

def calculate_lot_size(
    symbol: str,
    balance: float,
    entry: float,
    sl: float,
    risk_pct: float,
) -> float:
    """Return the lot size for a trade given risk parameters.

    Parameters
    ----------
    symbol:
        Trading instrument (e.g. ``"XAUUSD"``).
    balance:
        Current account balance in USD.
    entry:
        Planned entry price.
    sl:
        Stop-loss price.
    risk_pct:
        Percentage of balance to risk (e.g. ``1.0`` for 1 %).

    Returns
    -------
    float
        Lot size rounded to 2 decimal places.  Returns ``0.0`` when the
        stop-loss distance is zero or negative balance.
    """
    sl_distance = abs(entry - sl)
    if sl_distance == 0 or balance <= 0:
        log.warning(
            "invalid_lot_size_inputs",
            symbol=symbol,
            balance=balance,
            entry=entry,
            sl=sl,
        )
        return 0.0

    risk_amount = balance * (risk_pct / 100.0)

    if symbol in _CRYPTO_SYMBOLS:
        lot_size = risk_amount / sl_distance
    else:
        spec = SYMBOL_SPECS.get(symbol)
        if spec is None:
            log.error("unknown_symbol", symbol=symbol)
            return 0.0
        pip_value = spec.contract_size * spec.tick_size
        lot_size = risk_amount / (sl_distance / spec.tick_size * pip_value)

    lot_size = round(lot_size, 2)

    log.info(
        "lot_size_calculated",
        symbol=symbol,
        balance=balance,
        risk_pct=risk_pct,
        sl_distance=sl_distance,
        lot_size=lot_size,
    )
    return lot_size


# ---------------------------------------------------------------------------
# Dynamic risk adjustment
# ---------------------------------------------------------------------------

def get_adjusted_risk_pct(
    account_state: dict,
    market_context,
    consecutive_losses: int,
    consecutive_wins: int,
    confidence: int,
) -> float:
    """Return the dynamically adjusted risk percentage for the next trade.

    Parameters
    ----------
    account_state:
        Must contain ``"balance"`` (float) and ``"peak_balance"`` (float).
    market_context:
        Object (or dict-like) with optional attributes / keys:
        - ``atr`` (float): current ATR value.
        - ``avg_atr`` (float): rolling average ATR.
        - ``session`` (str): ``"asian"`` | ``"london"`` | ``"newyork"``.
        - ``symbol`` (str): instrument name.
    consecutive_losses:
        Number of consecutive losing trades.
    consecutive_wins:
        Number of consecutive winning trades.
    confidence:
        Trade confidence score (0-100).

    Returns
    -------
    float
        Adjusted risk percentage.  Returns ``0.0`` when trading should stop.
    """
    base_risk = settings.MAX_RISK_PER_TRADE_PCT

    # --- Drawdown check (most restrictive first) ---
    balance = account_state.get("balance", 0.0)
    peak = account_state.get("peak_balance", balance)
    drawdown_pct = ((peak - balance) / peak * 100.0) if peak > 0 else 0.0

    if drawdown_pct >= _DRAWDOWN_STOP_PCT:
        log.warning(
            "trading_stopped_drawdown",
            drawdown_pct=round(drawdown_pct, 2),
        )
        return 0.0

    if drawdown_pct > _DRAWDOWN_REDUCE_PCT:
        log.info(
            "risk_reduced_drawdown",
            drawdown_pct=round(drawdown_pct, 2),
            risk_pct=_REDUCED_RISK_PCT,
        )
        return _REDUCED_RISK_PCT

    # --- Losing streak ---
    if consecutive_losses >= _LOSING_STREAK_THRESHOLD:
        log.info(
            "risk_reduced_losing_streak",
            consecutive_losses=consecutive_losses,
            risk_pct=_REDUCED_RISK_PCT,
        )
        return _REDUCED_RISK_PCT

    # --- Volatility check ---
    atr = _attr(market_context, "atr")
    avg_atr = _attr(market_context, "avg_atr")
    if atr is not None and avg_atr is not None and avg_atr > 0:
        if atr > 1.5 * avg_atr:
            log.info(
                "risk_reduced_high_volatility",
                atr=atr,
                avg_atr=avg_atr,
                risk_pct=_REDUCED_RISK_PCT,
            )
            return _REDUCED_RISK_PCT

    # --- Low-volatility session (Asian session for gold) ---
    session = _attr(market_context, "session")
    symbol = _attr(market_context, "symbol")
    if session is not None and symbol is not None:
        if str(session).lower() == "asian" and str(symbol).upper() == "XAUUSD":
            log.info(
                "risk_reduced_asian_session_gold",
                risk_pct=_REDUCED_RISK_PCT,
            )
            return _REDUCED_RISK_PCT

    # --- Winning streak: stay at base (no euphoria increase) ---
    if consecutive_wins >= 3:
        log.info(
            "risk_kept_at_base_winning_streak",
            consecutive_wins=consecutive_wins,
            risk_pct=base_risk,
        )
        return base_risk

    # --- High confidence ---
    if confidence > _HIGH_CONFIDENCE_THRESHOLD:
        log.info(
            "risk_elevated_high_confidence",
            confidence=confidence,
            risk_pct=_HIGH_CONFIDENCE_RISK_PCT,
        )
        return _HIGH_CONFIDENCE_RISK_PCT

    return base_risk


# ---------------------------------------------------------------------------
# R:R validation
# ---------------------------------------------------------------------------

def validate_rr_ratio(
    entry: float,
    sl: float,
    tp: float,
    min_rr: float | None = None,
) -> bool:
    """Return ``True`` if the reward-to-risk ratio meets the minimum.

    Parameters
    ----------
    entry:
        Entry price.
    sl:
        Stop-loss price.
    tp:
        Take-profit price.
    min_rr:
        Minimum acceptable R:R.  Defaults to ``settings.MIN_RR_RATIO``.
    """
    if min_rr is None:
        min_rr = settings.MIN_RR_RATIO

    risk = abs(entry - sl)
    reward = abs(tp - entry)

    if risk == 0:
        log.warning("rr_validation_zero_risk", entry=entry, sl=sl, tp=tp)
        return False

    rr = reward / risk
    valid = rr >= min_rr

    log.info(
        "rr_validated",
        entry=entry,
        sl=sl,
        tp=tp,
        rr=round(rr, 2),
        min_rr=min_rr,
        valid=valid,
    )
    return valid


# ---------------------------------------------------------------------------
# Position value cap
# ---------------------------------------------------------------------------

def get_max_position_value(balance: float) -> float:
    """Return the maximum notional position value (10 % of balance)."""
    return round(balance * 0.10, 2)


# ---------------------------------------------------------------------------
# Preferred R:R by volatility
# ---------------------------------------------------------------------------

def get_preferred_rr(volatility_rank: str) -> float:
    """Return the preferred R:R ratio for a given volatility rank.

    Parameters
    ----------
    volatility_rank:
        One of ``"low"``, ``"normal"``, ``"high"``.

    Returns
    -------
    float
        Preferred R:R.  Falls back to ``settings.MIN_RR_RATIO`` for unknown
        ranks.
    """
    return _PREFERRED_RR.get(volatility_rank.lower(), settings.MIN_RR_RATIO)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attr(obj, name):
    """Retrieve *name* from *obj* via attribute access or key lookup."""
    if obj is None:
        return None
    try:
        return getattr(obj, name)
    except AttributeError:
        pass
    try:
        return obj[name]
    except (KeyError, TypeError):
        return None
