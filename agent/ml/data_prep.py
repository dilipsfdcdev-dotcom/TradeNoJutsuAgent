"""Prepare training datasets from trade journal + historical candles."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import structlog
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from torch.utils.data import Dataset
from datetime import datetime, timedelta

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# HistoricalContext — pre-computed statistics from full historical data
# ---------------------------------------------------------------------------

@dataclass
class HistoricalContext:
    """Pre-computed statistics from full historical data per symbol."""

    symbol: str
    atr_mean_by_session: dict[str, float] = field(default_factory=dict)
    atr_std_by_session: dict[str, float] = field(default_factory=dict)
    atr_percentiles: list[float] = field(default_factory=lambda: [0.0] * 5)  # [p10, p25, p50, p75, p90]
    vol_mean_by_session: dict[str, float] = field(default_factory=dict)
    vol_percentiles: list[float] = field(default_factory=lambda: [0.0] * 5)
    spread_mean: float = 0.0
    spread_percentiles: list[float] = field(default_factory=lambda: [0.0] * 5)
    price_high: float = 0.0
    price_low: float = 0.0
    session_long_wr: dict[str, float] = field(default_factory=dict)
    session_short_wr: dict[str, float] = field(default_factory=dict)
    dow_avg_range: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON requires string keys; dow_avg_range has int keys
        d["dow_avg_range"] = {str(k): v for k, v in d["dow_avg_range"].items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalContext":
        data = dict(data)  # shallow copy
        # Restore int keys for dow_avg_range
        if "dow_avg_range" in data:
            data["dow_avg_range"] = {int(k): v for k, v in data["dow_avg_range"].items()}
        return cls(**data)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("historical_context_saved", path=str(p), symbol=self.symbol)

    @classmethod
    def load(cls, path: str) -> "HistoricalContext":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _hour_to_session(hour: int) -> str:
    """Map UTC hour to session name."""
    if 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 12:
        return "london"
    elif 12 <= hour < 17:
        return "overlap"
    else:
        return "ny"


def compute_historical_context(symbol: str, df_m1: pd.DataFrame) -> HistoricalContext:
    """Compute context stats from full historical M1 data.

    Expects df_m1 with columns: time, open, high, low, close, volume, spread
    """
    df = df_m1.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    # Compute per-bar range as proxy ATR (true ATR needs prior close, approximate)
    df["range"] = df["high"] - df["low"]
    # Rolling 14-bar ATR proxy on 15-min resampled data for meaningful ATR
    df_15 = df[["open", "high", "low", "close", "volume"]].resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    df_15["atr"] = (df_15["high"] - df_15["low"]).rolling(14).mean()
    df_15["session"] = df_15.index.hour.map(_hour_to_session)
    df_15 = df_15.dropna(subset=["atr"])

    # ATR by session
    atr_mean_by_session = {}
    atr_std_by_session = {}
    for sess in ["asian", "london", "overlap", "ny"]:
        subset = df_15.loc[df_15["session"] == sess, "atr"]
        atr_mean_by_session[sess] = float(subset.mean()) if len(subset) > 0 else 0.0
        atr_std_by_session[sess] = float(subset.std()) if len(subset) > 0 else 0.0

    # ATR percentiles (global)
    atr_vals = df_15["atr"].dropna()
    atr_percentiles = [float(np.percentile(atr_vals, p)) for p in [10, 25, 50, 75, 90]] if len(atr_vals) > 0 else [0.0] * 5

    # Volume by session
    df["session"] = df.index.hour.map(_hour_to_session)
    vol_mean_by_session = {}
    for sess in ["asian", "london", "overlap", "ny"]:
        subset = df.loc[df["session"] == sess, "volume"]
        vol_mean_by_session[sess] = float(subset.mean()) if len(subset) > 0 else 0.0

    # Volume percentiles
    vol_vals = df["volume"].dropna()
    vol_percentiles = [float(np.percentile(vol_vals, p)) for p in [10, 25, 50, 75, 90]] if len(vol_vals) > 0 else [0.0] * 5

    # Spread stats
    spread_vals = df["spread"].dropna() if "spread" in df.columns else pd.Series(dtype=float)
    spread_mean = float(spread_vals.mean()) if len(spread_vals) > 0 else 0.0
    spread_percentiles = [float(np.percentile(spread_vals, p)) for p in [10, 25, 50, 75, 90]] if len(spread_vals) > 0 else [0.0] * 5

    # Price range
    price_high = float(df["high"].max()) if len(df) > 0 else 0.0
    price_low = float(df["low"].min()) if len(df) > 0 else 0.0

    # Session win rates — placeholder (needs trade data, default 0.5)
    sessions = ["asian", "london", "overlap", "ny"]
    session_long_wr = {s: 0.5 for s in sessions}
    session_short_wr = {s: 0.5 for s in sessions}

    # Day-of-week average daily range
    df["dow"] = df.index.dayofweek
    daily_ranges = df.groupby(df.index.date).agg({"high": "max", "low": "min"})
    daily_ranges["range"] = daily_ranges["high"] - daily_ranges["low"]
    daily_ranges.index = pd.to_datetime(daily_ranges.index)
    daily_ranges["dow"] = daily_ranges.index.dayofweek
    dow_avg_range = {}
    for d in range(5):
        subset = daily_ranges.loc[daily_ranges["dow"] == d, "range"]
        dow_avg_range[d] = float(subset.mean()) if len(subset) > 0 else 0.0

    return HistoricalContext(
        symbol=symbol,
        atr_mean_by_session=atr_mean_by_session,
        atr_std_by_session=atr_std_by_session,
        atr_percentiles=atr_percentiles,
        vol_mean_by_session=vol_mean_by_session,
        vol_percentiles=vol_percentiles,
        spread_mean=spread_mean,
        spread_percentiles=spread_percentiles,
        price_high=price_high,
        price_low=price_low,
        session_long_wr=session_long_wr,
        session_short_wr=session_short_wr,
        dow_avg_range=dow_avg_range,
    )


# ---------------------------------------------------------------------------
# Outcome-based labeling
# ---------------------------------------------------------------------------

LABEL_PARAMS = {
    "XAUUSD": {"forward_window": 15, "tp_atr_mult": 1.5, "sl_atr_mult": 1.0, "min_move_pips": 30},
    "BTCUSD": {"forward_window": 15, "tp_atr_mult": 1.5, "sl_atr_mult": 1.0, "min_move_pips": 30},
    "XAGUSD": {"forward_window": 15, "tp_atr_mult": 1.5, "sl_atr_mult": 1.0, "min_move_pips": 5},
}


def compute_outcome_labels(df: pd.DataFrame, candle_idx: int, symbol: str, atr: float) -> dict | None:
    """Look forward from candle to determine what happened.

    Returns dict with keys: direction_label, quality_score, optimal_entry, regime
    or None if not enough forward data.
    """
    params = LABEL_PARAMS.get(symbol, LABEL_PARAMS["XAUUSD"])
    fw = params["forward_window"]

    if candle_idx + fw >= len(df):
        return None

    entry = float(df.iloc[candle_idx]["close"])
    future = df.iloc[candle_idx + 1 : candle_idx + 1 + fw]
    if len(future) < fw // 2:
        return None

    tp_dist = atr * params["tp_atr_mult"]
    sl_dist = atr * params["sl_atr_mult"]

    # Simulate both directions
    long_result = _simulate_trade(future, entry, "long", tp_dist, sl_dist)
    short_result = _simulate_trade(future, entry, "short", tp_dist, sl_dist)

    # Decide direction label: 0=buy, 1=sell, 2=no-trade
    if long_result["hit_tp"] and not short_result["hit_tp"]:
        direction_label = 0  # buy
    elif short_result["hit_tp"] and not long_result["hit_tp"]:
        direction_label = 1  # sell
    elif long_result["hit_tp"] and short_result["hit_tp"]:
        # Both hit TP — pick the faster one
        direction_label = 0 if long_result["bars_to_tp"] <= short_result["bars_to_tp"] else 1
    else:
        direction_label = 2  # no-trade

    # Quality score: 0-1 based on how clean the move was
    if direction_label == 2:
        quality_score = 0.0
    else:
        chosen = long_result if direction_label == 0 else short_result
        # Quality: fast TP hit + low adverse excursion
        speed_score = max(0, 1.0 - chosen["bars_to_tp"] / fw)
        adverse_score = max(0, 1.0 - chosen["max_adverse"] / sl_dist) if sl_dist > 0 else 0.5
        quality_score = 0.6 * speed_score + 0.4 * adverse_score

    # Optimal entry: high-quality directional trades
    optimal_entry = 1 if quality_score >= 0.5 and direction_label != 2 else 0

    regime = _classify_regime(df, candle_idx)

    return {
        "direction_label": direction_label,
        "quality_score": round(quality_score, 4),
        "optimal_entry": optimal_entry,
        "regime": regime,
    }


def _simulate_trade(future_candles: pd.DataFrame, entry: float, direction: str,
                     tp_dist: float, sl_dist: float) -> dict:
    """Walk forward bar-by-bar simulating a trade.

    Returns dict with: hit_tp, hit_sl, bars_to_tp, max_adverse
    """
    hit_tp = False
    hit_sl = False
    bars_to_tp = len(future_candles)
    max_adverse = 0.0

    for i, (_, candle) in enumerate(future_candles.iterrows()):
        high = float(candle["high"])
        low = float(candle["low"])

        if direction == "long":
            adverse = entry - low
            favorable = high - entry
        else:
            adverse = high - entry
            favorable = entry - low

        max_adverse = max(max_adverse, adverse)

        if favorable >= tp_dist and not hit_tp:
            hit_tp = True
            bars_to_tp = i + 1
            break
        if adverse >= sl_dist:
            hit_sl = True
            break

    return {
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
        "bars_to_tp": bars_to_tp,
        "max_adverse": max_adverse,
    }


def _classify_regime(df: pd.DataFrame, idx: int) -> str:
    """Classify regime at index: trending / ranging / volatile based on ATR percentile."""
    lookback = min(100, idx)
    if lookback < 20:
        return "ranging"

    window = df.iloc[max(0, idx - lookback) : idx + 1]
    ranges = window["high"] - window["low"]
    current_range = float(ranges.iloc[-1])
    pctile = float((ranges < current_range).sum()) / len(ranges)

    # Also check directional consistency for trending
    closes = window["close"].values
    if len(closes) > 20:
        returns = np.diff(closes[-20:])
        pos_pct = (returns > 0).sum() / len(returns)
        directional = pos_pct > 0.65 or pos_pct < 0.35
    else:
        directional = False

    if pctile >= 0.75:
        return "volatile"
    elif directional:
        return "trending"
    else:
        return "ranging"


def generate_labeled_dataset(symbol: str, df_m1: pd.DataFrame, every_n: int = 15) -> pd.DataFrame:
    """Walk through historical candles, compute features + outcome labels for every Nth candle.

    Args:
        symbol: Trading symbol
        df_m1: M1 candle DataFrame with columns: time, open, high, low, close, volume, spread
        every_n: Evaluate every Nth candle (default 15 = every 15 minutes)

    Returns:
        DataFrame with feature columns + label columns
    """
    df = df_m1.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Compute rolling ATR (14-bar)
    df["_atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["_atr"] = df["_atr"].fillna(method="bfill")

    records = []
    total = len(df)
    params = LABEL_PARAMS.get(symbol, LABEL_PARAMS["XAUUSD"])
    fw = params["forward_window"]

    for idx in range(30, total - fw - 1, every_n):
        atr = float(df.iloc[idx]["_atr"])
        if atr <= 0:
            continue

        labels = compute_outcome_labels(df, idx, symbol, atr)
        if labels is None:
            continue

        row = df.iloc[idx]
        record = {
            "time": row.get("time", None) if "time" in df.columns else df.index[idx],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "atr": atr,
            "session": _hour_to_session(
                row["time"].hour if "time" in df.columns and hasattr(row["time"], "hour")
                else df.index[idx].hour if hasattr(df.index[idx], "hour") else 0
            ),
            "day_of_week": (
                row["time"].weekday() if "time" in df.columns and hasattr(row["time"], "weekday")
                else df.index[idx].weekday() if hasattr(df.index[idx], "weekday") else 0
            ),
        }
        record.update(labels)
        records.append(record)

    result = pd.DataFrame(records)
    logger.info("labeled_dataset_generated", symbol=symbol, samples=len(result),
                buy_pct=round((result["direction_label"] == 0).mean(), 3) if len(result) > 0 else 0,
                sell_pct=round((result["direction_label"] == 1).mean(), 3) if len(result) > 0 else 0,
                no_trade_pct=round((result["direction_label"] == 2).mean(), 3) if len(result) > 0 else 0)
    return result


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def compute_recency_weights(dates: pd.Series, decay_rate: float = 0.003) -> np.ndarray:
    """Exponential decay weights. Recent = high weight, old = low weight.

    weight = max(exp(-decay * age_days), 0.01)

    Args:
        dates: Series of datetime-like values
        decay_rate: Exponential decay rate per day (default 0.003 ~ half-life ~231 days)

    Returns:
        Array of weights in [0.01, 1.0]
    """
    dates = pd.to_datetime(dates)
    most_recent = dates.max()
    age_days = (most_recent - dates).dt.total_seconds() / 86400.0
    weights = np.exp(-decay_rate * age_days.values)
    weights = np.clip(weights, 0.01, 1.0)
    return weights


def compute_regime_boost(current_regime: str, sample_regimes: pd.Series) -> np.ndarray:
    """Boost samples from matching regimes. Volatile regime gets 2x boost.

    Args:
        current_regime: Current market regime (trending/ranging/volatile)
        sample_regimes: Series of regime labels for training samples

    Returns:
        Array of boost multipliers (1.0 = no boost, 2.0 = max boost)
    """
    boosts = np.ones(len(sample_regimes))

    # Same-regime boost
    matching = sample_regimes == current_regime
    boosts[matching] = 1.5

    # Extra boost for volatile regime samples when current regime is volatile
    if current_regime == "volatile":
        volatile_mask = sample_regimes == "volatile"
        boosts[volatile_mask] = 2.0

    return boosts


class TradingSequenceDataset(Dataset):
    """PyTorch Dataset for LSTM training."""
    def __init__(self, sequences_1m, sequences_3m, sequences_15m, dir_labels, conf_labels, regime_labels):
        self.seq_1m = torch.FloatTensor(np.array(sequences_1m))
        self.seq_3m = torch.FloatTensor(np.array(sequences_3m))
        self.seq_15m = torch.FloatTensor(np.array(sequences_15m))
        self.dir_labels = torch.LongTensor(np.array(dir_labels))
        self.conf_labels = torch.FloatTensor(np.array(conf_labels))
        self.regime_labels = torch.LongTensor(np.array(regime_labels))

    def __len__(self):
        return len(self.dir_labels)

    def __getitem__(self, idx):
        return (self.seq_1m[idx], self.seq_3m[idx], self.seq_15m[idx],
                self.dir_labels[idx], self.conf_labels[idx], self.regime_labels[idx])


def prepare_training_data(symbol: str, csv_path: str = None) -> dict | None:
    """
    Prepare XGBoost training data. If no CSV, generates synthetic data for bootstrap.
    Returns: {"features": pd.DataFrame, "labels": pd.Series}
    """
    # Generate synthetic training data for bootstrap
    np.random.seed(hash(symbol) % 2**31)
    n = 1000
    features = pd.DataFrame({
        "rsi_1m": np.random.uniform(20, 80, n), "rsi_3m": np.random.uniform(25, 75, n),
        "atr_pctile_1m": np.random.uniform(0, 1, n), "vol_ratio_1m": np.random.lognormal(0, 0.5, n),
        "ema_stack_1m": np.random.choice([-3, -2, -1, 0, 1, 2, 3], n),
        "bb_pos_1m": np.random.uniform(0, 1, n), "body_pct_1m": np.random.uniform(0, 1, n),
        "direction_1m": np.random.choice([-1, 1], n), "close_pos_1m": np.random.uniform(0, 1, n),
        "h1_bias": np.random.choice([-1, 0, 1], n), "confluence": np.random.uniform(0, 1, n),
        "m3_confirmed": np.random.choice([0, 1], n), "m3_momentum": np.random.choice([-1, 0, 1], n),
        "session": np.random.choice([0, 1, 2, 3], n), "sentiment": np.random.uniform(-1, 1, n),
        "hour_sin": np.sin(np.random.uniform(0, 2*np.pi, n)),
        "hour_cos": np.cos(np.random.uniform(0, 2*np.pi, n)),
        "has_poi": np.random.choice([0, 1], n), "dist_to_poi": np.random.exponential(2, n),
        "gates_passed": np.random.choice([0, 1], n),
    })

    # Generate labels based on feature correlations
    signal_strength = (features["confluence"] * 0.3 + features["m3_confirmed"] * 0.2 +
                       (features["ema_stack_1m"] > 0).astype(float) * 0.2 +
                       features["has_poi"] * 0.15 + (features["vol_ratio_1m"] > 1).astype(float) * 0.15)
    noise = np.random.randn(n) * 0.3
    labels = pd.Series((signal_strength + noise > 0.5).astype(int))

    return {"features": features, "labels": labels}


def prepare_lstm_dataset(symbol: str, csv_path: str = None) -> TradingSequenceDataset | None:
    """Prepare LSTM training dataset with synthetic data for bootstrap."""
    np.random.seed(hash(symbol) % 2**31 + 1)
    n = 500
    seqs_1m, seqs_3m, seqs_15m = [], [], []
    dir_labels, conf_labels, regime_labels = [], [], []

    for _ in range(n):
        seqs_1m.append(np.random.randn(30, 5).astype(np.float32))
        seqs_3m.append(np.random.randn(20, 5).astype(np.float32))
        seqs_15m.append(np.random.randn(10, 5).astype(np.float32))
        dir_labels.append(np.random.choice([0, 1, 2]))  # buy/sell/wait
        conf_labels.append(np.random.uniform(0, 1))
        regime_labels.append(np.random.choice([0, 1, 2]))  # trending/ranging/volatile

    return TradingSequenceDataset(seqs_1m, seqs_3m, seqs_15m, dir_labels, conf_labels, regime_labels)


async def prepare_training_data_from_db(session, symbol: str, weeks: int = 4) -> dict | None:
    """Pull features + labels from DB based on actual trades."""
    if session is None:
        return None
    from sqlalchemy import select, and_
    from agent.db.models import Trade
    from datetime import timezone

    cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    stmt = select(Trade).where(
        and_(Trade.symbol == symbol, Trade.status == "closed", Trade.entry_time >= cutoff)
    ).order_by(Trade.entry_time)

    result = await session.execute(stmt)
    trades = result.scalars().all()

    if len(trades) < 50:
        return None

    features_list, labels_list = [], []
    for trade in trades:
        snapshot = trade.indicators_snapshot or {}
        if not snapshot:
            continue
        features_list.append(snapshot)
        labels_list.append(1 if (trade.pnl or 0) > 0 else 0)

    if len(features_list) < 50:
        return None

    return {"features": pd.DataFrame(features_list).fillna(0), "labels": pd.Series(labels_list)}


async def prepare_lstm_dataset_from_db(session, symbol: str, months: int = 3) -> TradingSequenceDataset | None:
    """Prepare LSTM dataset from stored trade data."""
    # For now, return None — requires stored sequence data in DB
    return None
