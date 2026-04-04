#!/usr/bin/env python3
"""v4 historical data acquisition: download M1 from MT5, resample higher TFs,
run quality checks, compute HistoricalContext, and save as Parquet.

Usage:
    python scripts/seed_historical.py --symbol XAUUSD --days 1825
    python scripts/seed_historical.py --symbol BTCUSD --days 365 --output-dir data/historical
    python scripts/seed_historical.py --symbol XAUUSD --csv data/XAUUSD_M1.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

# Ensure the project root is on sys.path so `agent` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = structlog.get_logger()

# Maximum chunk size for MT5 requests (90 days of M1 data)
CHUNK_DAYS = 90


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical M1 data, resample higher TFs, and compute HistoricalContext.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/seed_historical.py --symbol XAUUSD --days 1825\n"
            "  python scripts/seed_historical.py --symbol BTCUSD --days 365 --output-dir data/historical\n"
            "  python scripts/seed_historical.py --symbol XAUUSD --csv data/XAUUSD_M1.csv\n"
        ),
    )
    parser.add_argument("--symbol", type=str, default="XAUUSD",
                        help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--days", type=int, default=1825,
                        help="Days of history to fetch (default: 1825 = ~5 years)")
    parser.add_argument("--output-dir", type=str, default="data/historical",
                        help="Output directory for Parquet files (default: data/historical)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with M1 data (skip MT5 download)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MT5 download in chunks
# ---------------------------------------------------------------------------

def download_m1_chunked(symbol: str, days: int) -> pd.DataFrame:
    """Download max available M1 data from MT5 in 90-day chunks.

    Returns a single deduplicated DataFrame sorted by time.
    """
    from agent.backtest.data_loader import load_from_mt5

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    chunks: list[pd.DataFrame] = []
    chunk_start = start

    logger.info("mt5_download_start", symbol=symbol,
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                total_days=days, estimated_chunks=(days + CHUNK_DAYS - 1) // CHUNK_DAYS)

    chunk_num = 0
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        chunk_num += 1

        logger.info("downloading_chunk", chunk=chunk_num,
                     start=chunk_start.strftime("%Y-%m-%d"),
                     end=chunk_end.strftime("%Y-%m-%d"))

        try:
            df_chunk = load_from_mt5(symbol, "M1", chunk_start, chunk_end)
            if not df_chunk.empty:
                chunks.append(df_chunk)
                logger.info("chunk_complete", chunk=chunk_num, rows=len(df_chunk))
            else:
                logger.warning("chunk_empty", chunk=chunk_num,
                               start=chunk_start.strftime("%Y-%m-%d"),
                               end=chunk_end.strftime("%Y-%m-%d"))
        except Exception as e:
            logger.error("chunk_failed", chunk=chunk_num, error=str(e))

        chunk_start = chunk_end
        # Brief pause between chunks to avoid overloading MT5
        time.sleep(0.5)

    if not chunks:
        logger.error("no_data_downloaded", symbol=symbol)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    logger.info("mt5_download_complete", symbol=symbol, total_rows=len(df),
                first=str(df["time"].iloc[0]), last=str(df["time"].iloc[-1]))
    return df


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_m1_from_csv(csv_path: str) -> pd.DataFrame:
    """Load M1 data from a CSV file for brokers with shallow MT5 M1 history.

    Accepts common column naming conventions.
    """
    from agent.backtest.data_loader import load_from_csv

    logger.info("loading_csv", path=csv_path)
    df = load_from_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    logger.info("csv_loaded", rows=len(df),
                first=str(df["time"].iloc[0]), last=str(df["time"].iloc[-1]))
    return df


# ---------------------------------------------------------------------------
# Resample M1 -> higher timeframes
# ---------------------------------------------------------------------------

def resample_m1(df_m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 DataFrame to a higher timeframe using OHLCV aggregation.

    Args:
        df_m1: M1 candles with columns: time, open, high, low, close, volume, spread
        rule: Pandas resample rule, e.g. '3min', '15min', '1h'

    Returns:
        Resampled DataFrame with same column structure.
    """
    df = df_m1.copy()
    df = df.set_index("time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "spread" in df.columns:
        agg["spread"] = "mean"

    resampled = df.resample(rule).agg(agg).dropna(subset=["open"])
    resampled = resampled.reset_index()
    return resampled


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame, label: str) -> dict:
    """Run data quality checks and log results.

    Checks:
        - No duplicate timestamps
        - OHLC integrity (high >= open/close, low <= open/close)
        - Gap detection (missing minutes beyond expected)

    Returns:
        dict with quality metrics.
    """
    issues: dict = {"label": label, "rows": len(df)}

    if df.empty:
        logger.warning("quality_check_empty", label=label)
        issues["empty"] = True
        return issues

    # 1. Duplicate check
    dupes = df["time"].duplicated().sum()
    issues["duplicates"] = int(dupes)
    if dupes > 0:
        logger.warning("quality_duplicates_found", label=label, count=dupes)

    # 2. OHLC integrity
    bad_high = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
    bad_low = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
    issues["ohlc_bad_high"] = int(bad_high)
    issues["ohlc_bad_low"] = int(bad_low)
    if bad_high > 0 or bad_low > 0:
        logger.warning("quality_ohlc_integrity", label=label,
                       bad_high=int(bad_high), bad_low=int(bad_low))

    # 3. Gap detection (only for M1)
    if "M1" in label or "m1" in label:
        time_diffs = df["time"].diff().dt.total_seconds().dropna()
        # Gaps > 5 minutes (300s) during weekdays
        large_gaps = time_diffs[time_diffs > 300]
        # Filter out weekend gaps (> 2 days = 172800s)
        weekday_gaps = large_gaps[large_gaps < 172800]
        issues["gaps_gt_5min"] = int(len(weekday_gaps))
        if len(weekday_gaps) > 0:
            gap_hours = weekday_gaps.sum() / 3600.0
            issues["total_gap_hours"] = round(gap_hours, 1)
            logger.info("quality_gaps", label=label,
                        gaps_gt_5min=int(len(weekday_gaps)),
                        total_gap_hours=round(gap_hours, 1))

    # 4. Date range
    issues["first"] = str(df["time"].iloc[0])
    issues["last"] = str(df["time"].iloc[-1])
    span_days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
    issues["span_days"] = span_days

    logger.info("quality_check_passed", **issues)
    return issues


# ---------------------------------------------------------------------------
# Save as Parquet
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save DataFrame to Parquet format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("parquet_saved", label=label, path=str(path),
                rows=len(df), size_mb=round(size_mb, 2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    symbol = args.symbol
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print(f"  TradeNoJutsu v4 — Historical Data Seeder")
    print(f"  Symbol: {symbol}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # Step 1: Acquire M1 data
    # -----------------------------------------------------------------
    if args.csv:
        df_m1 = load_m1_from_csv(args.csv)
    else:
        df_m1 = download_m1_chunked(symbol, args.days)

    if df_m1.empty:
        print("ERROR: No M1 data available. Exiting.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------
    # Step 2: Quality checks on M1
    # -----------------------------------------------------------------
    m1_quality = run_quality_checks(df_m1, f"{symbol}_M1")

    # -----------------------------------------------------------------
    # Step 3: Resample M1 -> 3M, 15M, 1H
    # -----------------------------------------------------------------
    print("Resampling M1 to higher timeframes...")
    df_3m = resample_m1(df_m1, "3min")
    df_15m = resample_m1(df_m1, "15min")
    df_1h = resample_m1(df_m1, "1h")

    run_quality_checks(df_3m, f"{symbol}_M3")
    run_quality_checks(df_15m, f"{symbol}_M15")
    run_quality_checks(df_1h, f"{symbol}_H1")

    # -----------------------------------------------------------------
    # Step 4: Save as Parquet
    # -----------------------------------------------------------------
    print("Saving Parquet files...")
    sym_dir = output_dir / symbol
    save_parquet(df_m1, sym_dir / f"{symbol}_M1.parquet", f"{symbol}_M1")
    save_parquet(df_3m, sym_dir / f"{symbol}_M3.parquet", f"{symbol}_M3")
    save_parquet(df_15m, sym_dir / f"{symbol}_M15.parquet", f"{symbol}_M15")
    save_parquet(df_1h, sym_dir / f"{symbol}_H1.parquet", f"{symbol}_H1")

    # -----------------------------------------------------------------
    # Step 5: Compute and save HistoricalContext
    # -----------------------------------------------------------------
    print("Computing HistoricalContext...")
    from agent.ml.data_prep import compute_historical_context

    ctx = compute_historical_context(symbol, df_m1)
    ctx_path = sym_dir / f"{symbol}_context.json"
    ctx.save(str(ctx_path))

    # -----------------------------------------------------------------
    # Step 6: Summary
    # -----------------------------------------------------------------
    span_days = m1_quality.get("span_days", 0)
    print(f"\n{'='*60}")
    print(f"  Seed Complete: {symbol}")
    print(f"{'='*60}")
    print(f"  M1 candles:  {len(df_m1):>10,}")
    print(f"  M3 candles:  {len(df_3m):>10,}")
    print(f"  M15 candles: {len(df_15m):>10,}")
    print(f"  H1 candles:  {len(df_1h):>10,}")
    print(f"  Date range:  {df_m1['time'].iloc[0]} — {df_m1['time'].iloc[-1]}")
    print(f"  Span:        {span_days} days")
    print(f"  Output dir:  {sym_dir.resolve()}")
    print(f"  Context:     {ctx_path.resolve()}")

    # HistoricalContext summary
    print(f"\n  ATR p50 (15m):        {ctx.atr_percentiles[2]:.4f}")
    print(f"  Price range:          {ctx.price_low:.2f} — {ctx.price_high:.2f}")
    print(f"  Avg spread:           {ctx.spread_mean:.2f}")
    print(f"  Sessions ATR (mean):  "
          f"asian={ctx.atr_mean_by_session.get('asian', 0):.4f}  "
          f"london={ctx.atr_mean_by_session.get('london', 0):.4f}  "
          f"overlap={ctx.atr_mean_by_session.get('overlap', 0):.4f}  "
          f"ny={ctx.atr_mean_by_session.get('ny', 0):.4f}")
    print(f"  DOW avg range:        {ctx.dow_avg_range}")
    print()


if __name__ == "__main__":
    main()
