#!/usr/bin/env python3
"""Download historical data from MT5 and save to CSV.

Usage:
    python scripts/seed_historical.py --symbol XAUUSD --timeframe M1 --days 90
    python scripts/seed_historical.py --symbol BTCUSD --timeframe M3 --days 30 --output data/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the project root is on sys.path so `agent` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical candle data from MT5 and save to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/seed_historical.py --symbol XAUUSD --timeframe M1 --days 90\n"
            "  python scripts/seed_historical.py --symbol BTCUSD --timeframe M1 --days 30 --output data/\n"
            "  python scripts/seed_historical.py --symbol XAUUSD --timeframe M1 "
            "--start 2024-01-01 --end 2024-06-30\n"
        ),
    )

    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", type=str, default="M1",
                        choices=["M1", "M3", "M5", "M15", "H1"],
                        help="Candle timeframe (default: M1)")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of history to download (default: 90)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory for CSV files (default: data/)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import structlog
    logger = structlog.get_logger()

    from agent.backtest.data_loader import load_from_mt5

    # -------------------------------------------------------------------
    # Determine date range
    # -------------------------------------------------------------------
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=args.days)

    logger.info(
        "seed_historical_start",
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )

    # -------------------------------------------------------------------
    # Download data
    # -------------------------------------------------------------------
    df = load_from_mt5(args.symbol, args.timeframe, start, end)

    if df.empty:
        print(
            f"ERROR: No data returned from MT5 for {args.symbol} {args.timeframe} "
            f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}).",
            file=sys.stderr,
        )
        print(
            "Make sure MT5 is running and the symbol is available in Market Watch.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -------------------------------------------------------------------
    # Save to CSV
    # -------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{args.symbol}_{args.timeframe}_"
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)

    print(f"\nDownloaded {len(df):,} candles for {args.symbol} {args.timeframe}")
    print(f"  Period:  {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"  First:   {df.iloc[0]['time']}")
    print(f"  Last:    {df.iloc[-1]['time']}")
    print(f"  Saved:   {filepath.resolve()}\n")

    logger.info(
        "seed_historical_complete",
        rows=len(df),
        filepath=str(filepath.resolve()),
    )


if __name__ == "__main__":
    main()
