#!/usr/bin/env python3
"""CLI backtest runner.

Usage:
    python scripts/run_backtest.py --symbol XAUUSD --start 2024-01-01 --end 2024-06-30 --mode rules --balance 10000
    python scripts/run_backtest.py --csv data/XAUUSD_M1.csv --mode rules
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path so `agent` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import structlog

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest on historical candle data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_backtest.py --symbol XAUUSD --start 2024-01-01 --end 2024-06-30\n"
            "  python scripts/run_backtest.py --csv data/XAUUSD_M1.csv --mode rules\n"
        ),
    )

    # Data source (mutually exclusive: MT5 date range OR CSV file)
    source = parser.add_argument_group("data source")
    source.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    source.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (for MT5 source)")
    source.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (for MT5 source)")
    source.add_argument("--csv", type=str, default=None, help="Path to CSV file with M1 candles")
    source.add_argument("--csv-3m", type=str, default=None, help="Path to CSV file with M3 candles (optional)")

    # Engine parameters
    engine = parser.add_argument_group("engine parameters")
    engine.add_argument("--mode", type=str, default="rules", choices=["rules", "ai"],
                        help="Signal generation mode (default: rules)")
    engine.add_argument("--balance", type=float, default=10_000.0, help="Initial balance (default: 10000)")
    engine.add_argument("--risk", type=float, default=1.0, help="Risk per trade %% (default: 1.0)")
    engine.add_argument("--slippage", type=float, default=1.0, help="Max slippage in pips (default: 1.0)")
    engine.add_argument("--spread", type=float, default=1.0, help="Spread cost in pips (default: 1.0)")
    engine.add_argument("--rr", type=float, default=2.0, help="Preferred reward:risk ratio (default: 2.0)")

    # Output
    output = parser.add_argument_group("output")
    output.add_argument("--report", type=str, default="backtest_report.html",
                        help="Output HTML report path (default: backtest_report.html)")
    output.add_argument("--no-report", action="store_true", help="Skip HTML report generation")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from agent.backtest.data_loader import load_from_csv, load_from_mt5
    from agent.backtest.engine import BacktestEngine
    from agent.backtest.report import save_report

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    df_1m = None
    df_3m = None

    if args.csv:
        logger.info("loading_csv", path=args.csv)
        df_1m = load_from_csv(args.csv)
        if args.csv_3m:
            df_3m = load_from_csv(args.csv_3m)
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        logger.info("loading_mt5", symbol=args.symbol, start=str(start), end=str(end))
        df_1m = load_from_mt5(args.symbol, "M1", start, end)
        df_3m = load_from_mt5(args.symbol, "M3", start, end)
    else:
        print("ERROR: Provide either --csv or both --start and --end for MT5 data.", file=sys.stderr)
        sys.exit(1)

    if df_1m is None or df_1m.empty:
        print("ERROR: No candle data loaded. Check your data source.", file=sys.stderr)
        sys.exit(1)

    logger.info("data_loaded", rows_1m=len(df_1m), rows_3m=len(df_3m) if df_3m is not None else 0)

    # -------------------------------------------------------------------
    # Run backtest
    # -------------------------------------------------------------------
    engine = BacktestEngine(
        symbol=args.symbol,
        initial_balance=args.balance,
        risk_pct=args.risk,
        slippage_pips=args.slippage,
        spread_cost_pips=args.spread,
        preferred_rr=args.rr,
    )

    result = engine.run(df_1m, df_3m, mode=args.mode)

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    m = result.metrics
    net = result.final_balance - result.initial_balance
    pct = (net / result.initial_balance * 100) if result.initial_balance else 0

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  --  {result.symbol}  ({result.mode.upper()} mode)")
    print("=" * 60)
    print(f"  Period:          {str(result.start_date)[:10]} to {str(result.end_date)[:10]}")
    print(f"  Initial Balance: {result.initial_balance:>12,.2f}")
    print(f"  Final Balance:   {result.final_balance:>12,.2f}")
    print(f"  Net P&L:         {net:>+12,.2f}  ({pct:+.1f}%)")
    print("-" * 60)
    print(f"  Total Trades:    {m.get('total_trades', 0):>6}")
    print(f"  Win Rate:        {m.get('win_rate', 0):>6.1f}%")
    print(f"  Profit Factor:   {m.get('profit_factor', 0):>6.2f}")
    print(f"  Max Drawdown:    {m.get('max_drawdown_pct', 0):>6.2f}%")
    print(f"  Sharpe Ratio:    {m.get('sharpe_ratio', 0):>6.2f}")
    print(f"  Avg R:R:         {m.get('avg_rr', 0):>6.2f}")
    print(f"  Max Win:         {m.get('max_win', 0):>+10.2f}")
    print(f"  Max Loss:        {m.get('max_loss', 0):>+10.2f}")
    print(f"  Avg Duration:    {m.get('avg_trade_duration_minutes', 0):>6.0f} min")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Generate report
    # -------------------------------------------------------------------
    if not args.no_report:
        report_path = save_report(result, filepath=args.report)
        print(f"\n  HTML report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
