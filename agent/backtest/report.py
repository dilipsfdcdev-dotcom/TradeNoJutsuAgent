"""Generate HTML backtest reports from BacktestResult."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import structlog

from agent.backtest.engine import BacktestResult, BacktestTrade

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Monthly breakdown helper
# ---------------------------------------------------------------------------

def _monthly_breakdown(trades: list[BacktestTrade]) -> list[dict]:
    """Group trades by month and compute per-month stats."""
    months: dict[str, list[float]] = {}
    for t in trades:
        if t.exit_time is None or t.pnl is None:
            continue
        key = str(t.exit_time)[:7]  # "YYYY-MM"
        months.setdefault(key, []).append(t.pnl)

    breakdown = []
    for month, pnls in sorted(months.items()):
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakdown.append({
            "month": month,
            "trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        })
    return breakdown


# ---------------------------------------------------------------------------
# Equity curve SVG
# ---------------------------------------------------------------------------

def _equity_curve_svg(equity_curve: list[dict], width: int = 900, height: int = 300) -> str:
    """Render an inline SVG chart of the equity curve."""
    if len(equity_curve) < 2:
        return '<p style="color:#888;">Not enough data for equity curve.</p>'

    equities = [p["equity"] for p in equity_curve]
    balances = [p["balance"] for p in equity_curve]
    n = len(equities)

    y_min = min(min(equities), min(balances)) * 0.998
    y_max = max(max(equities), max(balances)) * 1.002
    if y_max == y_min:
        y_max = y_min + 1

    margin_left = 70
    margin_right = 20
    margin_top = 20
    margin_bottom = 30
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_pos(i: int) -> float:
        return margin_left + (i / max(n - 1, 1)) * plot_w

    def y_pos(val: float) -> float:
        return margin_top + plot_h - ((val - y_min) / (y_max - y_min)) * plot_h

    # Downsample if too many points (keep every Nth)
    step = max(1, n // 1500)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    # Build polyline points
    equity_points = " ".join(f"{x_pos(i):.1f},{y_pos(equities[i]):.1f}" for i in indices)
    balance_points = " ".join(f"{x_pos(i):.1f},{y_pos(balances[i]):.1f}" for i in indices)

    # Y-axis grid lines
    num_grid = 5
    grid_lines = ""
    for g in range(num_grid + 1):
        val = y_min + (y_max - y_min) * g / num_grid
        y = y_pos(val)
        grid_lines += (
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" '
            f'y2="{y:.1f}" stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>\n'
            f'<text x="{margin_left - 5}" y="{y:.1f}" text-anchor="end" '
            f'font-size="10" fill="#aaa" dominant-baseline="middle">{val:,.0f}</text>\n'
        )

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
              style="background:#1a1a2e;border-radius:8px;">
      {grid_lines}
      <polyline points="{balance_points}" fill="none" stroke="#555" stroke-width="1.5"/>
      <polyline points="{equity_points}" fill="none" stroke="#00d4ff" stroke-width="2"/>
      <text x="{width // 2}" y="{height - 5}" text-anchor="middle"
            font-size="10" fill="#888">Candle index (sampled)</text>
      <text x="{margin_left + 10}" y="{margin_top + 12}" font-size="11" fill="#00d4ff">Equity</text>
      <text x="{margin_left + 80}" y="{margin_top + 12}" font-size="11" fill="#555">Balance</text>
    </svg>"""
    return svg


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_report(result: BacktestResult) -> str:
    """Generate a complete HTML report from backtest results.

    Returns the HTML string.
    """
    m = result.metrics
    monthly = _monthly_breakdown(result.trades)
    net_pnl = result.final_balance - result.initial_balance
    net_pnl_pct = (net_pnl / result.initial_balance * 100) if result.initial_balance else 0.0
    pnl_color = "#00e676" if net_pnl >= 0 else "#ff5252"

    equity_svg = _equity_curve_svg(result.equity_curve)

    # --- Monthly breakdown rows ---
    monthly_rows = ""
    for row in monthly:
        color = "#00e676" if row["pnl"] >= 0 else "#ff5252"
        monthly_rows += f"""<tr>
            <td>{row['month']}</td><td>{row['trades']}</td>
            <td>{row['wins']}</td><td>{row['losses']}</td>
            <td>{row['win_rate']}%</td>
            <td style="color:{color}">{row['pnl']:+.2f}</td>
            <td>{row['avg_pnl']:+.2f}</td>
        </tr>\n"""

    # --- Trade list rows (limit to 500 for performance) ---
    trade_rows = ""
    for i, t in enumerate(result.trades[:500]):
        pnl_val = t.pnl if t.pnl is not None else 0.0
        tc = "#00e676" if pnl_val >= 0 else "#ff5252"
        direction_badge = (
            '<span style="color:#00e676">BUY</span>'
            if t.direction == "buy"
            else '<span style="color:#ff5252">SELL</span>'
        )
        exit_reason = t.exit_reason or "-"
        exit_price = f"{t.exit_price:.5f}" if t.exit_price is not None else "-"
        exit_time = str(t.exit_time)[:19] if t.exit_time is not None else "-"
        trade_rows += f"""<tr>
            <td>{i + 1}</td>
            <td>{direction_badge}</td>
            <td>{t.entry_price:.5f}</td>
            <td>{str(t.entry_time)[:19]}</td>
            <td>{t.stop_loss:.5f}</td>
            <td>{t.take_profit:.5f}</td>
            <td>{exit_price}</td>
            <td>{exit_time}</td>
            <td>{exit_reason.upper()}</td>
            <td style="color:{tc}">{pnl_val:+.2f}</td>
            <td>{t.lot_size}</td>
        </tr>\n"""

    trades_note = ""
    if len(result.trades) > 500:
        trades_note = f'<p style="color:#888;">Showing 500 of {len(result.trades)} trades.</p>'

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Backtest Report &mdash; {html.escape(result.symbol)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f23; color: #e0e0e0; padding: 24px;
  }}
  h1 {{ color: #00d4ff; margin-bottom: 4px; }}
  h2 {{ color: #bb86fc; margin: 28px 0 12px; font-size: 1.2em; }}
  .subtitle {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .stat-card {{
    background: #1a1a2e; border-radius: 8px; padding: 16px;
    border-left: 3px solid #00d4ff;
  }}
  .stat-card .label {{ color: #888; font-size: 0.8em; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; }}
  table {{
    width: 100%; border-collapse: collapse; margin-top: 8px;
    font-size: 0.85em;
  }}
  th, td {{
    padding: 8px 10px; text-align: right; border-bottom: 1px solid #222;
  }}
  th {{ color: #888; text-transform: uppercase; font-size: 0.75em; letter-spacing: 0.5px; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: rgba(0,212,255,0.05); }}
  .chart-container {{ margin: 20px 0; }}
  .footer {{ color: #555; font-size: 0.75em; margin-top: 40px; text-align: center; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p class="subtitle">
  {html.escape(result.symbol)} &bull; {result.mode.upper()} mode &bull;
  {str(result.start_date)[:10]} to {str(result.end_date)[:10]} &bull;
  {m.get('total_trades', 0)} trades
</p>

<div class="summary-grid">
  <div class="stat-card">
    <div class="label">Net P&amp;L</div>
    <div class="value" style="color:{pnl_color}">{net_pnl:+,.2f} ({net_pnl_pct:+.1f}%)</div>
  </div>
  <div class="stat-card">
    <div class="label">Final Balance</div>
    <div class="value">{result.final_balance:,.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Win Rate</div>
    <div class="value">{m.get('win_rate', 0):.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Profit Factor</div>
    <div class="value">{m.get('profit_factor', 0):.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Max Drawdown</div>
    <div class="value" style="color:#ff5252">{m.get('max_drawdown_pct', 0):.2f}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Sharpe Ratio</div>
    <div class="value">{m.get('sharpe_ratio', 0):.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg R:R</div>
    <div class="value">{m.get('avg_rr', 0):.2f}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Duration</div>
    <div class="value">{m.get('avg_trade_duration_minutes', 0):.0f}m</div>
  </div>
  <div class="stat-card">
    <div class="label">Max Win / Max Loss</div>
    <div class="value">{m.get('max_win', 0):+.2f} / {m.get('max_loss', 0):+.2f}</div>
  </div>
</div>

<h2>Equity Curve</h2>
<div class="chart-container">
{equity_svg}
</div>

<h2>Monthly Breakdown</h2>
<table>
<thead>
<tr>
  <th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th>
  <th>Win Rate</th><th>P&amp;L</th><th>Avg P&amp;L</th>
</tr>
</thead>
<tbody>
{monthly_rows}
</tbody>
</table>

<h2>Trade List</h2>
{trades_note}
<table>
<thead>
<tr>
  <th>#</th><th>Dir</th><th>Entry</th><th>Entry Time</th>
  <th>SL</th><th>TP</th><th>Exit</th><th>Exit Time</th>
  <th>Reason</th><th>P&amp;L</th><th>Lots</th>
</tr>
</thead>
<tbody>
{trade_rows}
</tbody>
</table>

<div class="footer">
  Generated by TradeNoJutsu Backtester &bull;
  Initial Balance: {result.initial_balance:,.2f} &bull;
  Report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
</div>
</body>
</html>"""
    return html_str


def save_report(result: BacktestResult, filepath: str = "backtest_report.html") -> str:
    """Generate and save report to file. Returns the filepath."""
    content = generate_report(result)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("backtest_report_saved", filepath=str(path.resolve()), trades=len(result.trades))
    return str(path.resolve())
