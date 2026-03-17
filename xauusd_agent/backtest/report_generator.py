"""
HTML backtest report generator for the XAUUSD trading agent.

Produces a self-contained HTML file with inline SVG charts, summary tables,
ratchet effectiveness statistics, and per-regime/bias breakdowns. No
external CSS or JS dependencies are required.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xauusd_agent.infra.logger import get_logger

if TYPE_CHECKING:
    from xauusd_agent.backtest.nautilus_backtest import BacktestResult

logger = get_logger(__name__)


class BacktestReportGenerator:
    """Generate self-contained HTML backtest reports."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(
        self,
        result: BacktestResult,
        output_path: str = "backtest_report.html",
    ) -> str:
        """Generate a comprehensive HTML report.

        Sections:
            - Summary statistics table
            - Equity curve chart (inline SVG)
            - Drawdown chart (inline SVG)
            - Trade distribution histogram (inline SVG)
            - Ratchet statistics
            - Win rate by HTF bias direction
            - Win rate by market regime
            - Signal score vs outcome chart
            - Monthly P&L breakdown table

        Parameters
        ----------
        result:
            A :class:`BacktestResult` instance from the backtest engine.
        output_path:
            Filesystem path for the output HTML file.

        Returns
        -------
        str
            The absolute path of the written HTML file.
        """
        sections = [
            self._generate_header(),
            self._generate_summary_table(result),
            self._generate_equity_svg(result.equity_curve),
            self._generate_drawdown_svg(result.equity_curve),
            self._generate_trade_distribution_svg(result.trades),
            self._generate_ratchet_stats(result),
            self._generate_bias_table(result),
            self._generate_regime_table(result),
            self._generate_score_vs_outcome(result),
            self._generate_monthly_pnl_table(result),
            self._generate_footer(),
        ]

        html_content = "\n".join(sections)
        out = Path(output_path)
        out.write_text(html_content, encoding="utf-8")

        logger.info("Backtest report written to %s", out.resolve())
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Header / Footer
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_header() -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>XAUUSD Backtest Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 0.5rem; }}
  h2 {{ color: #495057; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: right; }}
  th {{ background: #e9ecef; text-align: left; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .metric-label {{ text-align: left; font-weight: 600; }}
  .positive {{ color: #198754; }}
  .negative {{ color: #dc3545; }}
  .neutral {{ color: #6c757d; }}
  .chart-container {{ margin: 1.5rem 0; background: #fff; border: 1px solid #dee2e6;
                      border-radius: 4px; padding: 1rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1rem; margin: 1rem 0; }}
  .stat-card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px;
                padding: 1rem; text-align: center; }}
  .stat-card .value {{ font-size: 1.5rem; font-weight: 700; }}
  .stat-card .label {{ font-size: 0.85rem; color: #6c757d; }}
  svg {{ max-width: 100%; height: auto; }}
  .timestamp {{ color: #6c757d; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>XAUUSD Backtest Report</h1>
<p class="timestamp">Generated: {timestamp}</p>
"""

    @staticmethod
    def _generate_footer() -> str:
        return """
</body>
</html>"""

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary_table(result: BacktestResult) -> str:
        def _fmt_pct(val: float) -> str:
            return f"{val * 100:.1f}%"

        def _fmt_usd(val: float) -> str:
            cls = "positive" if val > 0 else "negative" if val < 0 else "neutral"
            return f'<span class="{cls}">${val:,.2f}</span>'

        def _fmt_ratio(val: float) -> str:
            cls = "positive" if val > 1.0 else "negative" if val < 1.0 else "neutral"
            return f'<span class="{cls}">{val:.2f}</span>'

        rows = [
            ("Total Trades", str(result.total_trades)),
            ("Wins / Losses", f"{result.wins} / {result.losses}"),
            ("Win Rate", _fmt_pct(result.win_rate)),
            ("Total Profit", _fmt_usd(result.total_profit)),
            ("Profit Factor", _fmt_ratio(result.profit_factor)),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}"),
            ("Max Drawdown", _fmt_pct(result.max_drawdown_pct)),
            ("Avg Winner (R)", f"{result.avg_winner_R:.2f}R"),
            ("Avg Loser (R)", f"{result.avg_loser_R:.2f}R"),
        ]

        table_rows = "\n".join(
            f'  <tr><td class="metric-label">{label}</td><td>{value}</td></tr>'
            for label, value in rows
        )

        return f"""
<h2>Summary Statistics</h2>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>
{table_rows}
  </tbody>
</table>
"""

    # ------------------------------------------------------------------
    # Equity curve SVG
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_equity_svg(equity_curve: list[dict]) -> str:
        if not equity_curve:
            return "<h2>Equity Curve</h2><p>No equity data available.</p>"

        width = 900
        height = 300
        margin = {"top": 20, "right": 20, "bottom": 30, "left": 70}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        equities = [e["equity"] for e in equity_curve]
        n = len(equities)

        if n < 2:
            return "<h2>Equity Curve</h2><p>Insufficient data.</p>"

        y_min = min(equities) * 0.98
        y_max = max(equities) * 1.02
        y_range = y_max - y_min if y_max != y_min else 1.0

        points: list[str] = []
        for i, eq in enumerate(equities):
            x = margin["left"] + (i / (n - 1)) * plot_w
            y = margin["top"] + plot_h - ((eq - y_min) / y_range) * plot_h
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)

        # Y-axis labels
        y_labels = ""
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            val = y_min + frac * y_range
            y_pos = margin["top"] + plot_h - frac * plot_h
            y_labels += (
                f'<text x="{margin["left"] - 5}" y="{y_pos:.0f}" '
                f'text-anchor="end" font-size="11" fill="#6c757d">'
                f"${val:,.0f}</text>\n"
            )
            y_labels += (
                f'<line x1="{margin["left"]}" y1="{y_pos:.0f}" '
                f'x2="{width - margin["right"]}" y2="{y_pos:.0f}" '
                f'stroke="#e9ecef" stroke-width="1"/>\n'
            )

        return f"""
<h2>Equity Curve</h2>
<div class="chart-container">
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  {y_labels}
  <polyline points="{polyline}" fill="none" stroke="#0d6efd" stroke-width="1.5"/>
  <!-- Axes -->
  <line x1="{margin['left']}" y1="{margin['top']}" x2="{margin['left']}" y2="{margin['top'] + plot_h}"
        stroke="#212529" stroke-width="1"/>
  <line x1="{margin['left']}" y1="{margin['top'] + plot_h}" x2="{width - margin['right']}" y2="{margin['top'] + plot_h}"
        stroke="#212529" stroke-width="1"/>
</svg>
</div>
"""

    # ------------------------------------------------------------------
    # Drawdown chart SVG
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_drawdown_svg(equity_curve: list[dict]) -> str:
        if not equity_curve:
            return "<h2>Drawdown</h2><p>No drawdown data.</p>"

        width = 900
        height = 200
        margin = {"top": 20, "right": 20, "bottom": 30, "left": 70}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        dd_values = [e.get("drawdown_pct", 0.0) for e in equity_curve]
        n = len(dd_values)

        if n < 2:
            return "<h2>Drawdown</h2><p>Insufficient data.</p>"

        max_dd = max(dd_values) if dd_values else 0.01
        if max_dd == 0:
            max_dd = 0.01

        points: list[str] = []
        for i, dd in enumerate(dd_values):
            x = margin["left"] + (i / (n - 1)) * plot_w
            y = margin["top"] + (dd / max_dd) * plot_h
            points.append(f"{x:.1f},{y:.1f}")

        # Fill area
        fill_points = (
            f'{margin["left"]:.1f},{margin["top"]:.1f} '
            + " ".join(points)
            + f' {margin["left"] + plot_w:.1f},{margin["top"]:.1f}'
        )

        return f"""
<h2>Drawdown</h2>
<div class="chart-container">
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <polygon points="{fill_points}" fill="rgba(220,53,69,0.2)" stroke="none"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="#dc3545" stroke-width="1.5"/>
  <text x="{margin['left'] - 5}" y="{margin['top'] + 5}" text-anchor="end"
        font-size="11" fill="#6c757d">0%</text>
  <text x="{margin['left'] - 5}" y="{margin['top'] + plot_h}" text-anchor="end"
        font-size="11" fill="#6c757d">-{max_dd * 100:.1f}%</text>
  <line x1="{margin['left']}" y1="{margin['top']}" x2="{margin['left']}" y2="{margin['top'] + plot_h}"
        stroke="#212529" stroke-width="1"/>
</svg>
</div>
"""

    # ------------------------------------------------------------------
    # Trade distribution histogram
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_trade_distribution_svg(trades: list[dict]) -> str:
        if not trades:
            return "<h2>Trade P&amp;L Distribution</h2><p>No trades.</p>"

        pnls = [t.get("pnl", 0.0) for t in trades]
        if not pnls:
            return "<h2>Trade P&amp;L Distribution</h2><p>No trades.</p>"

        width = 900
        height = 250
        margin = {"top": 20, "right": 20, "bottom": 40, "left": 70}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        # Build histogram bins
        num_bins = min(40, max(10, len(pnls) // 10))
        min_pnl = min(pnls)
        max_pnl = max(pnls)
        pnl_range = max_pnl - min_pnl if max_pnl != min_pnl else 1.0
        bin_width = pnl_range / num_bins

        bins = [0] * num_bins
        for p in pnls:
            idx = int((p - min_pnl) / bin_width)
            idx = min(idx, num_bins - 1)
            bins[idx] += 1

        max_count = max(bins) if bins else 1
        bar_w = plot_w / num_bins

        bars = ""
        for i, count in enumerate(bins):
            x = margin["left"] + i * bar_w
            bar_h = (count / max_count) * plot_h if max_count > 0 else 0
            y = margin["top"] + plot_h - bar_h
            bin_mid = min_pnl + (i + 0.5) * bin_width
            colour = "#198754" if bin_mid > 0 else "#dc3545"
            bars += (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" opacity="0.75"/>\n'
            )

        return f"""
<h2>Trade P&amp;L Distribution</h2>
<div class="chart-container">
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  {bars}
  <line x1="{margin['left']}" y1="{margin['top'] + plot_h}"
        x2="{width - margin['right']}" y2="{margin['top'] + plot_h}"
        stroke="#212529" stroke-width="1"/>
  <text x="{margin['left']}" y="{height - 5}" font-size="11" fill="#6c757d">
    ${min_pnl:,.0f}</text>
  <text x="{width - margin['right']}" y="{height - 5}" text-anchor="end"
        font-size="11" fill="#6c757d">${max_pnl:,.0f}</text>
  <text x="{margin['left'] + plot_w / 2}" y="{height - 5}" text-anchor="middle"
        font-size="11" fill="#6c757d">P&amp;L ($)</text>
</svg>
</div>
"""

    # ------------------------------------------------------------------
    # Ratchet statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_ratchet_stats(result: BacktestResult) -> str:
        cards = [
            ("Ratchet Saves", str(result.ratchet_saves)),
            (
                "Ratchet Save %",
                f"{result.ratchet_save_pct * 100:.1f}%",
            ),
            (
                "Avg Locked Profit",
                f"${result.avg_locked_profit_pct:,.2f}",
            ),
            (
                "Max Profit Given Back",
                f"${result.max_profit_given_back:,.2f}",
            ),
        ]

        card_html = "\n".join(
            f'<div class="stat-card">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f"</div>"
            for label, value in cards
        )

        return f"""
<h2>Ratchet SL Effectiveness</h2>
<div class="stat-grid">
{card_html}
</div>
<p>
Ratchet saves are trades that would have been full stop-loss exits
but were instead closed at a ratcheted (moved) SL level, locking in
partial profit.
</p>
"""

    # ------------------------------------------------------------------
    # Win rate by HTF bias direction
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_bias_table(result: BacktestResult) -> str:
        if not result.win_rate_by_bias:
            return "<h2>Win Rate by HTF Bias</h2><p>No data.</p>"

        order = ["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"]
        rows = ""
        for direction in order:
            wr = result.win_rate_by_bias.get(direction)
            if wr is not None:
                cls = "positive" if wr > 0.5 else "negative" if wr < 0.5 else "neutral"
                rows += (
                    f'  <tr><td class="metric-label">{direction}</td>'
                    f'<td class="{cls}">{wr * 100:.1f}%</td></tr>\n'
                )

        # Include any directions not in the standard order
        for direction, wr in result.win_rate_by_bias.items():
            if direction not in order:
                cls = "positive" if wr > 0.5 else "negative" if wr < 0.5 else "neutral"
                rows += (
                    f'  <tr><td class="metric-label">{html.escape(direction)}</td>'
                    f'<td class="{cls}">{wr * 100:.1f}%</td></tr>\n'
                )

        return f"""
<h2>Win Rate by HTF Bias Direction</h2>
<table>
  <thead><tr><th>Direction</th><th>Win Rate</th></tr></thead>
  <tbody>
{rows}  </tbody>
</table>
"""

    # ------------------------------------------------------------------
    # Win rate by regime
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_regime_table(result: BacktestResult) -> str:
        if not result.win_rate_by_regime:
            return "<h2>Win Rate by Regime</h2><p>No data.</p>"

        rows = ""
        for regime, wr in sorted(result.win_rate_by_regime.items()):
            cls = "positive" if wr > 0.5 else "negative" if wr < 0.5 else "neutral"
            rows += (
                f'  <tr><td class="metric-label">{html.escape(regime)}</td>'
                f'<td class="{cls}">{wr * 100:.1f}%</td></tr>\n'
            )

        return f"""
<h2>Win Rate by Market Regime</h2>
<table>
  <thead><tr><th>Regime</th><th>Win Rate</th></tr></thead>
  <tbody>
{rows}  </tbody>
</table>
"""

    # ------------------------------------------------------------------
    # Score vs outcome
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_score_vs_outcome(result: BacktestResult) -> str:
        if not result.score_vs_outcome:
            return "<h2>Signal Score vs Outcome</h2><p>No data.</p>"

        rows = ""
        for bucket, stats in sorted(result.score_vs_outcome.items()):
            count = stats.get("count", 0)
            avg_r = stats.get("avg_R", 0.0)
            wr = stats.get("win_rate", 0.0)
            r_cls = "positive" if avg_r > 0 else "negative" if avg_r < 0 else "neutral"
            wr_cls = "positive" if wr > 0.5 else "negative" if wr < 0.5 else "neutral"
            rows += (
                f'  <tr>'
                f'<td class="metric-label">{html.escape(bucket)}</td>'
                f"<td>{count}</td>"
                f'<td class="{r_cls}">{avg_r:.2f}R</td>'
                f'<td class="{wr_cls}">{wr * 100:.1f}%</td>'
                f"</tr>\n"
            )

        return f"""
<h2>Signal Score vs Outcome</h2>
<table>
  <thead>
    <tr><th>Score Bucket</th><th>Trades</th><th>Avg R</th><th>Win Rate</th></tr>
  </thead>
  <tbody>
{rows}  </tbody>
</table>
"""

    # ------------------------------------------------------------------
    # Monthly P&L breakdown
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_monthly_pnl_table(result: BacktestResult) -> str:
        monthly = getattr(result, "monthly_pnl", {})
        if not monthly:
            return "<h2>Monthly P&amp;L</h2><p>No data.</p>"

        rows = ""
        cumulative = 0.0
        for month in sorted(monthly.keys()):
            pnl = monthly[month]
            cumulative += pnl
            cls = "positive" if pnl > 0 else "negative" if pnl < 0 else "neutral"
            cum_cls = "positive" if cumulative > 0 else "negative" if cumulative < 0 else "neutral"
            rows += (
                f'  <tr>'
                f'<td class="metric-label">{html.escape(month)}</td>'
                f'<td class="{cls}">${pnl:,.2f}</td>'
                f'<td class="{cum_cls}">${cumulative:,.2f}</td>'
                f"</tr>\n"
            )

        return f"""
<h2>Monthly P&amp;L Breakdown</h2>
<table>
  <thead><tr><th>Month</th><th>P&amp;L</th><th>Cumulative</th></tr></thead>
  <tbody>
{rows}  </tbody>
</table>
"""
