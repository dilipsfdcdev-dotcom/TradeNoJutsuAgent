"""FastAPI dashboard for monitoring the trading agent.

Provides REST API endpoints and a simple HTML dashboard for:
- Live portfolio status
- Trade history and reasoning
- Performance metrics and charts
- AI thinking logs
- Learning cycle history
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from tradenojutsu.infra.database import get_connection, get_open_trades, get_recent_trades
from tradenojutsu.infra.logger import get_logger

logger = get_logger("dashboard")


def create_app():
    """Create the FastAPI dashboard application."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="TradeNoJutsu Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main dashboard page."""
        return DASHBOARD_HTML

    @app.get("/api/status")
    async def api_status():
        """Current agent status and open positions."""
        open_trades = get_open_trades()
        return {
            "status": "running",
            "open_positions": len(open_trades),
            "positions": open_trades,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/api/trades")
    async def api_trades(limit: int = 50, status: str = "all"):
        """Trade history."""
        conn = get_connection()
        if status == "open":
            rows = conn.execute("SELECT * FROM trades WHERE status='open' ORDER BY id DESC").fetchall()
        elif status == "closed":
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return {"trades": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/api/performance")
    async def api_performance():
        """Performance metrics."""
        trades = get_recent_trades(200)
        closed = [t for t in trades if t.get("pnl") is not None]
        if not closed:
            return {"metrics": {}, "message": "No closed trades yet"}

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        import numpy as np
        metrics = {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0,
            "sharpe_ratio": float(np.mean(pnls) / np.std(pnls) * (252 ** 0.5)) if len(pnls) > 1 and np.std(pnls) > 0 else 0,
        }

        # Equity curve
        equity = [10000]  # starting capital
        for p in reversed(pnls):  # oldest first
            equity.append(equity[-1] + p)

        return {"metrics": metrics, "equity_curve": equity}

    @app.get("/api/thinking")
    async def api_thinking(limit: int = 20):
        """AI reasoning/thinking logs."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM thinking_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return {"thinking_logs": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/api/learning")
    async def api_learning(limit: int = 20):
        """Learning cycle history."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM learning_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return {"learning_logs": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/api/signals")
    async def api_signals(limit: int = 30):
        """Signal history."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return {"signals": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/api/snapshots")
    async def api_snapshots(limit: int = 50):
        """Performance snapshots over time."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM performance_snapshots ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return {"snapshots": [dict(r) for r in rows]}

    return app


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server."""
    import uvicorn
    app = create_app()
    logger.info(f"Dashboard starting at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# Embedded HTML dashboard
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradeNoJutsu Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e1e4e8; }
        .header { background: linear-gradient(135deg, #1a1f36, #2d1b69); padding: 20px 30px; }
        .header h1 { font-size: 24px; color: #fff; }
        .header .subtitle { color: #8b949e; font-size: 14px; margin-top: 4px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card h2 { font-size: 16px; color: #58a6ff; margin-bottom: 12px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
        .metric:last-child { border-bottom: none; }
        .metric .label { color: #8b949e; }
        .metric .value { font-weight: 600; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #d29922; }
        .trade-row { padding: 10px 0; border-bottom: 1px solid #21262d; font-size: 14px; }
        .trade-row:last-child { border-bottom: none; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .badge.long { background: #0d3321; color: #3fb950; }
        .badge.short { background: #3d1214; color: #f85149; }
        .thinking-entry { padding: 10px 0; border-bottom: 1px solid #21262d; }
        .thinking-entry .decision { font-weight: 600; }
        .thinking-entry .thought { color: #8b949e; font-size: 13px; margin-top: 4px; }
        .refresh-btn { background: #238636; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; }
        .refresh-btn:hover { background: #2ea043; }
        .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .status-dot.active { background: #3fb950; }
        canvas { max-height: 200px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TradeNoJutsu Dashboard</h1>
        <div class="subtitle">Self-Thinking, Self-Learning AI Trading Agent</div>
    </div>
    <div class="container">
        <div style="margin-bottom: 20px;">
            <button class="refresh-btn" onclick="loadAll()">Refresh All</button>
            <span id="last-update" style="margin-left: 12px; color: #8b949e; font-size: 13px;"></span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Portfolio Status</h2>
                <div id="status-content">Loading...</div>
            </div>
            <div class="card">
                <h2>Performance</h2>
                <div id="performance-content">Loading...</div>
            </div>
            <div class="card">
                <h2>Equity Curve</h2>
                <canvas id="equity-chart" height="200"></canvas>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Recent Trades</h2>
                <div id="trades-content">Loading...</div>
            </div>
            <div class="card">
                <h2>AI Thinking Log</h2>
                <div id="thinking-content">Loading...</div>
            </div>
            <div class="card">
                <h2>Learning History</h2>
                <div id="learning-content">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        async function fetchJSON(url) {
            const r = await fetch(url);
            return r.json();
        }

        async function loadStatus() {
            const data = await fetchJSON('/api/status');
            const el = document.getElementById('status-content');
            let html = `<div class="metric"><span class="label">Status</span><span class="value"><span class="status-dot active"></span>Running</span></div>`;
            html += `<div class="metric"><span class="label">Open Positions</span><span class="value">${data.open_positions}</span></div>`;
            data.positions.forEach(p => {
                const cls = p.direction === 'long' ? 'long' : 'short';
                html += `<div class="trade-row"><span class="badge ${cls}">${p.direction.toUpperCase()}</span> ${p.symbol} @ ${parseFloat(p.entry_price).toFixed(4)}</div>`;
            });
            el.innerHTML = html;
        }

        async function loadPerformance() {
            const data = await fetchJSON('/api/performance');
            const el = document.getElementById('performance-content');
            const m = data.metrics;
            if (!m || !m.total_trades) { el.innerHTML = 'No data yet'; return; }
            const pnlClass = m.total_pnl > 0 ? 'positive' : m.total_pnl < 0 ? 'negative' : 'neutral';
            el.innerHTML = `
                <div class="metric"><span class="label">Total Trades</span><span class="value">${m.total_trades}</span></div>
                <div class="metric"><span class="label">Win Rate</span><span class="value">${(m.win_rate*100).toFixed(1)}%</span></div>
                <div class="metric"><span class="label">Profit Factor</span><span class="value">${m.profit_factor.toFixed(2)}</span></div>
                <div class="metric"><span class="label">Sharpe Ratio</span><span class="value">${m.sharpe_ratio.toFixed(2)}</span></div>
                <div class="metric"><span class="label">Total PnL</span><span class="value ${pnlClass}">${m.total_pnl >= 0 ? '+' : ''}${m.total_pnl.toFixed(2)}</span></div>
                <div class="metric"><span class="label">Best Trade</span><span class="value positive">+${m.best_trade.toFixed(2)}</span></div>
                <div class="metric"><span class="label">Worst Trade</span><span class="value negative">${m.worst_trade.toFixed(2)}</span></div>
            `;

            // Draw equity curve
            if (data.equity_curve && data.equity_curve.length > 1) {
                drawEquityCurve(data.equity_curve);
            }
        }

        function drawEquityCurve(data) {
            const canvas = document.getElementById('equity-chart');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = 400;
            ctx.scale(2, 2);

            const w = canvas.offsetWidth, h = 200;
            const min = Math.min(...data), max = Math.max(...data);
            const range = max - min || 1;

            ctx.clearRect(0, 0, w, h);
            ctx.beginPath();
            ctx.strokeStyle = data[data.length-1] >= data[0] ? '#3fb950' : '#f85149';
            ctx.lineWidth = 2;

            data.forEach((v, i) => {
                const x = (i / (data.length - 1)) * w;
                const y = h - ((v - min) / range) * (h - 20) - 10;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        async function loadTrades() {
            const data = await fetchJSON('/api/trades?limit=10&status=closed');
            const el = document.getElementById('trades-content');
            if (!data.trades.length) { el.innerHTML = 'No trades yet'; return; }
            el.innerHTML = data.trades.map(t => {
                const pnl = t.pnl || 0;
                const cls = t.direction === 'long' ? 'long' : 'short';
                const pnlCls = pnl > 0 ? 'positive' : 'negative';
                return `<div class="trade-row">
                    <span class="badge ${cls}">${t.direction.toUpperCase()}</span> ${t.symbol}
                    <span class="${pnlCls}" style="float:right">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>
                    <br><small style="color:#8b949e">${t.strategy || 'unknown'}</small>
                </div>`;
            }).join('');
        }

        async function loadThinking() {
            const data = await fetchJSON('/api/thinking?limit=5');
            const el = document.getElementById('thinking-content');
            if (!data.thinking_logs.length) { el.innerHTML = 'No thinking logs yet'; return; }
            el.innerHTML = data.thinking_logs.map(t => {
                const thought = (t.chain_of_thought || '').substring(0, 150);
                return `<div class="thinking-entry">
                    <span class="decision">${t.decision.toUpperCase()}</span> ${t.symbol || ''}
                    <span style="color:#8b949e;float:right">${(t.confidence*100||0).toFixed(0)}%</span>
                    <div class="thought">${thought}...</div>
                </div>`;
            }).join('');
        }

        async function loadLearning() {
            const data = await fetchJSON('/api/learning?limit=5');
            const el = document.getElementById('learning-content');
            if (!data.learning_logs.length) { el.innerHTML = 'No learning cycles yet'; return; }
            el.innerHTML = data.learning_logs.map(l => {
                return `<div class="trade-row">
                    <b>Cycle #${l.cycle_num}</b>: ${l.param_changed}<br>
                    <small style="color:#8b949e">${l.old_value} → ${l.new_value} | ${l.reason || ''}</small>
                </div>`;
            }).join('');
        }

        async function loadAll() {
            await Promise.all([loadStatus(), loadPerformance(), loadTrades(), loadThinking(), loadLearning()]);
            document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
        }

        loadAll();
        setInterval(loadAll, 30000);
    </script>
</body>
</html>"""
