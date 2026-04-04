"""FastAPI WebSocket server pushing live data to the dashboard."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func, case, desc
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import settings
from agent.db.models import Trade, DailySummary, AgentLog, Setting
from agent.db.session import get_session

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON helper – make Decimal / datetime / date / UUID serialisable
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert non-JSON-native types so ``json.dumps`` succeeds."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if hasattr(obj, "hex"):  # UUID
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections grouped by channel.

    Channels: ``prices``, ``trades``, ``signals``, ``metrics``, ``alerts``.
    """

    CHANNELS = ("prices", "trades", "signals", "metrics", "alerts")

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {
            ch: set() for ch in self.CHANNELS
        }
        # Per-client throttle tracking for the prices channel (1 msg/sec).
        self._last_price_send: dict[WebSocket, float] = {}

    async def connect(self, channel: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(channel, set()).add(ws)
        logger.info("ws.connect", channel=channel, clients=len(self._connections[channel]))

    def disconnect(self, channel: str, ws: WebSocket) -> None:
        self._connections.get(channel, set()).discard(ws)
        self._last_price_send.pop(ws, None)
        logger.info("ws.disconnect", channel=channel, clients=len(self._connections.get(channel, set())))

    async def broadcast(self, channel: str, data: Any) -> None:
        """Send *data* to every client on *channel*.

        For the ``prices`` channel the message is throttled to at most one
        send per second **per client**.
        """
        payload = _serialise(data)
        dead: list[WebSocket] = []
        now = time.monotonic()

        for ws in list(self._connections.get(channel, set())):
            # Per-client throttle for prices channel
            if channel == "prices":
                last = self._last_price_send.get(ws, 0.0)
                if now - last < 1.0:
                    continue
                self._last_price_send[ws] = now

            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(channel, ws)


# Module-level manager – importable by other modules to push data.
manager = ConnectionManager()


async def push(channel: str, data: Any) -> None:
    """Convenience wrapper around ``manager.broadcast``."""
    await manager.broadcast(channel, data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="TradeNoJutsu Agent API", version="1.0.0")

    # -- CORS --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # WebSocket endpoints
    # -----------------------------------------------------------------------

    async def _ws_loop(channel: str, ws: WebSocket) -> None:
        """Generic per-channel WebSocket lifecycle."""
        await manager.connect(channel, ws)
        try:
            while True:
                # Keep the connection alive; the client may send pings /
                # commands but we don't act on them here.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(channel, ws)

    @app.websocket("/ws/prices")
    async def ws_prices(ws: WebSocket) -> None:
        """Tick data for all symbols (throttled to 1/sec per client)."""
        await _ws_loop("prices", ws)

    @app.websocket("/ws/trades")
    async def ws_trades(ws: WebSocket) -> None:
        """Open position updates."""
        await _ws_loop("trades", ws)

    @app.websocket("/ws/signals")
    async def ws_signals(ws: WebSocket) -> None:
        """AI reasoning log (every analysis)."""
        await _ws_loop("signals", ws)

    @app.websocket("/ws/metrics")
    async def ws_metrics(ws: WebSocket) -> None:
        """Performance metrics (every 30 sec)."""
        await _ws_loop("metrics", ws)

    @app.websocket("/ws/alerts")
    async def ws_alerts(ws: WebSocket) -> None:
        """Circuit breaker events, errors."""
        await _ws_loop("alerts", ws)

    # -----------------------------------------------------------------------
    # REST endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/trades")
    async def get_trades(
        symbol: str | None = Query(None),
        status: str | None = Query(None),
        limit: int = Query(50, ge=1, le=500),
        session: AsyncSession = Depends(get_session),
    ) -> list[dict]:
        """Return recent trades, optionally filtered by symbol / status."""
        stmt = select(Trade).order_by(desc(Trade.created_at)).limit(limit)
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol.upper())
        if status:
            stmt = stmt.where(Trade.status == status.lower())
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [_trade_to_dict(t) for t in rows]

    @app.get("/api/metrics")
    async def get_metrics(
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        """Current performance metrics aggregated from trades."""
        result = await session.execute(
            select(
                func.count(Trade.id).label("total_trades"),
                func.count(case((Trade.status == "closed", Trade.id))).label("closed_trades"),
                func.count(case((Trade.pnl > 0, Trade.id))).label("winning"),
                func.count(case((Trade.pnl < 0, Trade.id))).label("losing"),
                func.coalesce(func.sum(Trade.pnl), 0).label("net_pnl"),
                func.coalesce(
                    func.sum(case((Trade.pnl > 0, Trade.pnl), else_=Decimal(0))),
                    0,
                ).label("gross_profit"),
                func.coalesce(
                    func.sum(case((Trade.pnl < 0, Trade.pnl), else_=Decimal(0))),
                    0,
                ).label("gross_loss"),
                func.avg(Trade.pnl).label("avg_pnl"),
            )
        )
        row = result.one()
        total = row.closed_trades or 0
        winning = row.winning or 0
        return _serialise(
            {
                "total_trades": row.total_trades,
                "closed_trades": total,
                "winning": winning,
                "losing": row.losing or 0,
                "win_rate": round(winning / total * 100, 1) if total else 0.0,
                "net_pnl": row.net_pnl,
                "gross_profit": row.gross_profit,
                "gross_loss": row.gross_loss,
                "avg_pnl": row.avg_pnl,
                "profit_factor": (
                    round(float(row.gross_profit) / abs(float(row.gross_loss)), 2)
                    if row.gross_loss and float(row.gross_loss) != 0
                    else None
                ),
            }
        )

    @app.get("/api/equity-curve")
    async def get_equity_curve(
        days: int = Query(30, ge=1, le=365),
        session: AsyncSession = Depends(get_session),
    ) -> list[dict]:
        """Daily equity curve from the DailySummary table."""
        since = date.today() - timedelta(days=days)
        result = await session.execute(
            select(DailySummary)
            .where(DailySummary.date >= since)
            .order_by(DailySummary.date)
        )
        rows = result.scalars().all()
        return [
            _serialise(
                {
                    "date": r.date,
                    "starting_balance": r.starting_balance,
                    "ending_balance": r.ending_balance,
                    "net_pnl": r.net_pnl,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "total_trades": r.total_trades,
                    "winning_trades": r.winning_trades,
                    "losing_trades": r.losing_trades,
                }
            )
            for r in rows
        ]

    @app.get("/api/backtest-results")
    async def get_backtest_results(
        session: AsyncSession = Depends(get_session),
    ) -> dict | None:
        """Return the latest backtest results stored in the settings table."""
        result = await session.execute(
            select(Setting).where(Setting.key == "backtest_results")
        )
        row = result.scalar_one_or_none()
        if row is None:
            return {"results": None}
        return _serialise({"results": row.value, "updated_at": row.updated_at})

    @app.post("/api/settings")
    async def update_settings(
        payload: dict,
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        """Update risk parameters.

        Accepted keys mirror ``Settings`` risk fields:
        ``MAX_RISK_PER_TRADE_PCT``, ``MAX_DAILY_LOSS_PCT``,
        ``MAX_OPEN_TRADES``, ``MAX_DRAWDOWN_PCT``, ``MIN_RR_RATIO``.
        """
        allowed = {
            "MAX_RISK_PER_TRADE_PCT",
            "MAX_DAILY_LOSS_PCT",
            "MAX_OPEN_TRADES",
            "MAX_DRAWDOWN_PCT",
            "MIN_RR_RATIO",
        }
        updated: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in allowed:
                continue
            # Persist to DB so it survives restarts.
            result = await session.execute(
                select(Setting).where(Setting.key == key)
            )
            setting = result.scalar_one_or_none()
            if setting is None:
                setting = Setting(key=key, value={"v": value})
                session.add(setting)
            else:
                setting.value = {"v": value}
            # Also hot-patch the in-memory settings object.
            if hasattr(settings, key):
                object.__setattr__(settings, key, type(getattr(settings, key))(value))
            updated[key] = value

        await session.commit()
        logger.info("settings.updated", changes=updated)
        return {"updated": updated}

    @app.get("/api/settings")
    async def get_all_settings(
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        """Return all current settings for the setup page."""
        result = await session.execute(select(Setting))
        rows = result.scalars().all()
        db_settings = {r.key: r.value for r in rows}
        return _serialise({
            "MT5_LOGIN": settings.MT5_LOGIN,
            "MT5_SERVER": settings.MT5_SERVER,
            "MT5_PATH": settings.MT5_PATH,
            "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY[:8] + "..." if settings.ANTHROPIC_API_KEY else "",
            "CLAUDE_MODEL": settings.CLAUDE_MODEL,
            "DATABASE_URL": settings.DATABASE_URL,
            "REDIS_URL": settings.REDIS_URL,
            "NEWS_API_KEY": settings.NEWS_API_KEY[:8] + "..." if settings.NEWS_API_KEY else "",
            "MAX_RISK_PER_TRADE_PCT": settings.MAX_RISK_PER_TRADE_PCT,
            "MAX_DAILY_LOSS_PCT": settings.MAX_DAILY_LOSS_PCT,
            "MAX_OPEN_TRADES": settings.MAX_OPEN_TRADES,
            "MAX_DRAWDOWN_PCT": settings.MAX_DRAWDOWN_PCT,
            "MIN_RR_RATIO": settings.MIN_RR_RATIO,
            "SPREAD_FILTER_MULTIPLIER": settings.SPREAD_FILTER_MULTIPLIER,
            "SYMBOLS": settings.SYMBOLS,
            "TIMEFRAMES": settings.TIMEFRAMES,
            "agent_paused": db_settings.get("agent_paused", {}).get("paused", False),
        })

    @app.post("/api/test-mt5")
    async def test_mt5(payload: dict) -> dict:
        """Test MT5 connection with provided credentials."""
        try:
            from agent.data.mt5_feed import init_mt5, get_account_info, shutdown_mt5
            # Temporarily set credentials
            login = payload.get("login", settings.MT5_LOGIN)
            if login:
                object.__setattr__(settings, "MT5_LOGIN", int(login))
            if payload.get("password"):
                object.__setattr__(settings, "MT5_PASSWORD", payload["password"])
            if payload.get("server"):
                object.__setattr__(settings, "MT5_SERVER", payload["server"])
            if payload.get("path"):
                object.__setattr__(settings, "MT5_PATH", payload["path"])

            connected = init_mt5()
            if connected:
                info = get_account_info()
                return _serialise({"connected": True, "account": info})
            return {"connected": False, "error": "Failed to connect to MT5"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    @app.post("/api/test-claude")
    async def test_claude(payload: dict) -> dict:
        """Test Claude API connection."""
        try:
            import anthropic
            api_key = payload.get("api_key", settings.ANTHROPIC_API_KEY)
            model = payload.get("model", settings.CLAUDE_MODEL)
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": "Say hello"}],
            )
            return {"connected": True, "model": model, "response": response.content[0].text}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    @app.post("/api/test-db")
    async def test_db(session: AsyncSession = Depends(get_session)) -> dict:
        """Test database connectivity."""
        try:
            await session.execute(select(func.count(Trade.id)))
            return {"postgresql": True, "error": None}
        except Exception as e:
            return {"postgresql": False, "error": str(e)}

    @app.get("/api/health")
    async def health() -> dict:
        """Basic health check endpoint."""
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/veto-status")
    async def get_veto_status() -> dict:
        """Return current veto register state."""
        try:
            from agent.brain.veto_register import veto_register
            all_vetoes = veto_register.get_all()
            return _serialise({
                k: v.to_dict() if hasattr(v, "to_dict") else str(v)
                for k, v in all_vetoes.items()
            })
        except Exception:
            return {}

    @app.post("/api/toggle")
    async def toggle_agent(
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        """Pause or resume the trading agent."""
        result = await session.execute(
            select(Setting).where(Setting.key == "agent_paused")
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = Setting(key="agent_paused", value={"paused": True})
            session.add(row)
        else:
            row.value = {"paused": not row.value.get("paused", False)}

        await session.commit()
        paused = row.value["paused"]
        logger.info("agent.toggle", paused=paused)

        # Broadcast an alert so the dashboard updates instantly.
        await push(
            "alerts",
            {"type": "agent_toggle", "paused": paused, "ts": datetime.now(timezone.utc).isoformat()},
        )
        return {"paused": paused}

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade_to_dict(t: Trade) -> dict:
    return _serialise(
        {
            "id": t.id,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "lot_size": t.lot_size,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "status": t.status,
            "pnl": t.pnl,
            "pnl_pips": t.pnl_pips,
            "rr_planned": t.rr_planned,
            "rr_actual": t.rr_actual,
            "confidence": t.confidence,
            "risk_pct": t.risk_pct,
            "ai_reasoning": t.ai_reasoning,
            "created_at": t.created_at,
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint (for ``uvicorn agent.monitoring.websocket_server:app``)
# ---------------------------------------------------------------------------

app = create_app()
