"""TradeNoJutsu Agent -- entry point, startup sequence, and main analysis loop.

Responsibilities:
1. Load configuration from .env
2. Initialise PostgreSQL, Redis, and MT5 connections
3. Verify MT5 account info
4. Start background tasks (price feed, news, calendar, position manager)
5. Run the main analysis loop on every 1-minute candle close
6. Handle graceful shutdown (SIGINT / SIGTERM)
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from datetime import date, datetime, timezone

import redis.asyncio as aioredis
import structlog
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from agent.config import settings
from agent.data.mt5_feed import (
    get_account_info,
    get_candles,
    get_tick,
    init_mt5,
    shutdown_mt5,
)
from agent.data.news_feed import (
    cache_news,
    fetch_news,
    get_recent_news,
)
from agent.data.economic_calendar import (
    calendar_refresh_loop,
    fetch_calendar,
    get_upcoming_events,
)
from agent.signals.indicators import compute_all_indicators
from agent.signals.patterns import detect_all_patterns
from agent.signals.market_structure import get_market_context
from agent.signals.sentiment import get_sentiment
from agent.brain.analyst import analyze as analyst_analyze
from agent.brain.trade_planner import plan_trade
from agent.execution.mt5_executor import (
    close_all_positions,
    get_open_positions,
    get_order_history,
    send_market_order,
)
from agent.execution.position_manager import manage_positions
from agent.execution.circuit_breaker import circuit_breaker
from agent.monitoring.trade_journal import (
    get_consecutive_losses,
    get_daily_summary,
    get_recent_trades_for_symbol,
    record_trade,
)
from agent.db.session import async_session, engine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Set up structlog with JSON rendering for production, console for dev."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None
_scheduler: AsyncIOScheduler | None = None
_shutdown_event = asyncio.Event()


# =========================================================================
# 1. INITIALISATION HELPERS
# =========================================================================

async def _init_redis() -> aioredis.Redis:
    """Create and verify the async Redis connection."""
    client = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
    )
    await client.ping()
    log.info("redis.connected", url=settings.REDIS_URL)
    return client


async def _init_postgres() -> None:
    """Verify that the PostgreSQL connection pool is healthy."""
    async with engine.connect() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )
    log.info("postgres.connected", url=settings.DATABASE_URL)


def _init_mt5_or_die() -> None:
    """Initialise MT5 and abort if it fails."""
    if not init_mt5():
        log.critical("mt5.init_failed -- aborting startup")
        sys.exit(1)

    account = get_account_info()
    if not account:
        log.critical("mt5.account_info_unavailable -- aborting startup")
        shutdown_mt5()
        sys.exit(1)

    balance = account.get("balance", 0)
    equity = account.get("equity", 0)
    circuit_breaker.update_peak_balance(balance)

    log.info(
        "mt5.account_verified",
        balance=balance,
        equity=equity,
        margin=account.get("margin"),
        free_margin=account.get("free_margin"),
    )


# =========================================================================
# 2. BACKGROUND TASKS
# =========================================================================

async def _price_feed_tick() -> None:
    """Fetch latest ticks for every symbol and publish to Redis."""
    global _redis
    if _redis is None:
        return

    for symbol in settings.symbols_list:
        try:
            tick = get_tick(symbol)
            if tick:
                import json
                key = f"tick:{symbol}"
                await _redis.set(key, json.dumps(tick), ex=30)
                await _redis.publish(f"tick_update:{symbol}", json.dumps(tick))
        except Exception:
            log.exception("price_feed.tick_error", symbol=symbol)


async def _price_feed_candles() -> None:
    """Fetch candles for every symbol/TF combo and cache in Redis."""
    global _redis
    if _redis is None:
        return

    import json

    for symbol in settings.symbols_list:
        for tf in settings.timeframes_list:
            try:
                df = get_candles(symbol, tf, settings.LOOKBACK_CANDLES)
                if df.empty:
                    continue
                records = df.copy()
                records["time"] = records["time"].astype(str)
                payload = records.to_json(orient="records")
                key = f"candles:{symbol}:{tf}"
                await _redis.set(key, payload)
                await _redis.publish(f"candles_update:{symbol}:{tf}", payload)
            except Exception:
                log.exception("price_feed.candle_error", symbol=symbol, tf=tf)


async def _news_feed_tick() -> None:
    """Poll for news and cache in Redis."""
    global _redis
    if _redis is None:
        return
    try:
        articles = await fetch_news()
        if articles:
            await cache_news(_redis, articles)
    except Exception:
        log.exception("news_feed.error")


async def _calendar_refresh() -> None:
    """Refresh the economic calendar cache."""
    global _redis
    if _redis is None:
        return
    try:
        await fetch_calendar(_redis)
    except Exception:
        log.exception("calendar_refresh.error")


async def _position_manager_tick() -> None:
    """Run position management (trailing stops, partial closes, time exits)."""
    try:
        # Compute current ATR values for each symbol with open positions
        atr_values: dict[str, float] = {}
        for symbol in settings.symbols_list:
            df = get_candles(symbol, "M1", 30)
            if not df.empty and "atr" not in df.columns:
                from agent.signals.indicators import compute_atr
                df = compute_atr(df, period=14)
            if not df.empty and "atr" in df.columns:
                last_atr = df["atr"].dropna().iloc[-1] if not df["atr"].dropna().empty else 0.0
                atr_values[symbol] = float(last_atr)

        manage_positions(atr_values)
    except Exception:
        log.exception("position_manager.error")


def _start_websocket_server() -> None:
    """Start the FastAPI WebSocket server in a background thread.

    Uses uvicorn with a new event loop so it does not interfere with the
    main asyncio loop.
    """
    try:
        from agent.monitoring.websocket_server import app as ws_app  # type: ignore[import-untyped]

        config = uvicorn.Config(
            ws_app,
            host="0.0.0.0",
            port=settings.WS_PORT,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.run()
    except ImportError:
        log.warning(
            "websocket_server.not_available",
            hint="agent.monitoring.websocket_server module not found -- skipping",
        )
    except Exception:
        log.exception("websocket_server.crashed")


# =========================================================================
# 3. MAIN ANALYSIS LOOP
# =========================================================================

async def _run_analysis_cycle() -> None:
    """Analyse every configured symbol and execute qualifying trades.

    Triggered on each new 1-minute candle close.
    """
    global _redis

    for symbol in settings.symbols_list:
        try:
            await _analyse_symbol(symbol)
        except Exception:
            log.exception("analysis.symbol_error", symbol=symbol)


async def _analyse_symbol(symbol: str) -> None:
    """Full analysis pipeline for a single symbol."""
    global _redis

    # -- Market data -------------------------------------------------------
    candles_1m = get_candles(symbol, "M1", settings.LOOKBACK_CANDLES)
    candles_3m = get_candles(symbol, "M3", settings.LOOKBACK_CANDLES)

    if candles_1m.empty or candles_3m.empty:
        log.warning("analysis.insufficient_data", symbol=symbol)
        return

    # -- Indicators --------------------------------------------------------
    candles_1m = compute_all_indicators(candles_1m)
    candles_3m = compute_all_indicators(candles_3m)

    # -- Patterns ----------------------------------------------------------
    patterns = detect_all_patterns(candles_1m)

    # -- Market structure --------------------------------------------------
    structure = get_market_context(symbol, candles_3m)

    # -- Sentiment ---------------------------------------------------------
    sentiment = None
    if _redis is not None:
        try:
            headlines = await get_recent_news(_redis, symbol, minutes=30)
            sentiment = await get_sentiment(_redis, symbol, headlines)
        except Exception:
            log.exception("analysis.sentiment_error", symbol=symbol)

    # -- Account info ------------------------------------------------------
    account = get_account_info()
    if not account:
        log.warning("analysis.no_account_info", symbol=symbol)
        return

    balance = account.get("balance", 0.0)
    circuit_breaker.update_peak_balance(balance)

    # -- Open positions and recent trades ----------------------------------
    open_positions = get_open_positions()
    recent_trades: list[dict] = []
    consecutive_losses = 0
    consecutive_wins = 0
    daily_pnl = 0.0

    async with async_session() as db:
        recent_trades = await get_recent_trades_for_symbol(db, symbol, limit=10)
        consecutive_losses = await get_consecutive_losses(db, symbol)

        # Count consecutive wins
        for t in recent_trades:
            pnl = t.get("pnl", 0) or 0
            if pnl > 0:
                consecutive_wins += 1
            else:
                break

        # Daily P&L
        today = date.today()
        summary = await get_daily_summary(db, today)
        daily_pnl = summary.get("net_pnl", 0.0) or 0.0

    # -- Circuit breaker check ---------------------------------------------
    tick = get_tick(symbol)
    current_spread = tick.get("spread", 0.0) if tick else 0.0

    # Compute average spread from recent candles
    avg_spread = 0.0
    if "spread" in candles_1m.columns:
        avg_spread = float(candles_1m["spread"].mean()) if not candles_1m["spread"].isna().all() else 0.0

    # Upcoming economic events for time restrictions
    upcoming_events_raw: list[dict] = []
    if _redis is not None:
        try:
            events = await get_upcoming_events(_redis, hours=1)
            upcoming_events_raw = [
                {
                    "title": ev.name,
                    "impact": ev.impact,
                    "datetime": ev.datetime_utc,
                    "pre_minutes": 15,
                    "post_minutes": 15,
                }
                for ev in events
            ]
        except Exception:
            log.exception("analysis.calendar_error", symbol=symbol)

    can_trade, reason = circuit_breaker.can_trade(
        symbol=symbol,
        current_spread=current_spread,
        avg_spread=avg_spread,
        daily_pnl=daily_pnl,
        account_info=account,
        upcoming_events=upcoming_events_raw,
    )

    if not can_trade:
        log.info("analysis.trading_blocked", symbol=symbol, reason=reason)
        return

    # -- AI Analysis -------------------------------------------------------
    decision = await analyst_analyze(
        symbol=symbol,
        candles_1m=candles_1m,
        candles_3m=candles_3m,
        patterns=patterns,
        market_context=structure,
        sentiment=sentiment,
        account_info=account,
        open_positions=open_positions,
        recent_trades=recent_trades,
        daily_pnl=daily_pnl,
    )

    log.info(
        "analysis.decision",
        symbol=symbol,
        action=decision.action,
        confidence=decision.confidence,
        reasoning=decision.reasoning[:120] if decision.reasoning else "",
    )

    # -- Trade execution ---------------------------------------------------
    if decision.action == "wait" or decision.confidence < 70:
        return

    max_daily_loss = balance * (settings.MAX_DAILY_LOSS_PCT / 100.0)
    daily_loss_remaining = max_daily_loss - abs(min(daily_pnl, 0.0))

    order = plan_trade(
        decision=decision,
        context=structure,
        account_info=account,
        open_positions=open_positions,
        consecutive_losses=consecutive_losses,
        consecutive_wins=consecutive_wins,
        daily_loss_remaining=daily_loss_remaining,
    )

    if order is None:
        log.info("analysis.trade_plan_rejected", symbol=symbol)
        return

    # Execute the order
    result = send_market_order(
        symbol=order.symbol,
        direction=order.direction,
        lot=order.lot_size,
        sl=order.stop_loss,
        tp=order.take_profit,
        comment=order.comment,
        magic=order.magic_number,
    )

    # Record in the trade journal
    if result.success:
        log.info(
            "analysis.order_filled",
            symbol=order.symbol,
            direction=order.direction,
            ticket=result.ticket,
            price=result.price_filled,
            lot=order.lot_size,
        )

        trade_data = {
            "symbol": order.symbol,
            "direction": order.direction,
            "entry_price": result.price_filled or order.entry_price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "lot_size": order.lot_size,
            "entry_time": result.time or datetime.now(timezone.utc),
            "status": "open",
            "confidence": order.confidence,
            "risk_pct": order.risk_pct,
            "rr_planned": order.rr_planned,
            "magic_number": order.magic_number,
            "trend_3m": structure.trend,
            "atr_at_entry": structure.atr,
            "session": structure.session,
            "sentiment_score": sentiment.score if sentiment else None,
            "patterns_detected": [p.type for p in patterns[:5]],
            "ai_reasoning": decision.reasoning,
        }

        async with async_session() as db:
            await record_trade(db, trade_data)
    else:
        log.error(
            "analysis.order_failed",
            symbol=order.symbol,
            error_code=result.error_code,
            error_message=result.error_message,
        )


# =========================================================================
# 4. SHUTDOWN
# =========================================================================

async def _graceful_shutdown(close_positions: bool = False) -> None:
    """Clean up all resources."""
    global _redis, _scheduler

    log.info("shutdown.starting", close_positions=close_positions)

    # Stop the scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("shutdown.scheduler_stopped")

    # Optionally close all open positions
    if close_positions:
        try:
            closed = close_all_positions()
            log.info("shutdown.positions_closed", count=closed)
        except Exception:
            log.exception("shutdown.close_positions_error")

    # Flush Redis buffers and close
    if _redis is not None:
        try:
            await _redis.aclose()
            log.info("shutdown.redis_closed")
        except Exception:
            log.exception("shutdown.redis_close_error")

    # Close MT5
    try:
        shutdown_mt5()
    except Exception:
        log.exception("shutdown.mt5_error")

    # Close PostgreSQL engine
    try:
        await engine.dispose()
        log.info("shutdown.postgres_closed")
    except Exception:
        log.exception("shutdown.postgres_close_error")

    log.info("shutdown.complete")


def _signal_handler(sig: signal.Signals) -> None:
    """Handle SIGINT / SIGTERM by setting the shutdown event."""
    log.info("signal.received", signal=sig.name)
    _shutdown_event.set()


# =========================================================================
# 5. MAIN ENTRY POINT
# =========================================================================

async def main() -> None:
    """Full startup sequence, background tasks, analysis loop, and shutdown."""
    global _redis, _scheduler

    # -- 1. Logging --------------------------------------------------------
    _configure_logging()
    log.info("startup.begin", symbols=settings.symbols_list, timeframes=settings.timeframes_list)

    # -- 2. Signal handlers ------------------------------------------------
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    # -- 3. Connections ----------------------------------------------------
    try:
        await _init_postgres()
    except Exception:
        log.exception("startup.postgres_failed")
        sys.exit(1)

    try:
        _redis = await _init_redis()
    except Exception:
        log.exception("startup.redis_failed")
        sys.exit(1)

    try:
        _init_mt5_or_die()
    except SystemExit:
        raise
    except Exception:
        log.exception("startup.mt5_failed")
        sys.exit(1)

    # -- 4. Seed initial data ----------------------------------------------
    try:
        await fetch_calendar(_redis)
        log.info("startup.calendar_seeded")
    except Exception:
        log.warning("startup.calendar_seed_failed")

    try:
        articles = await fetch_news()
        if articles:
            await cache_news(_redis, articles)
        log.info("startup.news_seeded", count=len(articles))
    except Exception:
        log.warning("startup.news_seed_failed")

    # -- 5. APScheduler background jobs ------------------------------------
    _scheduler = AsyncIOScheduler()

    # Price feed: ticks every 1 second, candles every 5 seconds
    _scheduler.add_job(
        _price_feed_tick,
        "interval",
        seconds=1,
        id="price_feed_tick",
        max_instances=1,
        misfire_grace_time=5,
    )
    _scheduler.add_job(
        _price_feed_candles,
        "interval",
        seconds=5,
        id="price_feed_candles",
        max_instances=1,
        misfire_grace_time=10,
    )

    # News feed: every 60 seconds
    _scheduler.add_job(
        _news_feed_tick,
        "interval",
        seconds=settings.NEWS_CHECK_INTERVAL_SEC,
        id="news_feed",
        max_instances=1,
        misfire_grace_time=30,
    )

    # Economic calendar: every hour
    _scheduler.add_job(
        _calendar_refresh,
        "interval",
        hours=1,
        id="calendar_refresh",
        max_instances=1,
        misfire_grace_time=120,
    )

    # Position manager: every 5 seconds
    _scheduler.add_job(
        _position_manager_tick,
        "interval",
        seconds=5,
        id="position_manager",
        max_instances=1,
        misfire_grace_time=10,
    )

    # Main analysis loop: every 60 seconds (1-minute candle close)
    _scheduler.add_job(
        _run_analysis_cycle,
        "interval",
        seconds=60,
        id="analysis_loop",
        max_instances=1,
        misfire_grace_time=30,
    )

    _scheduler.start()
    log.info(
        "startup.scheduler_running",
        jobs=[j.id for j in _scheduler.get_jobs()],
    )

    # -- 6. WebSocket server in a daemon thread ----------------------------
    ws_thread = threading.Thread(
        target=_start_websocket_server,
        name="ws-server",
        daemon=True,
    )
    ws_thread.start()
    log.info("startup.websocket_server_started", port=settings.WS_PORT)

    # -- 7. Run initial analysis cycle right away --------------------------
    try:
        await _run_analysis_cycle()
    except Exception:
        log.exception("startup.initial_analysis_error")

    log.info("startup.complete", msg="Agent is live")

    # -- 8. Wait for shutdown signal ---------------------------------------
    await _shutdown_event.wait()

    # -- 9. Graceful shutdown ----------------------------------------------
    await _graceful_shutdown(close_positions=False)


# =========================================================================
# Script entry
# =========================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
