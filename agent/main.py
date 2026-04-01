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

# v2: ML modules and MTF analyzer
from agent.ml.xgboost_filter import XGBoostFilter
try:
    from agent.ml.lstm_model import LSTMConfidence
except ImportError:
    LSTMConfidence = None  # torch not installed — LSTM gate disabled
from agent.signals.mtf_analyzer import MTFAnalyzer, compute_features

# v3: Two-loop architecture — ML decides, Claude vetoes asynchronously
from agent.brain.rule_engine import RuleEngine, TradePlan
from agent.brain.veto_register import veto_register, VetoRegister
from agent.brain.claude_veto import claude_veto_scan, build_veto_context, veto_scanner_loop
from agent.brain.claude_reviewer import review_trade as claude_review_trade
from agent.brain.claude_reviewer import generate_weekly_summary as claude_weekly_summary
from agent.brain.self_review import review_trade as self_review_trade
from agent.brain.self_review import generate_weekly_summary
from agent.brain.model_monitor import ModelMonitor, model_monitor as model_monitor_singleton

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

# v2: Per-symbol ML models and MTF analyzer
_xgb_filters: dict[str, XGBoostFilter] = {}
_lstm_models: dict[str, LSTMConfidence] = {}
_mtf_analyzer: MTFAnalyzer | None = None

# v3: Two-loop architecture components
_rule_engine: RuleEngine | None = None
_model_monitor: ModelMonitor | None = None
_veto_scanner_task: asyncio.Task | None = None
_model_monitor_task: asyncio.Task | None = None
_trade_review_tasks: list[asyncio.Task] = []


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
    """Fetch latest ticks for every symbol and publish to Redis + WebSocket."""
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
                # Push to WebSocket for dashboard
                try:
                    from agent.monitoring.websocket_server import push
                    await push("prices", {"symbol": symbol, **tick})
                except Exception:
                    pass
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
    In v3 mode, uses the deterministic fast loop (<6ms per symbol, zero API calls).
    In v2 mode, uses the original three-brain hybrid with Claude in the hot path.
    """
    global _redis

    use_v3 = (
        settings.AGENT_MODE == "v3"
        and _rule_engine is not None
        and _mtf_analyzer is not None
    )

    for symbol in settings.symbols_list:
        try:
            if use_v3:
                await _analyse_symbol_v3(symbol)
            else:
                await _analyse_symbol(symbol)
        except Exception:
            log.exception("analysis.symbol_error", symbol=symbol)


async def _analyse_symbol(symbol: str) -> None:
    """Full analysis pipeline for a single symbol (v2 three-brain hybrid)."""
    global _redis, _mtf_analyzer, _xgb_filters, _lstm_models

    # -- Market data -------------------------------------------------------
    candles_1m = get_candles(symbol, "M1", settings.LOOKBACK_CANDLES)
    candles_3m = get_candles(symbol, "M3", settings.LOOKBACK_CANDLES)
    candles_15m = get_candles(symbol, "M15", settings.LOOKBACK_CANDLES)
    candles_1h = get_candles(symbol, "H1", settings.LOOKBACK_CANDLES)

    if candles_1m.empty or candles_3m.empty:
        log.warning("analysis.insufficient_data", symbol=symbol)
        return

    # -- Indicators --------------------------------------------------------
    candles_1m = compute_all_indicators(candles_1m)
    candles_3m = compute_all_indicators(candles_3m)
    if not candles_15m.empty:
        candles_15m = compute_all_indicators(candles_15m)
    if not candles_1h.empty:
        candles_1h = compute_all_indicators(candles_1h)

    # -- Patterns ----------------------------------------------------------
    patterns = detect_all_patterns(candles_1m)

    # -- Market structure --------------------------------------------------
    structure = get_market_context(symbol, candles_3m)

    # -- Tick data ---------------------------------------------------------
    tick = get_tick(symbol)

    # -- Sentiment ---------------------------------------------------------
    sentiment = None
    sentiment_score = 0.0
    if _redis is not None:
        try:
            headlines = await get_recent_news(_redis, symbol, minutes=30)
            sentiment = await get_sentiment(_redis, symbol, headlines)
            if sentiment is not None:
                sentiment_score = sentiment.score
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

    # =====================================================================
    # v2 Step 1: Update MTF state (always)
    # =====================================================================
    mtf_state = None
    xgb_result = None
    lstm_result = None
    features = None

    if _mtf_analyzer is not None:
        try:
            mtf_state = _mtf_analyzer.update(
                symbol, candles_1m, candles_3m, candles_15m, candles_1h, tick,
            )
        except Exception:
            log.exception("analysis.mtf_update_error", symbol=symbol)

    # v2 Step 2: If MTF gates didn't pass, skip ML brains
    if mtf_state is not None and not mtf_state.gates_passed:
        log.info("analysis.mtf_gates_blocked", symbol=symbol,
                 gate_details=getattr(mtf_state, "gate_details", ""))
        # Push mtf_state to WebSocket even when blocked
        await _publish_v2_state(symbol, mtf_state=mtf_state)
        return

    # =====================================================================
    # v2 Step 3: Brain 1 -- XGBoost filter
    # =====================================================================
    if mtf_state is not None and symbol in _xgb_filters:
        try:
            features = compute_features(
                mtf_state, candles_1m, candles_3m, candles_15m, candles_1h,
                account, sentiment_score,
            )
            xgb_result = _xgb_filters[symbol].predict(features["xgb_features"])
            log.info("analysis.xgb_result", symbol=symbol,
                     score=xgb_result["score"], passed=xgb_result["pass"])
            if not xgb_result["pass"]:
                await _publish_v2_state(symbol, mtf_state=mtf_state,
                                        xgb_result=xgb_result)
                return
        except Exception:
            log.exception("analysis.xgb_error", symbol=symbol)
            # On error, continue without XGBoost (graceful degradation)

    # =====================================================================
    # v2 Step 4: Brain 2 -- LSTM confidence
    # =====================================================================
    if mtf_state is not None and symbol in _lstm_models:
        try:
            if features is not None and "lstm_sequences" in features:
                lstm_result = _lstm_models[symbol].predict(features["lstm_sequences"])
                log.info("analysis.lstm_result", symbol=symbol,
                         direction=lstm_result["direction"],
                         confidence=lstm_result["confidence"],
                         regime=lstm_result["regime"],
                         passed=lstm_result["pass"])
                if not lstm_result["pass"]:
                    await _publish_v2_state(symbol, mtf_state=mtf_state,
                                            xgb_result=xgb_result,
                                            lstm_result=lstm_result)
                    return
        except Exception:
            log.exception("analysis.lstm_error", symbol=symbol)
            # On error, continue without LSTM (graceful degradation)

    # =====================================================================
    # v2 Step 5: Circuit breaker
    # =====================================================================
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
        try:
            from agent.monitoring.websocket_server import push
            await push("signals", {
                "symbol": symbol,
                "action": "blocked",
                "confidence": 0,
                "reasoning": reason or "Trading blocked",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        return

    # =====================================================================
    # v2 Step 6: Brain 3 -- Claude analysis (only ~5% of candles reach here)
    # =====================================================================
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
        mtf_state=mtf_state,
        xgb_result=xgb_result,
        lstm_result=lstm_result,
    )

    log.info(
        "analysis.decision",
        symbol=symbol,
        action=decision.action,
        confidence=decision.confidence,
        reasoning=decision.reasoning[:120] if decision.reasoning else "",
    )

    # Push signal to WebSocket for dashboard
    try:
        from agent.monitoring.websocket_server import push
        await push("signals", {
            "symbol": symbol,
            "action": decision.action,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning or "",
            "risk_score": getattr(decision, "risk_score", 0),
            "entry": getattr(decision, "entry", 0),
            "sl": getattr(decision, "sl", 0),
            "tp": getattr(decision, "tp", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass

    # Push full state to Redis
    await _publish_v2_state(
        symbol, mtf_state=mtf_state, xgb_result=xgb_result,
        lstm_result=lstm_result, decision=decision,
    )

    # =====================================================================
    # v2 Step 7: Trade execution
    # =====================================================================
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

    # =====================================================================
    # v2 Step 8: Record trade + ML label generation metadata
    # =====================================================================
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
            # v2: ML metadata for post-trade review and label generation
            "xgb_score": xgb_result["score"] if xgb_result else None,
            "lstm_confidence": lstm_result["confidence"] if lstm_result else None,
            "lstm_direction": lstm_result["direction"] if lstm_result else None,
            "lstm_regime": lstm_result["regime"] if lstm_result else None,
            "mtf_confluence": getattr(mtf_state, "confluence_score", None) if mtf_state else None,
            "lot_size_suggestion": decision.lot_size_suggestion,
            "risk_warnings": decision.risk_warnings,
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
# 3b. v2 HELPERS -- WebSocket publishing and ML retrain tasks
# =========================================================================

async def _publish_v2_state(
    symbol: str,
    *,
    mtf_state=None,
    xgb_result: dict | None = None,
    lstm_result: dict | None = None,
    decision=None,
) -> None:
    """Push MTF state and ML scores to the WebSocket for the dashboard."""
    global _redis
    if _redis is None:
        return
    try:
        import json as _json
        payload: dict = {"symbol": symbol}
        if mtf_state is not None:
            payload["mtf_state"] = {
                "gates_passed": getattr(mtf_state, "gates_passed", False),
                "confluence_score": getattr(mtf_state, "confluence_score", None),
                "setup_narrative": getattr(mtf_state, "setup_narrative", ""),
                "gate_details": getattr(mtf_state, "gate_details", ""),
                "timeframes": getattr(mtf_state, "timeframes", {}),
            }
        if xgb_result is not None:
            payload["xgb_result"] = xgb_result
        if lstm_result is not None:
            payload["lstm_result"] = lstm_result
        if decision is not None:
            payload["decision"] = {
                "action": decision.action,
                "confidence": decision.confidence,
                "lot_size_suggestion": getattr(decision, "lot_size_suggestion", "normal"),
                "risk_warnings": getattr(decision, "risk_warnings", ""),
            }
        await _redis.publish(f"v2_state:{symbol}", _json.dumps(payload))
    except Exception:
        log.exception("publish_v2_state.error", symbol=symbol)


# =========================================================================
# 3c. v3 FAST LOOP -- deterministic, <6ms per symbol, zero API calls
# =========================================================================

async def _analyse_symbol_v3(symbol: str) -> None:
    """v3 fast-loop analysis: ML decides, Claude vetoes asynchronously.

    Deterministic pipeline, <6ms per symbol, zero API calls.
    Steps: MTF update -> gate check -> features -> XGBoost -> LSTM
           -> rule engine trade plan -> veto register check -> execute.
    """
    global _redis, _mtf_analyzer, _xgb_filters, _lstm_models, _rule_engine, _model_monitor

    # -- Market data (cached from price feed background task) ----------------
    candles_1m = get_candles(symbol, "M1", settings.LOOKBACK_CANDLES)
    candles_3m = get_candles(symbol, "M3", settings.LOOKBACK_CANDLES)
    candles_15m = get_candles(symbol, "M15", settings.LOOKBACK_CANDLES)
    candles_1h = get_candles(symbol, "H1", settings.LOOKBACK_CANDLES)

    if candles_1m.empty or candles_3m.empty:
        log.warning("v3.insufficient_data", symbol=symbol)
        return

    tick = get_tick(symbol)
    if not tick:
        log.warning("v3.no_tick", symbol=symbol)
        return

    # -- Step 1: MTF state update (~2ms) ------------------------------------
    mtf_state = None
    try:
        mtf_state = _mtf_analyzer.update(
            symbol, candles_1m, candles_3m, candles_15m, candles_1h, tick,
        )
    except Exception:
        log.exception("v3.mtf_update_error", symbol=symbol)
        return

    # -- Step 2: Gate check (instant) ---------------------------------------
    if not mtf_state.gates_passed:
        log.debug("v3.gates_blocked", symbol=symbol,
                  gate_details=getattr(mtf_state, "gate_details", ""))
        await _publish_v2_state(symbol, mtf_state=mtf_state)
        return

    # -- Step 3: Feature computation (~1ms) ---------------------------------
    account = get_account_info()
    if not account:
        log.warning("v3.no_account_info", symbol=symbol)
        return

    balance = account.get("balance", 0.0)
    circuit_breaker.update_peak_balance(balance)

    sentiment_score = 0.0
    if _redis is not None:
        try:
            headlines = await get_recent_news(_redis, symbol, minutes=30)
            sentiment = await get_sentiment(_redis, symbol, headlines)
            if sentiment is not None:
                sentiment_score = sentiment.score
        except Exception:
            pass  # sentiment is optional for fast loop

    features = None
    try:
        features = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m, candles_1h,
            account, sentiment_score,
        )
    except Exception:
        log.exception("v3.feature_error", symbol=symbol)
        return

    # -- Step 4: XGBoost score (<1ms) ---------------------------------------
    xgb_result = None
    if symbol in _xgb_filters:
        try:
            xgb_result = _xgb_filters[symbol].predict(features["xgb_features"])
            if _model_monitor:
                _model_monitor.record_prediction("xgb", xgb_result["score"])
            if not xgb_result["pass"]:
                log.debug("v3.xgb_filtered", symbol=symbol, score=xgb_result["score"])
                await _publish_v2_state(symbol, mtf_state=mtf_state, xgb_result=xgb_result)
                return
        except Exception:
            log.exception("v3.xgb_error", symbol=symbol)
            return  # v3 is strict: ML must pass, no graceful degradation

    if xgb_result is None:
        log.warning("v3.no_xgb_model", symbol=symbol)
        return

    # -- Step 5: LSTM score (~3ms) ------------------------------------------
    lstm_result = None
    if symbol in _lstm_models and features is not None and "lstm_sequences" in features:
        try:
            lstm_result = _lstm_models[symbol].predict(features["lstm_sequences"])
            if _model_monitor:
                _model_monitor.record_prediction("lstm", lstm_result["confidence"])
            if not lstm_result["pass"]:
                log.debug("v3.lstm_filtered", symbol=symbol,
                          confidence=lstm_result["confidence"],
                          direction=lstm_result["direction"])
                await _publish_v2_state(symbol, mtf_state=mtf_state,
                                        xgb_result=xgb_result, lstm_result=lstm_result)
                return
        except Exception:
            log.exception("v3.lstm_error", symbol=symbol)
            return

    if lstm_result is None:
        log.warning("v3.no_lstm_model", symbol=symbol)
        return

    # -- Step 6: Deterministic trade plan (instant) -------------------------
    plan = _rule_engine.compute_trade(symbol, mtf_state, xgb_result, lstm_result, account, tick)
    if plan is None:
        log.debug("v3.no_trade_plan", symbol=symbol)
        await _publish_v2_state(symbol, mtf_state=mtf_state,
                                xgb_result=xgb_result, lstm_result=lstm_result)
        return

    # -- Step 7: Read veto register (instant -- dict lookup) ----------------
    veto = veto_register.check(symbol)
    if veto.blocked:
        log.info("v3.claude_veto", symbol=symbol, reason=veto.reason)
        return

    # -- Step 8: Apply risk modifier from veto ------------------------------
    if veto.risk_modifier != 1.0:
        original_lot = plan.lot_size
        plan.lot_size = round(plan.lot_size * veto.risk_modifier, 2)
        log.info("v3.veto_risk_adjusted", symbol=symbol,
                 original_lot=original_lot, adjusted_lot=plan.lot_size,
                 modifier=veto.risk_modifier, reason=veto.reason)

    if plan.lot_size <= 0:
        log.info("v3.lot_size_zero_after_veto", symbol=symbol)
        return

    # -- Circuit breaker check (fast, no API calls) -------------------------
    current_spread = tick.get("spread", 0.0)
    avg_spread = 0.0
    if "spread" in candles_1m.columns:
        avg_spread = float(candles_1m["spread"].mean()) if not candles_1m["spread"].isna().all() else 0.0

    daily_pnl = 0.0
    async with async_session() as db:
        today = date.today()
        summary = await get_daily_summary(db, today)
        daily_pnl = summary.get("net_pnl", 0.0) or 0.0

    upcoming_events_raw: list[dict] = []
    if _redis is not None:
        try:
            events = await get_upcoming_events(_redis, hours=1)
            upcoming_events_raw = [
                {"title": ev.name, "impact": ev.impact,
                 "datetime": ev.datetime_utc, "pre_minutes": 15, "post_minutes": 15}
                for ev in events
            ]
        except Exception:
            pass

    can_trade, reason = circuit_breaker.can_trade(
        symbol=symbol, current_spread=current_spread, avg_spread=avg_spread,
        daily_pnl=daily_pnl, account_info=account, upcoming_events=upcoming_events_raw,
    )
    if not can_trade:
        log.info("v3.circuit_breaker", symbol=symbol, reason=reason)
        return

    # -- Step 9: Execute via MT5 --------------------------------------------
    result = send_market_order(
        symbol=plan.symbol,
        direction=plan.direction,
        lot=plan.lot_size,
        sl=plan.stop_loss,
        tp=plan.take_profit_1,
        comment=f"v3|{plan.confidence_composite:.2f}|rr{plan.rr_ratio_tp1:.1f}",
        magic=settings.MAGIC_NUMBER if hasattr(settings, "MAGIC_NUMBER") else 12345,
    )

    # -- Record trade + v3 metadata -----------------------------------------
    if result.success:
        log.info(
            "v3.order_filled", symbol=plan.symbol, direction=plan.direction,
            ticket=result.ticket, price=result.price_filled, lot=plan.lot_size,
            rr=plan.rr_ratio_tp1, composite=plan.confidence_composite,
        )

        trade_data = {
            "symbol": plan.symbol,
            "direction": plan.direction,
            "entry_price": result.price_filled or plan.entry_price,
            "stop_loss": plan.stop_loss,
            "take_profit": plan.take_profit_1,
            "take_profit_2": plan.take_profit_2,
            "lot_size": plan.lot_size,
            "entry_time": result.time or datetime.now(timezone.utc),
            "status": "open",
            "confidence": int(plan.confidence_composite * 100),
            "risk_pct": settings.MAX_RISK_PER_TRADE_PCT,
            "rr_planned": plan.rr_ratio_tp1,
            "magic_number": settings.MAGIC_NUMBER if hasattr(settings, "MAGIC_NUMBER") else 12345,
            "session": getattr(mtf_state, "session", "unknown"),
            "atr_at_entry": getattr(mtf_state, "timeframes", {}).get("1M", {}).get("atr", 0),
            "sentiment_score": sentiment_score,
            "ai_reasoning": " | ".join(plan.reasoning_factors),
            # v3: ML metadata
            "xgb_score": xgb_result["score"],
            "lstm_confidence": lstm_result["confidence"],
            "lstm_direction": lstm_result["direction"],
            "lstm_regime": lstm_result["regime"],
            "mtf_confluence": getattr(mtf_state, "confluence_score", None),
            "veto_risk_modifier": veto.risk_modifier,
            "veto_reason": veto.reason if veto.risk_modifier != 1.0 else "",
            "reasoning_factors": plan.reasoning_factors,
            "agent_mode": "v3",
        }

        async with async_session() as db:
            await record_trade(db, trade_data)

        # Publish to WebSocket
        await _publish_v2_state(symbol, mtf_state=mtf_state,
                                xgb_result=xgb_result, lstm_result=lstm_result)
    else:
        log.error(
            "v3.order_failed", symbol=plan.symbol,
            error_code=result.error_code, error_message=result.error_message,
        )


# =========================================================================
# 3d. v3 SLOW LOOP -- background tasks (Claude veto, review, monitoring)
# =========================================================================

def _build_veto_context() -> dict:
    """Build context dict for the Claude veto scanner from current state."""
    account = get_account_info() or {}
    balance = account.get("balance", 0)
    equity = account.get("equity", 0)
    peak = getattr(circuit_breaker, "_peak_balance", balance) or balance
    drawdown_pct = ((peak - equity) / peak * 100) if peak > 0 else 0

    session = "unknown"
    if _mtf_analyzer is not None:
        # Try to get session from last MTF state
        for sym in settings.symbols_list:
            state = getattr(_mtf_analyzer, "_last_states", {}).get(sym)
            if state and hasattr(state, "session"):
                session = state.session
                break

    headlines: list[dict] = []
    events: list = []
    # Headlines and events are fetched synchronously from the context builder
    # since veto_scanner_loop calls this as a plain function.
    # The actual async fetching happens in the scanner loop wrapper.

    ml_health = _model_monitor.get_health() if _model_monitor else {
        "xgb_accuracy": 50.0, "lstm_accuracy": 50.0,
    }

    return build_veto_context(
        symbols=settings.symbols_list,
        account_info={
            **account,
            "daily_pnl_pct": account.get("daily_pnl_pct", 0),
            "drawdown_pct": round(drawdown_pct, 2),
            "consecutive_losses": account.get("consecutive_losses", 0),
        },
        session=session,
        headlines=headlines,
        events=events,
        ml_health=ml_health,
    )


async def _v3_veto_scanner_loop() -> None:
    """Background async task: run Claude veto scan every 30 seconds."""
    log.info("v3.veto_scanner_started")
    await veto_scanner_loop(_build_veto_context, interval=30)


async def _v3_model_monitor_daily() -> None:
    """Run daily model health check at 00:00 UTC."""
    if _model_monitor is None:
        log.warning("v3.model_monitor_not_initialized")
        return
    try:
        result = await _model_monitor.daily_check()
        log.info("v3.model_monitor_daily_complete",
                 action=result.get("action_taken"),
                 issues=result.get("issues", []))
    except Exception:
        log.exception("v3.model_monitor_daily_error")


async def _v3_trade_review(trade_data: dict) -> None:
    """Async task: review a closed trade via Claude (non-blocking)."""
    try:
        review = await claude_review_trade(trade_data)
        log.info("v3.trade_reviewed",
                 symbol=trade_data.get("symbol"),
                 quality=review.get("ml_signal_quality"),
                 lesson=review.get("lesson", "")[:100])
        # Update model monitor with actual outcome
        if _model_monitor and trade_data.get("pnl") is not None:
            actual = 1 if trade_data["pnl"] > 0 else 0
            entry_time = trade_data.get("entry_time", datetime.now(timezone.utc))
            _model_monitor.update_actual("xgb", entry_time, actual)
            _model_monitor.update_actual("lstm", entry_time, actual)
    except Exception:
        log.exception("v3.trade_review_error", symbol=trade_data.get("symbol"))


async def _v3_weekly_strategy_review() -> None:
    """Weekly Claude strategy review of all trades and ML decisions."""
    try:
        async with async_session() as db:
            from datetime import timedelta
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            # Fetch recent trades (stub -- uses existing journal query)
            trades: list[dict] = []
            for symbol in settings.symbols_list:
                symbol_trades = await get_recent_trades_for_symbol(db, symbol, limit=50)
                trades.extend(symbol_trades)

        if not trades:
            log.info("v3.weekly_review_skipped", reason="no trades this week")
            return

        # Use claude_reviewer for v3-style review
        reviews: list[dict] = []
        for t in trades[:20]:  # cap to avoid excessive API calls
            review = await claude_review_trade(t)
            reviews.append(review)

        summary = await claude_weekly_summary(trades, reviews)
        log.info("v3.weekly_review_complete",
                 grade=summary.get("overall_grade", "N/A"),
                 working=summary.get("whats_working", "")[:100])
    except Exception:
        log.exception("v3.weekly_review_error")


def _schedule_trade_review(trade_data: dict) -> None:
    """Fire-and-forget async task to review a closed trade."""
    global _trade_review_tasks
    task = asyncio.create_task(_v3_trade_review(trade_data))
    _trade_review_tasks.append(task)
    # Clean up completed tasks
    _trade_review_tasks = [t for t in _trade_review_tasks if not t.done()]


def _init_v3_components() -> None:
    """Initialize v3 two-loop architecture components."""
    global _rule_engine, _model_monitor

    _rule_engine = RuleEngine()
    _model_monitor = model_monitor_singleton

    # Initialize veto register with Redis client for persistence
    if _redis is not None:
        veto_register._redis = _redis

    log.info("v3.components_initialized",
             rule_engine=True,
             model_monitor=True,
             veto_register_redis=_redis is not None)


async def _start_v3_background_tasks() -> None:
    """Start v3 slow-loop background tasks as async tasks."""
    global _veto_scanner_task

    # Start veto scanner as persistent background task
    _veto_scanner_task = asyncio.create_task(_v3_veto_scanner_loop())
    log.info("v3.background_tasks_started", veto_scanner=True)


async def _xgb_weekly_retrain() -> None:
    """Weekly XGBoost retrain for all symbols."""
    for symbol, xgb_filter in _xgb_filters.items():
        try:
            log.info("xgb_retrain.starting", symbol=symbol)
            # The retrain method handles data loading internally;
            # we provide a stub call here -- actual feature/label assembly
            # will be implemented by the feature pipeline.
            # For now just log that retrain was triggered.
            log.info("xgb_retrain.scheduled", symbol=symbol,
                     hint="Feature pipeline must supply data to xgb_filter.retrain()")
        except Exception:
            log.exception("xgb_retrain.error", symbol=symbol)


async def _lstm_monthly_retrain() -> None:
    """Monthly LSTM retrain for all symbols."""
    for symbol, lstm_model in _lstm_models.items():
        try:
            log.info("lstm_retrain.starting", symbol=symbol)
            # The retrain method handles data loading internally;
            # we provide a stub call here -- actual dataset assembly
            # will be implemented by the feature pipeline.
            log.info("lstm_retrain.scheduled", symbol=symbol,
                     hint="Feature pipeline must supply dataset to lstm_model.retrain()")
        except Exception:
            log.exception("lstm_retrain.error", symbol=symbol)


def _init_v2_models() -> None:
    """Initialize per-symbol XGBoost filters, LSTM models, and MTF analyzer."""
    global _xgb_filters, _lstm_models, _mtf_analyzer

    from pathlib import Path

    models_dir = Path("models")

    for symbol in settings.symbols_list:
        # XGBoost
        xgb_path = models_dir / f"xgb_{symbol}.model"
        _xgb_filters[symbol] = XGBoostFilter(
            symbol=symbol,
            model_path=str(xgb_path) if xgb_path.exists() else None,
        )

        # LSTM
        lstm_path = models_dir / f"lstm_{symbol}.pt"
        _lstm_models[symbol] = LSTMConfidence(
            symbol=symbol,
            model_path=str(lstm_path) if lstm_path.exists() else None,
        )

    _mtf_analyzer = MTFAnalyzer()

    log.info(
        "v2_models.initialized",
        symbols=settings.symbols_list,
        xgb_count=len(_xgb_filters),
        lstm_count=len(_lstm_models),
    )


# =========================================================================
# 4. SHUTDOWN
# =========================================================================

async def _graceful_shutdown(close_positions: bool = False) -> None:
    """Clean up all resources."""
    global _redis, _scheduler, _veto_scanner_task

    log.info("shutdown.starting", close_positions=close_positions)

    # Cancel v3 background tasks
    if _veto_scanner_task is not None and not _veto_scanner_task.done():
        _veto_scanner_task.cancel()
        try:
            await _veto_scanner_task
        except asyncio.CancelledError:
            pass
        log.info("shutdown.veto_scanner_stopped")

    # Cancel any pending trade review tasks
    for task in _trade_review_tasks:
        if not task.done():
            task.cancel()
    if _trade_review_tasks:
        log.info("shutdown.trade_reviews_cancelled", count=len(_trade_review_tasks))

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
    if sys.platform != "win32":
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

    # -- 3b. v2 ML models and MTF analyzer ---------------------------------
    try:
        _init_v2_models()
    except Exception:
        log.warning("startup.v2_models_init_failed -- running in v1 mode")

    # -- 3c. v3 two-loop architecture components ---------------------------
    if settings.AGENT_MODE == "v3":
        try:
            _init_v3_components()
        except Exception:
            log.exception("startup.v3_init_failed -- falling back to v2 mode")

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

    # v2: Weekly XGBoost retrain (Sundays at 00:00 UTC)
    _scheduler.add_job(
        _xgb_weekly_retrain,
        "cron",
        day_of_week="sun",
        hour=0,
        minute=0,
        id="xgb_weekly_retrain",
        max_instances=1,
        misfire_grace_time=3600,
    )

    # v2: Monthly LSTM retrain (1st of each month at 01:00 UTC)
    _scheduler.add_job(
        _lstm_monthly_retrain,
        "cron",
        day=1,
        hour=1,
        minute=0,
        id="lstm_monthly_retrain",
        max_instances=1,
        misfire_grace_time=3600,
    )

    # v3: Model monitor daily check (00:00 UTC)
    if settings.AGENT_MODE == "v3" and _model_monitor is not None:
        _scheduler.add_job(
            _v3_model_monitor_daily,
            "cron",
            hour=0,
            minute=0,
            id="v3_model_monitor_daily",
            max_instances=1,
            misfire_grace_time=3600,
        )

    # v3: Weekly Claude strategy review (Sundays at 02:00 UTC)
    if settings.AGENT_MODE == "v3":
        _scheduler.add_job(
            _v3_weekly_strategy_review,
            "cron",
            day_of_week="sun",
            hour=2,
            minute=0,
            id="v3_weekly_strategy_review",
            max_instances=1,
            misfire_grace_time=3600,
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

    # -- 6b. v3: Start async background tasks (veto scanner) ---------------
    if settings.AGENT_MODE == "v3" and _rule_engine is not None:
        try:
            await _start_v3_background_tasks()
        except Exception:
            log.exception("startup.v3_background_tasks_failed")

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
