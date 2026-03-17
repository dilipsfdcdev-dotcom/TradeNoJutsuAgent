"""
XAUUSD Autonomous Self-Learning Trading Agent v2.0
Main orchestrator with two independent async loops.

COROUTINE A: signal_loop() -- fires every M3 candle (3 min)
COROUTINE B: ratchet_loop() -- fires every 15 seconds
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

from xauusd_agent.infra.logger import get_logger
from xauusd_agent.infra.database import (
    init_db,
    get_bot_state,
    set_bot_state,
    insert_signal,
    insert_trade,
    insert_ratchet_event,
    insert_regime_log,
    get_pending_commands,
    mark_command_processed,
    get_open_trades,
    get_recent_trades,
    get_daily_pnl,
    update_trade,
)
from xauusd_agent.infra.watchdog import WatchdogManager
from xauusd_agent.infra.scheduler import AsyncScheduler

from xauusd_agent.data.fetcher_mt5 import MT5DataFetcher
from xauusd_agent.data.fetcher_news import fetch_gold_headlines
from xauusd_agent.data.fetcher_macro import fetch_macro_data
from xauusd_agent.data.economic_calendar import EconomicCalendar

from xauusd_agent.features.multi_timeframe import build_multi_tf_features

from xauusd_agent.brain.htf_analyzer import HTFBiasEngine
from xauusd_agent.brain.signal_engine import M3M5SignalEngine
from xauusd_agent.brain.llm_reasoner import check as llm_check
from xauusd_agent.brain.news_filter import NewsFilter
from xauusd_agent.brain.signal_combiner import combine

from xauusd_agent.execution.mt5_connector import MT5Connector
from xauusd_agent.execution.mt5_executor import MT5Executor
from xauusd_agent.execution.lot_calculator import calculate_lot, compute_sl_tp
from xauusd_agent.execution.sl_manager import RatchetSLManager
from xauusd_agent.execution.position_tracker import PositionTracker

from xauusd_agent.models.ppo_agent import PPOAgent
from xauusd_agent.models.lgbm_signal import LGBMSignal

from xauusd_agent.telegram_bot.bot import TelegramBot

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOP_INTERVAL_SIGNAL = 180  # 3 minutes
LOOP_INTERVAL_RATCHET = 15  # 15 seconds
_SYMBOL = "XAUUSD"
_PIP_SIZE = 0.1


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------


def load_settings() -> dict:
    """Load settings.yaml from config/."""
    settings_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(settings_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# TradingAgent
# ---------------------------------------------------------------------------


class TradingAgent:
    """Top-level orchestrator for the XAUUSD autonomous trading agent."""

    def __init__(self) -> None:
        load_dotenv(Path(__file__).parent / "config" / ".env")
        self.settings = load_settings()
        self.running = True

        # Components (initialised in setup)
        self.db_pool: Any = None
        self.connector: MT5Connector | None = None
        self.executor: MT5Executor | None = None
        self.data_fetcher: MT5DataFetcher | None = None
        self.calendar: EconomicCalendar | None = None
        self.htf_engine: HTFBiasEngine | None = None
        self.signal_engine: M3M5SignalEngine | None = None
        self.news_filter: NewsFilter | None = None
        self.ppo_agent: PPOAgent | None = None
        self.lgbm_signal: LGBMSignal | None = None
        self.ratchet_manager: RatchetSLManager | None = None
        self.position_tracker: PositionTracker | None = None
        self.telegram_bot: TelegramBot | None = None
        self.watchdog: WatchdogManager | None = None
        self.scheduler: AsyncScheduler | None = None

        # State
        self.current_regime: str = "NORMAL"
        self.last_regime_update: float = 0.0
        self.completed_trades_since_learning: int = 0

    # ================================================================== #
    #  SETUP                                                              #
    # ================================================================== #

    async def setup(self) -> None:
        """Initialise all components.  Each step is guarded so a single
        subsystem failure does not prevent the rest from starting."""

        # 1. Database
        try:
            self.db_pool = await init_db()
            logger.info("Database pool initialised")
        except Exception:
            logger.critical("Database init failed -- cannot continue", exc_info=True)
            raise

        # 2. MT5 connector
        try:
            self.connector = MT5Connector()
            connected = self.connector.connect()
            if not connected:
                logger.critical("MT5 connection failed -- cannot continue")
                raise RuntimeError("MT5 connection failed")
            logger.info("MT5 connector ready")
        except Exception:
            logger.critical("MT5 init failed", exc_info=True)
            raise

        # 3. MT5 executor
        magic = self.settings.get("trading", {}).get("magic_number", 20250317)
        self.executor = MT5Executor(self.connector, magic_number=magic)
        logger.info("MT5 executor ready (magic=%d)", magic)

        # 4. Data fetcher
        self.data_fetcher = MT5DataFetcher()
        logger.info("MT5 data fetcher ready")

        # 5. Economic calendar
        self.calendar = EconomicCalendar()
        try:
            self.calendar.load_calendar()
            logger.info("Economic calendar loaded")
        except Exception:
            logger.warning("Economic calendar initial load failed", exc_info=True)

        # 6. Brain components
        self.htf_engine = HTFBiasEngine(self.settings.get("htf_bias", {}))
        logger.info("HTF bias engine ready")

        self.signal_engine = M3M5SignalEngine(self.settings.get("signals", {}))
        logger.info("M3/M5 signal engine ready")

        self.news_filter = NewsFilter(self.settings.get("news", {}))
        logger.info("News filter ready")

        # 7. ML models (graceful degradation if missing)
        models_dir = Path(__file__).parent / "models" / "checkpoints"

        self.ppo_agent = PPOAgent(
            model_path=str(models_dir / "ppo_xauusd.zip")
        )
        try:
            self.ppo_agent.load()
        except Exception:
            logger.warning("PPO model load failed -- will run without PPO", exc_info=True)

        self.lgbm_signal = LGBMSignal(
            model_path=str(models_dir / "lgbm_signal.txt"),
            onnx_path=str(models_dir / "lgbm_signal.onnx"),
        )
        try:
            self.lgbm_signal.load()
        except Exception:
            logger.warning("LightGBM model load failed -- will run without LGBM", exc_info=True)

        # 8. Ratchet SL manager
        self.ratchet_manager = RatchetSLManager(self.settings.get("ratchet", {}))
        logger.info("Ratchet SL manager ready")

        # 9. Position tracker
        self.position_tracker = PositionTracker(self.db_pool, self.executor)
        logger.info("Position tracker ready")

        # 10. Telegram bot
        tg_token = os.getenv("TELEGRAM_TOKEN", "")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            try:
                self.telegram_bot = TelegramBot(
                    token=tg_token,
                    chat_id=tg_chat,
                    db_pool=self.db_pool,
                )
                await self.telegram_bot.start()
                logger.info("Telegram bot started")
            except Exception:
                logger.warning("Telegram bot start failed", exc_info=True)
                self.telegram_bot = None
        else:
            logger.warning("Telegram credentials not set -- bot disabled")

        # 11. Watchdog
        self.watchdog = WatchdogManager()
        self.watchdog.register(
            "signal_loop",
            self.signal_loop,
            timeout_s=LOOP_INTERVAL_SIGNAL * 3,
        )
        self.watchdog.register(
            "ratchet_loop",
            self.ratchet_loop,
            timeout_s=LOOP_INTERVAL_RATCHET * 5,
        )
        logger.info("Watchdog manager configured")

        # 12. Scheduler (daily summary, weekly LGBM retrain, monthly PPO)
        self.scheduler = AsyncScheduler()
        self.scheduler.add_daily(
            "daily_summary",
            self._daily_summary_task,
            target_time=dt_time(22, 0),
        )
        logger.info("Scheduler configured")

        # 13. Persist initial bot state
        await set_bot_state(self.db_pool, "status", "RUNNING")
        await set_bot_state(
            self.db_pool,
            "risk_pct",
            str(self.settings.get("risk", {}).get("risk_pct", 1.0)),
        )
        await set_bot_state(
            self.db_pool,
            "entry_threshold",
            str(self.settings.get("signals", {}).get("entry_threshold", 62)),
        )
        logger.info("Bot state set to RUNNING")

    # ================================================================== #
    #  SIGNAL LOOP  (every 3 min)                                        #
    # ================================================================== #

    async def signal_loop(self) -> None:
        """Main signal generation loop -- fires every M3 candle."""

        while self.running:
            try:
                # 1. Check bot state ----------------------------------------
                state = await get_bot_state(self.db_pool, "status")
                if state in ("HALTED", "PAUSED"):
                    logger.info("Bot is %s, signal loop sleeping", state)
                    await asyncio.sleep(LOOP_INTERVAL_SIGNAL)
                    continue

                # 2. Process pending Telegram commands ----------------------
                await self._process_commands()

                # 3. Session restrictions -----------------------------------
                if self._is_session_restricted():
                    logger.info("Session restricted, skipping signal generation")
                    await asyncio.sleep(LOOP_INTERVAL_SIGNAL)
                    continue

                # 4. Fetch multi-TF data ------------------------------------
                all_tf = self.data_fetcher.fetch_all_timeframes()
                if not all_tf or "M3" not in all_tf or all_tf["M3"].empty:
                    logger.error("Failed to fetch M3 data")
                    await asyncio.sleep(LOOP_INTERVAL_SIGNAL)
                    continue

                # 5. Update regime periodically -----------------------------
                regime_interval = (
                    self.settings.get("regime", {}).get("update_interval_min", 30) * 60
                )
                if time.time() - self.last_regime_update > regime_interval:
                    await self._update_regime(all_tf)

                # 6. HTF bias -----------------------------------------------
                htf_bias = self.htf_engine.get_bias(all_tf)

                # Persist HTF state for Telegram /status
                await set_bot_state(
                    self.db_pool, "htf_direction", htf_bias.get("direction", "NEUTRAL")
                )
                await set_bot_state(
                    self.db_pool, "htf_score", str(htf_bias.get("score", 0))
                )

                # 7. Signal engine ------------------------------------------
                spread = self.data_fetcher.get_spread_pips()
                signal_result = self.signal_engine.get_signal(all_tf, htf_bias, spread)

                # 8. LLM reasoner (only when score above threshold) ---------
                llm_result = _default_llm_result()
                max_score = max(
                    signal_result.get("buy_score", 0),
                    signal_result.get("sell_score", 0),
                )
                llm_threshold = self.settings.get("signals", {}).get("llm_check_above", 58)
                if max_score > llm_threshold:
                    llm_result = await self._run_llm_check(signal_result)

                # 9-10. ML inference ----------------------------------------
                ppo_action, ppo_confidence = self._run_ppo(all_tf)
                lgbm_prob = self._run_lgbm(all_tf)

                # 11. Combine all signals -----------------------------------
                news_blackout = self.calendar.is_blackout(
                    self.settings.get("news", {}).get("blackout_before_min", 30),
                    self.settings.get("news", {}).get("blackout_after_min", 30),
                )
                daily_dd = await self._get_daily_drawdown()
                open_positions = self.executor.get_open_positions()

                action, confidence, reason = combine(
                    signal_engine_result=signal_result,
                    ppo_action=ppo_action,
                    ppo_confidence=ppo_confidence,
                    lgbm_prob=lgbm_prob,
                    llm_result=llm_result,
                    news_blackout=news_blackout,
                    daily_drawdown_pct=daily_dd,
                    open_positions=len(open_positions),
                    max_positions=self.settings.get("risk", {}).get("max_concurrent", 3),
                    regime=self.current_regime,
                    settings=self.settings.get("signals", {}),
                )

                # 12. Execute if actionable ---------------------------------
                was_executed = False
                if action in ("BUY", "SELL"):
                    was_executed = await self._execute_trade(
                        action, signal_result, htf_bias, llm_result, spread, all_tf
                    )

                # 13. Log signal to DB --------------------------------------
                await self._log_signal(
                    signal_result,
                    ppo_action,
                    ppo_confidence,
                    lgbm_prob,
                    llm_result,
                    htf_bias,
                    action,
                    reason,
                    spread,
                    news_blackout,
                    was_executed,
                )

                # 14. Sync positions and check for closed trades ------------
                newly_closed = await self.position_tracker.sync_positions()
                if newly_closed:
                    self.completed_trades_since_learning += len(newly_closed)
                    for trade in newly_closed:
                        await self._on_trade_closed(trade)

                # 15. Learning cycle check ----------------------------------
                cycle_threshold = self.settings.get("learning", {}).get(
                    "cycle_every_n_trades", 10
                )
                if self.completed_trades_since_learning >= cycle_threshold:
                    await self._run_learning_cycle()

                # 16. Update live state in DB for Telegram queries ----------
                await self._update_live_state()

                # 17. Watchdog heartbeat ------------------------------------
                if self.watchdog:
                    self.watchdog.heartbeat("signal_loop")

            except Exception:
                logger.error("Signal loop error", exc_info=True)
                if self.telegram_bot:
                    try:
                        await self.telegram_bot.send_alert(
                            "<b>Signal loop error</b> -- check logs"
                        )
                    except Exception:
                        pass

            await asyncio.sleep(LOOP_INTERVAL_SIGNAL)

    # ================================================================== #
    #  RATCHET LOOP  (every 15 s)                                        #
    # ================================================================== #

    async def ratchet_loop(self) -> None:
        """Ratchet stop-loss management loop -- completely independent of
        the signal loop.  Runs even when the bot is PAUSED (only stops
        when HALTED)."""

        while self.running:
            try:
                state = await get_bot_state(self.db_pool, "status")
                if state == "HALTED":
                    await asyncio.sleep(LOOP_INTERVAL_RATCHET)
                    continue

                # Fetch enriched positions (includes initial_risk_usd, pip_value, etc.)
                enriched = await self.position_tracker.get_enriched_positions()
                if not enriched:
                    if self.watchdog:
                        self.watchdog.heartbeat("ratchet_loop")
                    await asyncio.sleep(LOOP_INTERVAL_RATCHET)
                    continue

                for pos in enriched:
                    current_price = pos.get("current_price")
                    if current_price is None or current_price <= 0:
                        continue

                    new_sl = self.ratchet_manager.compute_new_sl(pos, current_price)
                    if new_sl is None:
                        continue

                    success = self.executor.modify_sl(pos["ticket"], new_sl)
                    if not success:
                        logger.error(
                            "Failed to ratchet SL for ticket %d to %.5f",
                            pos["ticket"],
                            new_sl,
                        )
                        continue

                    # Compute R-multiple for logging
                    profit_r = self._compute_profit_r(pos, current_price)
                    ratchet_label = self._ratchet_label(profit_r)

                    # Persist event to DB
                    pip_value = pos.get("pip_value", 1.0)
                    pip_size = pos.get("pip_size", _PIP_SIZE)
                    lot_size = pos.get("lot_size", 0.0)
                    direction = pos.get("direction", "BUY").upper()
                    open_price = pos.get("open_price", 0.0)

                    if direction == "BUY":
                        locked_pips = (new_sl - open_price) / pip_size
                    else:
                        locked_pips = (open_price - new_sl) / pip_size
                    profit_locked_usd = max(0.0, locked_pips * pip_value * lot_size)

                    await insert_ratchet_event(self.db_pool, {
                        "ticket": pos["ticket"],
                        "old_sl": pos.get("current_sl", 0.0),
                        "new_sl": new_sl,
                        "current_price": current_price,
                        "profit_R": round(profit_r, 4),
                        "ratchet_level": ratchet_label,
                        "profit_locked_usd": round(profit_locked_usd, 2),
                    })

                    # Update trade record
                    await update_trade(self.db_pool, pos["ticket"], {
                        "ratchet_triggered": True,
                        "ratchet_max_R": round(profit_r, 4),
                        "profit_locked_usd": round(profit_locked_usd, 2),
                    })

                    logger.info(
                        "Ratchet applied: ticket=%d SL %.5f -> %.5f (%s, %.2fR)",
                        pos["ticket"],
                        pos.get("current_sl", 0.0),
                        new_sl,
                        ratchet_label,
                        profit_r,
                    )

                    # Telegram alert
                    if self.telegram_bot:
                        try:
                            await self.telegram_bot.alert_ratchet(
                                pos, new_sl, profit_r, ratchet_label
                            )
                        except Exception:
                            logger.warning("Ratchet Telegram alert failed", exc_info=True)

                # Check for newly closed positions
                newly_closed = await self.position_tracker.sync_positions()
                for trade in newly_closed:
                    await self._on_trade_closed(trade)

                if self.watchdog:
                    self.watchdog.heartbeat("ratchet_loop")

            except Exception:
                logger.error("Ratchet loop error", exc_info=True)

            await asyncio.sleep(LOOP_INTERVAL_RATCHET)

    # ================================================================== #
    #  TRADE EXECUTION                                                    #
    # ================================================================== #

    async def _execute_trade(
        self,
        action: str,
        signal_result: dict,
        htf_bias: dict,
        llm_result: dict,
        spread: float,
        all_tf: dict,
    ) -> bool:
        """Execute a trade: compute SL/TP, lot size, send order.

        Returns ``True`` if the order was filled successfully.
        """
        direction = action.upper()
        is_counter_trend = signal_result.get("is_counter_trend", False)

        # Check counter-trend permission
        if is_counter_trend:
            ct_allowed = self.settings.get("risk", {}).get("counter_trend_allowed", True)
            if not ct_allowed:
                logger.info("Counter-trend trade blocked by settings")
                return False

        # Check spread limit
        max_spread = self.settings.get("risk", {}).get("max_spread_pips", 8.0)
        if spread > max_spread:
            logger.info("Spread %.2f exceeds max %.2f -- skipping", spread, max_spread)
            return False

        # Compute SL/TP from M3/M5 ATR
        m3 = all_tf.get("M3")
        m5 = all_tf.get("M5")
        if m3 is None or m5 is None or m3.empty or m5.empty:
            logger.error("M3 or M5 data missing -- cannot compute SL/TP")
            return False

        sl_tp = compute_sl_tp(
            m3=m3,
            m5=m5,
            direction=direction,
            is_counter_trend=is_counter_trend,
            regime=self.current_regime,
            settings=self.settings.get("sl_tp"),
        )
        if sl_tp is None:
            logger.info("SL/TP computation returned None (too volatile) -- skipping")
            return False

        sl_pips, tp_pips, atr_value = sl_tp

        # Get live account balance
        account = self.data_fetcher.get_live_account()
        if not account:
            logger.error("Failed to get account info -- skipping trade")
            return False
        balance = account.get("balance", 0.0)
        if balance <= 0:
            logger.error("Account balance is zero or negative")
            return False

        # Read risk_pct from DB (may have been changed via /risk command)
        risk_pct_str = await get_bot_state(self.db_pool, "risk_pct")
        risk_pct = float(risk_pct_str) if risk_pct_str else self.settings.get(
            "risk", {}
        ).get("risk_pct", 1.0)

        # Calculate lot size
        lot, risk_usd = calculate_lot(
            live_balance=balance,
            risk_pct=risk_pct,
            sl_pips=sl_pips,
            is_counter_trend=is_counter_trend,
            current_regime=self.current_regime,
            symbol=_SYMBOL,
            settings=self.settings.get("sl_tp"),
        )
        if lot <= 0:
            logger.error("Lot calculation returned 0 -- skipping trade")
            return False

        # Compute price-level SL and TP
        price_info = self.data_fetcher.get_current_price()
        if not price_info:
            logger.error("Cannot get current price -- skipping trade")
            return False

        if direction == "BUY":
            entry_price = price_info["ask"]
            sl_price = entry_price - sl_pips * _PIP_SIZE
            tp_price = entry_price + tp_pips * _PIP_SIZE
        else:
            entry_price = price_info["bid"]
            sl_price = entry_price + sl_pips * _PIP_SIZE
            tp_price = entry_price - tp_pips * _PIP_SIZE

        # Round to appropriate precision
        sl_price = round(sl_price, 2)
        tp_price = round(tp_price, 2)

        # Send order
        tp_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        comment = f"TNJ|{direction[0]}|{self.current_regime[:3]}"
        result = self.executor.open_trade(
            symbol=_SYMBOL,
            direction=direction,
            lot=lot,
            sl=sl_price,
            tp=tp_price,
            comment=comment,
        )

        if not result.get("success"):
            logger.error(
                "Order rejected: retcode=%s comment=%s",
                result.get("retcode"),
                result.get("comment"),
            )
            return False

        ticket = result["ticket"]
        fill_price = result["price"]
        logger.info(
            "Trade opened: %s %s %.2f lots @ %.2f ticket=%d",
            direction,
            _SYMBOL,
            lot,
            fill_price,
            ticket,
        )

        # Build feature snapshot for the DB record
        feature_snapshot = self._build_feature_snapshot(
            signal_result, htf_bias, llm_result, spread
        )

        # Log signal first to get signal_id
        signal_id = await insert_signal(self.db_pool, {
            "timestamp": datetime.now(timezone.utc),
            "symbol": _SYMBOL,
            "action": direction,
            "buy_score": signal_result.get("buy_score"),
            "sell_score": signal_result.get("sell_score"),
            "entry_threshold": signal_result.get("threshold"),
            "htf_score": htf_bias.get("score"),
            "htf_direction": htf_bias.get("direction"),
            "d1_score": htf_bias.get("d1"),
            "h4_score": htf_bias.get("h4"),
            "h1_score": htf_bias.get("h1"),
            "m15_at_support": htf_bias.get("m15_at_support", False),
            "regime": self.current_regime,
            "is_counter_trend": is_counter_trend,
            "was_executed": True,
            "spread_pips": spread,
        })

        # Persist trade to DB
        await insert_trade(self.db_pool, {
            "ticket": ticket,
            "signal_id": signal_id,
            "open_time": datetime.now(timezone.utc),
            "direction": direction,
            "lot_size": lot,
            "open_price": fill_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "initial_sl_pips": sl_pips,
            "initial_risk_usd": risk_usd,
            "status": "OPEN",
            "is_counter_trend": is_counter_trend,
            "htf_bias_score": htf_bias.get("score"),
            "regime": self.current_regime,
            "balance_at_entry": balance,
            "feature_snapshot": json.dumps(feature_snapshot),
        })

        # Telegram alert
        if self.telegram_bot:
            try:
                # Compute RSI values for alert
                from xauusd_agent.brain.signal_engine import _rsi as sig_rsi
                m3_rsi = float(sig_rsi(m3["close"], 7).iloc[-1])
                m5_rsi = float(sig_rsi(m5["close"], 14).iloc[-1])

                await self.telegram_bot.alert_trade_open({
                    "ticket": ticket,
                    "direction": direction,
                    "symbol": _SYMBOL,
                    "open_price": fill_price,
                    "lot_size": lot,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "risk_usd": risk_usd,
                    "balance_at_entry": balance,
                    "risk_pct": risk_pct,
                    "sl_pips": sl_pips,
                    "tp_pips": tp_pips,
                    "tp_ratio": tp_ratio,
                    "htf_direction": htf_bias.get("direction"),
                    "htf_score": htf_bias.get("score"),
                    "regime": self.current_regime,
                    "buy_score": signal_result.get("buy_score"),
                    "sell_score": signal_result.get("sell_score"),
                    "threshold": signal_result.get("threshold"),
                    "rsi_m3": m3_rsi,
                    "rsi_m5": m5_rsi,
                    "llm_action": llm_result.get("action"),
                })
            except Exception:
                logger.warning("Trade open Telegram alert failed", exc_info=True)

        return True

    # ================================================================== #
    #  REGIME DETECTION                                                   #
    # ================================================================== #

    async def _update_regime(self, all_tf: dict) -> None:
        """Detect the current market regime from ATR ratios and log it."""
        try:
            m5 = all_tf.get("M5")
            if m5 is None or m5.empty or len(m5) < 50:
                return

            # ATR-based regime detection
            high = m5["high"]
            low = m5["low"]
            close = m5["close"]
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            import pandas as pd
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr_fast = float(tr.ewm(span=7, adjust=False).mean().iloc[-1])
            atr_slow = float(tr.ewm(span=50, adjust=False).mean().iloc[-1])

            atr_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0

            high_vol_thresh = self.settings.get("regime", {}).get("high_vol_atr_ratio", 2.0)
            low_vol_thresh = self.settings.get("regime", {}).get("low_vol_atr_ratio", 0.5)

            old_regime = self.current_regime
            if atr_ratio >= high_vol_thresh:
                self.current_regime = "HIGH_VOLATILE"
            elif atr_ratio <= low_vol_thresh:
                self.current_regime = "LOW_VOLATILE"
            else:
                self.current_regime = "NORMAL"

            self.last_regime_update = time.time()

            # Persist regime state
            await set_bot_state(self.db_pool, "regime", self.current_regime)

            # Log to regime_log table
            htf_bias = self.htf_engine.get_bias(all_tf)
            await insert_regime_log(self.db_pool, {
                "regime": self.current_regime,
                "htf_score": htf_bias.get("score"),
                "d1_score": htf_bias.get("d1"),
                "h4_score": htf_bias.get("h4"),
                "h1_score": htf_bias.get("h1"),
                "atr_m5": atr_fast,
                "atr_ratio": round(atr_ratio, 4),
            })

            logger.info(
                "Regime updated: %s (ATR ratio=%.2f)", self.current_regime, atr_ratio
            )

            # Alert on regime change
            if old_regime != self.current_regime and self.telegram_bot:
                lot_adj = "100%"
                if self.current_regime == "HIGH_VOLATILE":
                    mult = self.settings.get("regime", {}).get("high_vol_lot_mult", 0.75)
                    lot_adj = f"{int(mult * 100)}%"
                elif self.current_regime == "LOW_VOLATILE":
                    mult = self.settings.get("regime", {}).get("low_vol_lot_mult", 1.10)
                    lot_adj = f"{int(mult * 100)}%"

                try:
                    await self.telegram_bot.alert_regime_change(
                        old_regime,
                        self.current_regime,
                        {"atr_ratio": atr_ratio, "lot_adjustment": lot_adj},
                    )
                except Exception:
                    logger.warning("Regime change Telegram alert failed", exc_info=True)

        except Exception:
            logger.error("Regime update failed", exc_info=True)

    # ================================================================== #
    #  LLM CHECK                                                          #
    # ================================================================== #

    async def _run_llm_check(self, signal_result: dict) -> dict:
        """Run the LLM reasoner for trade confirmation."""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                return _default_llm_result("no API key")

            headlines = fetch_gold_headlines(
                os.getenv("NEWSAPI_KEY", ""), max_results=5
            )
            events = self.calendar.get_upcoming_events(hours_ahead=2)

            result = await llm_check(
                signal=signal_result.get("action", "HOLD"),
                headlines=[h.get("title", "") for h in headlines],
                upcoming_events=events,
                api_key=api_key,
                model=self.settings.get("news", {}).get(
                    "llm_model", "claude-haiku-3-5-20251001"
                ),
                timeout=self.settings.get("news", {}).get("llm_timeout_s", 5),
                cache_ttl=self.settings.get("news", {}).get("llm_cache_s", 90),
            )
            return result

        except Exception as e:
            logger.warning("LLM check failed: %s", e)
            return _default_llm_result(f"error: {e}")

    # ================================================================== #
    #  ML MODEL INFERENCE                                                 #
    # ================================================================== #

    def _run_ppo(self, all_tf: dict) -> tuple[int, float]:
        """Run PPO agent prediction.  Returns (action, confidence)."""
        if self.ppo_agent is None or self.ppo_agent.model is None:
            return 0, 0.0
        try:
            features = build_multi_tf_features(all_tf)
            obs = features.iloc[-1].values.astype("float32")
            obs = np.nan_to_num(obs, nan=0.0)
            return self.ppo_agent.predict(obs)
        except Exception:
            logger.warning("PPO prediction failed", exc_info=True)
            return 0, 0.0

    def _run_lgbm(self, all_tf: dict) -> float:
        """Run LightGBM probability prediction."""
        if self.lgbm_signal is None:
            return 0.5
        if self.lgbm_signal.model is None and self.lgbm_signal.onnx_session is None:
            return 0.5
        try:
            features = build_multi_tf_features(all_tf)
            feat_vec = features.iloc[-1].values.astype("float32")
            feat_vec = np.nan_to_num(feat_vec, nan=0.0)
            return self.lgbm_signal.predict_proba(feat_vec)
        except Exception:
            logger.warning("LightGBM prediction failed", exc_info=True)
            return 0.5

    # ================================================================== #
    #  COMMAND PROCESSING                                                 #
    # ================================================================== #

    async def _process_commands(self) -> None:
        """Process pending Telegram commands from the bot_commands table."""
        try:
            commands = await get_pending_commands(self.db_pool)
        except Exception:
            logger.error("Failed to fetch pending commands", exc_info=True)
            return

        for cmd in commands:
            cmd_name = cmd.get("command", "").lower()
            cmd_value = cmd.get("value", "")
            cmd_id = cmd.get("id")

            try:
                if cmd_name == "stop":
                    await set_bot_state(self.db_pool, "status", "PAUSED")
                    logger.info("Bot PAUSED via command")

                elif cmd_name == "resume":
                    await set_bot_state(self.db_pool, "status", "RUNNING")
                    logger.info("Bot RESUMED via command")

                elif cmd_name == "halt":
                    await set_bot_state(self.db_pool, "status", "HALTED")
                    logger.warning("EMERGENCY HALT via command -- closing all positions")
                    close_results = self.executor.close_all_positions(_SYMBOL)
                    for cr in close_results:
                        if cr.get("success"):
                            logger.info(
                                "Emergency close: ticket=%d profit=%.2f",
                                cr.get("ticket", 0),
                                cr.get("profit", 0.0),
                            )

                elif cmd_name == "risk":
                    try:
                        new_risk = float(cmd_value)
                        if 0.1 <= new_risk <= 2.0:
                            await set_bot_state(
                                self.db_pool, "risk_pct", str(new_risk)
                            )
                            logger.info("Risk updated to %.1f%% via command", new_risk)
                        else:
                            logger.warning(
                                "Invalid risk value: %.2f (must be 0.1-2.0)", new_risk
                            )
                    except (ValueError, TypeError):
                        logger.warning("Invalid risk command value: %s", cmd_value)

                else:
                    logger.warning("Unknown command: %s", cmd_name)

                await mark_command_processed(self.db_pool, cmd_id)

            except Exception:
                logger.error(
                    "Error processing command %s (id=%d)", cmd_name, cmd_id, exc_info=True
                )
                await mark_command_processed(self.db_pool, cmd_id)

    # ================================================================== #
    #  SIGNAL LOGGING                                                     #
    # ================================================================== #

    async def _log_signal(
        self,
        signal_result: dict,
        ppo_action: int,
        ppo_confidence: float,
        lgbm_prob: float,
        llm_result: dict,
        htf_bias: dict,
        action: str,
        reason: str,
        spread: float,
        news_blackout: bool,
        was_executed: bool,
    ) -> None:
        """Log a signal evaluation to the signals table."""
        try:
            # Extract RSI and MACD from signal components if available
            rsi_m3 = None
            rsi_m5 = None
            macd_m3 = None

            skip_reason = None if was_executed else reason[:100] if reason else None

            await insert_signal(self.db_pool, {
                "timestamp": datetime.now(timezone.utc),
                "symbol": _SYMBOL,
                "action": action,
                "buy_score": signal_result.get("buy_score"),
                "sell_score": signal_result.get("sell_score"),
                "entry_threshold": signal_result.get("threshold"),
                "ppo_action": ppo_action,
                "ppo_confidence": round(ppo_confidence, 4),
                "lgbm_probability": round(lgbm_prob, 4),
                "llm_action": llm_result.get("action"),
                "llm_reason": llm_result.get("reason", "")[:200],
                "htf_score": htf_bias.get("score"),
                "htf_direction": htf_bias.get("direction"),
                "d1_score": htf_bias.get("d1"),
                "h4_score": htf_bias.get("h4"),
                "h1_score": htf_bias.get("h1"),
                "m15_at_support": htf_bias.get("m15_at_support", False),
                "regime": self.current_regime,
                "is_counter_trend": signal_result.get("is_counter_trend", False),
                "news_blackout": news_blackout,
                "skip_reason": skip_reason,
                "was_executed": was_executed,
                "rsi_m3": rsi_m3,
                "rsi_m5": rsi_m5,
                "macd_m3": macd_m3,
                "spread_pips": spread,
            })
        except Exception:
            logger.error("Failed to log signal to DB", exc_info=True)

    # ================================================================== #
    #  DRAWDOWN CALCULATION                                               #
    # ================================================================== #

    async def _get_daily_drawdown(self) -> float:
        """Calculate today's drawdown as a positive percentage."""
        try:
            daily_pnl = await get_daily_pnl(self.db_pool)
            account = self.data_fetcher.get_live_account()
            balance = account.get("balance", 0.0) if account else 0.0

            if balance <= 0:
                return 0.0

            # Drawdown is expressed as a positive number
            if daily_pnl < 0:
                return abs(daily_pnl) / balance * 100.0
            return 0.0

        except Exception:
            logger.error("Failed to calculate daily drawdown", exc_info=True)
            return 0.0

    # ================================================================== #
    #  SESSION RESTRICTION CHECK                                          #
    # ================================================================== #

    def _is_session_restricted(self) -> bool:
        """Return ``True`` if trading is restricted:
        - Friday after 20:00 GMT
        - Sunday before 21:00 GMT
        """
        now_utc = datetime.now(timezone.utc)
        weekday = now_utc.weekday()  # 0=Monday, 6=Sunday
        hour = now_utc.hour
        minute = now_utc.minute

        session_cfg = self.settings.get("session", {})

        # Parse Friday cutoff (default "20:00")
        friday_after_str = session_cfg.get("no_new_friday_after", "20:00")
        friday_h, friday_m = (int(x) for x in friday_after_str.split(":"))

        # Parse Sunday cutoff (default "21:00")
        sunday_before_str = session_cfg.get("no_new_sunday_before", "21:00")
        sunday_h, sunday_m = (int(x) for x in sunday_before_str.split(":"))

        # Friday after cutoff
        if weekday == 4 and (hour > friday_h or (hour == friday_h and minute >= friday_m)):
            return True

        # Saturday -- market closed
        if weekday == 5:
            return True

        # Sunday before market open
        if weekday == 6 and (hour < sunday_h or (hour == sunday_h and minute < sunday_m)):
            return True

        return False

    # ================================================================== #
    #  LEARNING CYCLE                                                     #
    # ================================================================== #

    async def _run_learning_cycle(self) -> None:
        """Run the self-learning memory agent cycle to adjust parameters."""
        try:
            recent = await get_recent_trades(self.db_pool, n=50)
            if not recent:
                return

            # Compute performance stats
            closed = [t for t in recent if t.get("status") != "OPEN"]
            if len(closed) < 5:
                return

            wins = [t for t in closed if (t.get("profit_usd") or 0) > 0]
            losses = [t for t in closed if (t.get("profit_usd") or 0) <= 0]
            win_rate = len(wins) / len(closed) * 100 if closed else 0

            avg_winner_r = (
                sum(t.get("profit_R", 0) or 0 for t in wins) / len(wins)
                if wins
                else 0
            )
            avg_loser_r = (
                sum(abs(t.get("profit_R", 0) or 0) for t in losses) / len(losses)
                if losses
                else 0
            )

            changes: list[dict] = []
            max_changes = self.settings.get("learning", {}).get("max_param_changes", 3)

            # Rule-based parameter adjustments
            # 1. Entry threshold adjustment based on win rate
            current_threshold = self.settings.get("signals", {}).get("entry_threshold", 62)
            if win_rate < 40 and current_threshold < 78:
                new_threshold = min(current_threshold + 2, 78)
                changes.append({
                    "param_name": "entry_threshold",
                    "old_value": current_threshold,
                    "new_value": new_threshold,
                    "reason": f"win rate {win_rate:.0f}% < 40% -- raising threshold",
                    "trades_analysed": len(closed),
                    "win_rate": win_rate,
                })
            elif win_rate > 65 and current_threshold > 52:
                new_threshold = max(current_threshold - 2, 52)
                changes.append({
                    "param_name": "entry_threshold",
                    "old_value": current_threshold,
                    "new_value": new_threshold,
                    "reason": f"win rate {win_rate:.0f}% > 65% -- lowering threshold",
                    "trades_analysed": len(closed),
                    "win_rate": win_rate,
                })

            # 2. Risk % adjustment based on drawdown
            daily_dd = await self._get_daily_drawdown()
            current_risk = self.settings.get("risk", {}).get("risk_pct", 1.0)
            if daily_dd > 2.0 and current_risk > 0.5 and len(changes) < max_changes:
                new_risk = max(current_risk - 0.1, 0.5)
                changes.append({
                    "param_name": "risk_pct",
                    "old_value": current_risk,
                    "new_value": new_risk,
                    "reason": f"daily drawdown {daily_dd:.1f}% > 2% -- reducing risk",
                    "trades_analysed": len(closed),
                    "win_rate": win_rate,
                })

            # 3. Counter-trend toggle
            ct_trades = [t for t in closed if t.get("is_counter_trend")]
            if len(ct_trades) >= 3 and len(changes) < max_changes:
                ct_wins = [t for t in ct_trades if (t.get("profit_usd") or 0) > 0]
                ct_wr = len(ct_wins) / len(ct_trades) * 100
                if ct_wr < 30:
                    current_ct = self.settings.get("risk", {}).get(
                        "counter_trend_allowed", True
                    )
                    if current_ct:
                        changes.append({
                            "param_name": "counter_trend_allowed",
                            "old_value": 1.0,
                            "new_value": 0.0,
                            "reason": (
                                f"counter-trend WR {ct_wr:.0f}% < 30% "
                                f"-- disabling counter-trend"
                            ),
                            "trades_analysed": len(closed),
                            "win_rate": win_rate,
                        })

            # Apply changes
            if changes:
                await self._apply_learning_changes(changes)

                # Log changes to DB
                from xauusd_agent.infra.database import insert_learning_log

                for change in changes:
                    await insert_learning_log(self.db_pool, {
                        "trigger_type": "memory_agent",
                        "param_name": change["param_name"],
                        "old_value": change["old_value"],
                        "new_value": change["new_value"],
                        "reason": change["reason"],
                        "trades_analysed": change.get("trades_analysed"),
                        "win_rate": change.get("win_rate"),
                    })

                logger.info(
                    "Learning cycle: %d changes applied",
                    len(changes),
                    extra={"changes": [c["param_name"] for c in changes]},
                )

                # Telegram alert
                if self.telegram_bot:
                    try:
                        await self.telegram_bot.alert_learning(changes)
                    except Exception:
                        logger.warning("Learning Telegram alert failed", exc_info=True)

                # Reload settings and re-init affected components
                self.settings = load_settings()
                self.signal_engine = M3M5SignalEngine(self.settings.get("signals", {}))

            self.completed_trades_since_learning = 0

        except Exception:
            logger.error("Learning cycle failed", exc_info=True)

    async def _apply_learning_changes(self, changes: list[dict]) -> None:
        """Write parameter changes back to settings.yaml."""
        settings_path = Path(__file__).parent / "config" / "settings.yaml"

        try:
            with open(settings_path) as f:
                cfg = yaml.safe_load(f)

            for change in changes:
                param = change["param_name"]
                new_val = change["new_value"]

                # Map param name to settings path
                if param == "entry_threshold":
                    cfg.setdefault("signals", {})["entry_threshold"] = new_val
                elif param == "risk_pct":
                    cfg.setdefault("risk", {})["risk_pct"] = new_val
                    await set_bot_state(self.db_pool, "risk_pct", str(new_val))
                elif param == "counter_trend_allowed":
                    cfg.setdefault("risk", {})["counter_trend_allowed"] = bool(new_val)
                elif param == "sl_mult_trend":
                    cfg.setdefault("sl_tp", {})["sl_mult_trend"] = new_val
                elif param == "tp_ratio_trend":
                    cfg.setdefault("sl_tp", {})["tp_ratio_trend"] = new_val
                elif param == "blackout_before_min":
                    cfg.setdefault("news", {})["blackout_before_min"] = int(new_val)
                elif param == "blackout_after_min":
                    cfg.setdefault("news", {})["blackout_after_min"] = int(new_val)
                else:
                    logger.warning("Unknown learning param: %s", param)

            with open(settings_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            logger.info("Settings file updated with %d changes", len(changes))

        except Exception:
            logger.error("Failed to write learning changes to settings.yaml", exc_info=True)

    # ================================================================== #
    #  CLOSED TRADE HANDLER                                               #
    # ================================================================== #

    async def _on_trade_closed(self, trade: dict) -> None:
        """Handle a newly-closed trade: log, alert, update counters."""
        ticket = trade.get("ticket", 0)
        profit = trade.get("profit", 0.0)
        logger.info(
            "Trade closed: ticket=%d profit=%.2f", ticket, profit
        )

        # Update trade record with close info
        try:
            await update_trade(self.db_pool, ticket, {
                "close_price": trade.get("close_price", 0.0),
                "profit_usd": profit,
                "close_time": trade.get(
                    "close_time", datetime.now(timezone.utc)
                ),
                "status": "CLOSED",
            })
        except Exception:
            logger.error("Failed to update closed trade %d in DB", ticket, exc_info=True)

        # Telegram alert
        if self.telegram_bot:
            try:
                account = self.data_fetcher.get_live_account()
                balance = account.get("balance", 0.0) if account else 0.0

                await self.telegram_bot.alert_trade_close({
                    "ticket": ticket,
                    "direction": trade.get("direction", ""),
                    "symbol": _SYMBOL,
                    "profit_usd": profit,
                    "profit_R": trade.get("profit_R", 0),
                    "profit_pips": trade.get("profit_pips", 0),
                    "open_time": trade.get("open_time"),
                    "close_time": trade.get("close_time"),
                    "ratchet_triggered": trade.get("ratchet_triggered", False),
                    "profit_locked_usd": trade.get("profit_locked_usd", 0),
                    "balance": balance,
                })
            except Exception:
                logger.warning(
                    "Trade close Telegram alert failed for %d", ticket, exc_info=True
                )

    # ================================================================== #
    #  LIVE STATE UPDATER                                                 #
    # ================================================================== #

    async def _update_live_state(self) -> None:
        """Push current account data to bot_state for Telegram queries."""
        try:
            account = self.data_fetcher.get_live_account()
            if account:
                await set_bot_state(
                    self.db_pool, "balance", str(round(account.get("balance", 0), 2))
                )
                await set_bot_state(
                    self.db_pool, "equity", str(round(account.get("equity", 0), 2))
                )
        except Exception:
            logger.debug("Failed to update live state", exc_info=True)

    # ================================================================== #
    #  DAILY SUMMARY TASK                                                 #
    # ================================================================== #

    async def _daily_summary_task(self) -> None:
        """Generate and send the daily trading summary."""
        try:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            recent = await get_recent_trades(self.db_pool, n=100)
            today_trades = [
                t for t in recent
                if t.get("open_time") and t["open_time"].date() == now.date()
            ]

            closed = [t for t in today_trades if t.get("status") != "OPEN"]
            wins = [t for t in closed if (t.get("profit_usd") or 0) > 0]
            losses = [t for t in closed if (t.get("profit_usd") or 0) <= 0]
            open_count = len([t for t in today_trades if t.get("status") == "OPEN"])

            pnl = sum(float(t.get("profit_usd") or 0) for t in closed)
            ratchet_saves = sum(1 for t in closed if t.get("ratchet_triggered"))
            ratchet_protected = sum(
                float(t.get("profit_locked_usd") or 0) for t in closed
                if t.get("ratchet_triggered")
            )

            winner_rs = [
                float(t.get("profit_R") or 0) for t in wins if t.get("profit_R")
            ]
            loser_rs = [
                abs(float(t.get("profit_R") or 0)) for t in losses if t.get("profit_R")
            ]
            avg_w = sum(winner_rs) / len(winner_rs) if winner_rs else 0
            avg_l = sum(loser_rs) / len(loser_rs) if loser_rs else 0

            account = self.data_fetcher.get_live_account()
            balance = account.get("balance", 0) if account else 0
            pnl_pct = (pnl / balance * 100) if balance > 0 else 0

            htf_score_str = await get_bot_state(self.db_pool, "htf_score")
            htf_avg = float(htf_score_str) if htf_score_str else 0.0

            summary_data = {
                "date": now,
                "trades_total": len(today_trades),
                "wins": len(wins),
                "losses": len(losses),
                "open_count": open_count,
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct,
                "ratchet_saves": ratchet_saves,
                "ratchet_protected_usd": ratchet_protected,
                "avg_winner_r": avg_w,
                "avg_loser_r": avg_l,
                "regime_summary": self.current_regime,
                "htf_avg": htf_avg,
                "learning_summary": (
                    f"{self.completed_trades_since_learning} trades since last cycle"
                ),
                "balance": balance,
            }

            if self.telegram_bot:
                await self.telegram_bot.alert_daily_summary(summary_data)
                logger.info("Daily summary sent")

        except Exception:
            logger.error("Daily summary task failed", exc_info=True)

    # ================================================================== #
    #  HELPER METHODS                                                     #
    # ================================================================== #

    @staticmethod
    def _compute_profit_r(position: dict, current_price: float) -> float:
        """Compute the current profit in R-multiples."""
        direction = position.get("direction", "BUY").upper()
        open_price = position.get("open_price", 0.0)
        initial_risk_usd = position.get("initial_risk_usd", 0.0)
        pip_value = position.get("pip_value", 1.0)
        pip_size = position.get("pip_size", _PIP_SIZE)
        lot_size = position.get("lot_size", 0.0)

        if direction == "BUY":
            profit_pips = (current_price - open_price) / pip_size
        else:
            profit_pips = (open_price - current_price) / pip_size

        profit_usd = profit_pips * pip_value * lot_size
        if initial_risk_usd <= 0:
            return 0.0
        return profit_usd / initial_risk_usd

    @staticmethod
    def _ratchet_label(profit_r: float) -> str:
        """Return the human-readable label for the highest ratchet level reached."""
        label = "none"
        for r_threshold, lock_spec in RatchetSLManager.RATCHET:
            if profit_r >= r_threshold:
                if lock_spec == "breakeven":
                    label = "breakeven"
                else:
                    label = f"lock_{int(float(lock_spec) * 100)}pct"
        return label

    @staticmethod
    def _build_feature_snapshot(
        signal_result: dict,
        htf_bias: dict,
        llm_result: dict,
        spread: float,
    ) -> dict:
        """Build a compact feature snapshot for the trade record."""
        return {
            "buy_score": signal_result.get("buy_score"),
            "sell_score": signal_result.get("sell_score"),
            "threshold": signal_result.get("threshold"),
            "is_counter_trend": signal_result.get("is_counter_trend"),
            "htf_score": htf_bias.get("score"),
            "htf_direction": htf_bias.get("direction"),
            "d1": htf_bias.get("d1"),
            "h4": htf_bias.get("h4"),
            "h1": htf_bias.get("h1"),
            "m15_at_support": htf_bias.get("m15_at_support"),
            "m15_at_resist": htf_bias.get("m15_at_resist"),
            "llm_action": llm_result.get("action"),
            "llm_reason": llm_result.get("reason"),
            "spread_pips": spread,
        }

    # ================================================================== #
    #  SHUTDOWN                                                           #
    # ================================================================== #

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("Shutdown requested")
        self.running = False

        # Stop scheduler
        if self.scheduler:
            try:
                await self.scheduler.stop()
                logger.info("Scheduler stopped")
            except Exception:
                logger.warning("Scheduler stop error", exc_info=True)

        # Stop watchdog
        if self.watchdog:
            try:
                await self.watchdog.stop()
                logger.info("Watchdog stopped")
            except Exception:
                logger.warning("Watchdog stop error", exc_info=True)

        # Stop Telegram bot
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_alert(
                    "<b>XAUUSD Agent shutting down</b>"
                )
                await self.telegram_bot.stop()
                logger.info("Telegram bot stopped")
            except Exception:
                logger.warning("Telegram bot stop error", exc_info=True)

        # Disconnect MT5
        if self.connector:
            try:
                self.connector.disconnect()
                logger.info("MT5 disconnected")
            except Exception:
                logger.warning("MT5 disconnect error", exc_info=True)

        # Close DB pool
        if self.db_pool:
            try:
                await self.db_pool.close()
                logger.info("Database pool closed")
            except Exception:
                logger.warning("Database pool close error", exc_info=True)

        # Persist halted state
        logger.info("XAUUSD Agent v2.0 shut down")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_llm_result(reason: str = "below LLM threshold") -> dict:
    """Return a default LLM result that does not block trades."""
    return {"action": "CONFIRM", "reason": reason}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Launch the XAUUSD trading agent."""
    agent = TradingAgent()

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(agent.shutdown()),
        )

    try:
        await agent.setup()
        logger.info("XAUUSD Agent v2.0 started -- entering main loops")

        if agent.telegram_bot:
            try:
                await agent.telegram_bot.send_alert(
                    "<b>XAUUSD Agent v2.0 started</b>"
                )
            except Exception:
                pass

        # Start the scheduler
        if agent.scheduler:
            await agent.scheduler.start()

        # Run both loops concurrently
        await asyncio.gather(
            agent.signal_loop(),
            agent.ratchet_loop(),
        )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.critical("Fatal error in main", exc_info=True)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
