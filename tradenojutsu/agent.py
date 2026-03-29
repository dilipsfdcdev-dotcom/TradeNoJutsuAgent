"""TradeNoJutsu Agent - Main orchestrator that ties everything together.

This is the central nervous system of the trading agent. It runs the
continuous loop: fetch data -> analyze -> think -> decide -> execute -> learn.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tradenojutsu.analysis.technical import TechnicalAnalyzer
from tradenojutsu.brain.reasoner import AIReasoner
from tradenojutsu.brain.signal_combiner import SignalCombiner
from tradenojutsu.config import load_settings
from tradenojutsu.data.market_data import MarketDataFetcher
from tradenojutsu.data.models import Direction, PerformanceMetrics
from tradenojutsu.data.news_sentiment import NewsSentimentAnalyzer
from tradenojutsu.execution.paper_trader import PaperTrader
from tradenojutsu.infra.database import insert_signal
from tradenojutsu.infra.logger import get_logger, setup_logging
from tradenojutsu.learning.ml_ensemble import MLEnsemble
from tradenojutsu.learning.self_learner import SelfLearner
from tradenojutsu.risk.manager import RiskManager
from tradenojutsu.strategy.strategies import get_all_strategies

logger = get_logger("agent")


class TradeNoJutsuAgent:
    """The autonomous self-thinking, self-learning trading agent.

    Lifecycle:
    1. Initialize all components from config
    2. Run continuous analysis loop
    3. For each cycle:
       a. Fetch latest market data
       b. Compute technical indicators
       c. Run strategy evaluations
       d. Get ML ensemble predictions
       e. AI Brain thinks (chain-of-thought reasoning)
       f. Combine all signals
       g. Risk check and execute if approved
       h. Update open positions
       i. Self-learning cycle if threshold met
    """

    def __init__(self, config_path: str | None = None):
        settings = load_settings()
        self.settings = settings
        self.mode = settings["agent"]["mode"]

        setup_logging(settings["agent"].get("log_level", "INFO"))
        logger.info(f"Initializing TradeNoJutsu Agent (mode={self.mode})")

        # Components
        self.data_fetcher = MarketDataFetcher(
            source=settings["trading"].get("data_source", "yfinance"),
            exchange=settings["trading"].get("exchange", "binance"),
        )
        self.analyzer = TechnicalAnalyzer()

        self.reasoner = AIReasoner(
            model=settings["agent"].get("llm_model", "claude-sonnet-4-6"),
            api_key=settings.get("secrets", {}).get("anthropic_api_key"),
        )

        self.ml_ensemble = MLEnsemble()
        self.ml_ensemble.load_models()

        sig_cfg = settings.get("signals", {})
        weights = sig_cfg.get("model_weights", {})
        self.combiner = SignalCombiner(
            technical_weight=weights.get("technical", 0.30),
            ml_weight=weights.get("ml_ensemble", 0.35),
            llm_weight=weights.get("llm_reasoning", 0.35),
            entry_threshold=sig_cfg.get("entry_threshold", 65),
            min_confluence=sig_cfg.get("min_confluence", 2),
        )

        risk_cfg = settings.get("risk", {})
        self.risk_manager = RiskManager(
            capital=settings.get("backtest", {}).get("initial_capital", 10000),
            risk_per_trade_pct=risk_cfg.get("risk_per_trade_pct", 1.0),
            risk_per_trade_max=risk_cfg.get("risk_per_trade_max", 2.0),
            max_concurrent=risk_cfg.get("max_concurrent_positions", 3),
            max_daily_trades=risk_cfg.get("max_daily_trades", 15),
            daily_drawdown_max_pct=risk_cfg.get("daily_drawdown_max_pct", 3.0),
            max_portfolio_risk_pct=risk_cfg.get("max_portfolio_risk_pct", 6.0),
            sl_atr_mult=risk_cfg.get("stop_loss_atr_mult", 1.5),
            tp_rr_ratio=risk_cfg.get("take_profit_rr_ratio", 2.0),
            trailing_stop_enabled=risk_cfg.get("trailing_stop_enabled", True),
            trailing_stop_atr_mult=risk_cfg.get("trailing_stop_atr_mult", 1.0),
        )

        self.paper_trader = PaperTrader(self.risk_manager)
        self.strategies = get_all_strategies()

        learn_cfg = settings.get("learning", {})
        self.learner = SelfLearner(
            reasoner=self.reasoner if learn_cfg.get("enabled", True) else None,
            cycle_every_n=learn_cfg.get("cycle_every_n_trades", 10),
            max_changes_per_cycle=learn_cfg.get("max_param_changes_per_cycle", 3),
            reward_metric=learn_cfg.get("reward_metric", "sharpe"),
        )

        # News sentiment analyzer
        self.news_analyzer = NewsSentimentAnalyzer(
            llm_model=settings["agent"].get("news_llm_model", "claude-haiku-4-5-20251001"),
            api_key=settings.get("secrets", {}).get("anthropic_api_key"),
        )

        # Telegram alerts (optional)
        self.telegram = None
        tg_cfg = settings.get("telegram", {})
        if tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
            from tradenojutsu.telegram_bot.alerts import TelegramAlerter
            self.telegram = TelegramAlerter(tg_cfg["bot_token"], tg_cfg["chat_id"])

        self.symbols = settings["trading"].get("symbols", ["XAUUSD"])
        self.timeframes = settings["trading"].get("primary_timeframes", ["15m"])
        self._running = False

    async def run(self, interval_seconds: int = 60) -> None:
        """Main agent loop - runs continuously."""
        self._running = True
        logger.info(
            f"TradeNoJutsu Agent started | Symbols: {self.symbols} | "
            f"Mode: {self.mode} | Interval: {interval_seconds}s"
        )

        cycle = 0
        while self._running:
            cycle += 1
            try:
                logger.info(f"--- Cycle {cycle} ---")
                await self._run_cycle()
            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}", exc_info=True)

            await asyncio.sleep(interval_seconds)

    async def _run_cycle(self) -> None:
        """Execute one full analysis-decision-execution cycle."""
        for symbol in self.symbols:
            for tf in self.timeframes:
                await self._analyze_symbol(symbol, tf)

    async def _analyze_symbol(self, symbol: str, timeframe: str) -> None:
        """Full pipeline for a single symbol+timeframe."""

        # 1. Fetch data
        df = self.data_fetcher.fetch_ohlcv(symbol, timeframe, lookback_days=180)
        if df.empty:
            logger.warning(f"No data for {symbol} ({timeframe})")
            return

        # 2. Compute indicators
        df = self.analyzer.compute_indicators(df)

        # 3. Build market state
        market_state = self.analyzer.build_market_state(symbol, df)
        logger.info(f"{symbol}: price={market_state.price:.4f} regime={market_state.regime.value}")

        # 4. Strategy signals
        best_strategy_signal = None
        for strategy in self.strategies:
            sig = strategy.evaluate(df, symbol)
            if sig and sig.is_actionable:
                if best_strategy_signal is None or sig.score > best_strategy_signal.score:
                    best_strategy_signal = sig

        # 5. Technical score
        direction = market_state.trend_direction
        if best_strategy_signal:
            direction = best_strategy_signal.direction
        tech_score, tech_components = self.analyzer.score_signal(df, direction)

        # 6. ML ensemble prediction
        ml_score, ml_direction = self.ml_ensemble.predict(df)

        # 6b. News sentiment analysis
        try:
            sentiment = self.news_analyzer.analyze_sentiment(symbol)
            market_state.indicators["news_sentiment"] = sentiment.overall_sentiment
            logger.info(
                f"News sentiment: {sentiment.sentiment_label} "
                f"({sentiment.overall_sentiment:+.2f}, {sentiment.news_count} headlines)"
            )
        except Exception as e:
            logger.warning(f"News sentiment failed: {e}")
            sentiment = None

        # 7. AI Brain thinks (the "self-thinking" part)
        llm_decision = self.reasoner.think(
            market_state=market_state,
            technical_scores=tech_components,
        )

        # 8. Combine all signals
        combined = self.combiner.combine(
            symbol=symbol,
            technical_score=tech_score,
            technical_direction=direction,
            technical_components=tech_components,
            ml_score=ml_score,
            ml_direction=ml_direction,
            llm_decision=llm_decision,
        )

        # Log signal
        insert_signal(
            symbol=symbol,
            direction=combined.direction.value,
            score=combined.score,
            components=combined.components,
            reasoning=combined.reasoning,
            accepted=combined.is_actionable,
        )

        logger.info(
            f"Signal: {combined.direction.value} {symbol} | "
            f"Score: {combined.score:.1f} | Strength: {combined.strength.value}"
        )

        # 9. Execute if actionable
        if combined.is_actionable and self.mode in ("paper", "live"):
            price = market_state.price
            atr = market_state.indicators.get("atr_pct", 1.0) / 100 * price

            risk_params = self.risk_manager.calculate_risk_params(
                combined, price, atr, market_state.regime
            )
            trade = self.paper_trader.execute_signal(combined, price, risk_params)

            if trade:
                logger.info(f"Trade executed: {trade.direction.value} {trade.symbol}")
                if self.telegram:
                    await self.telegram.alert_trade_opened(trade, combined)

        # 10. Update open positions
        prices = {symbol: market_state.price}
        atr_val = market_state.indicators.get("atr_pct", 1.0) / 100 * market_state.price
        atrs = {symbol: atr_val}
        closed_trades = self.paper_trader.update_positions(prices, atrs)

        # 11. Self-learning check + alerts for closed trades
        for ct in closed_trades:
            if self.telegram:
                await self.telegram.alert_trade_closed(ct)
            result = self.learner.on_trade_closed()
            if result:
                logger.info(f"Learning cycle complete: {result.get('changes', [])}")
                if self.telegram:
                    await self.telegram.alert_learning_cycle(result)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent stopping...")

    def run_backtest(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "1d",
        lookback_days: int = 365,
    ) -> Any:
        """Run a backtest on historical data."""
        from tradenojutsu.backtest.engine import BacktestEngine

        df = self.data_fetcher.fetch_ohlcv(symbol, timeframe, lookback_days)
        if df.empty:
            logger.error(f"No data for backtest: {symbol}")
            return None

        engine = BacktestEngine(
            initial_capital=self.settings.get("backtest", {}).get("initial_capital", 10000),
            commission_pct=self.settings.get("backtest", {}).get("commission_pct", 0.1),
            slippage_pct=self.settings.get("backtest", {}).get("slippage_pct", 0.05),
        )

        return engine.run(df, symbol, self.strategies)
