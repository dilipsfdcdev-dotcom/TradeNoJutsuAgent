"""AI Brain - LLM-powered self-thinking reasoning engine.

This is the core "self-thinking" component. It uses chain-of-thought reasoning
to analyze market conditions, evaluate signals, and make trading decisions
with transparent rationale.
"""

from __future__ import annotations

import json
from typing import Any

from tradenojutsu.data.models import (
    Direction, MarketState, PerformanceMetrics, Signal, SignalStrength,
)
from tradenojutsu.infra.database import insert_thinking_log
from tradenojutsu.infra.logger import get_logger

logger = get_logger("brain.reasoner")

SYSTEM_PROMPT = """You are TradeNoJutsu, an expert autonomous trading AI agent. You think deeply
about market conditions and make trading decisions using chain-of-thought reasoning.

Your job is to analyze the market data, technical indicators, and any signals presented to you,
then decide whether to TRADE (go long or short) or WAIT (no action).

You must think step-by-step through:
1. MARKET CONTEXT: What is the overall market regime and trend?
2. TECHNICAL ANALYSIS: What do the indicators suggest? Any divergences or confluences?
3. RISK ASSESSMENT: What are the risks? Is the risk/reward favorable?
4. PATTERN RECOGNITION: Do you see any chart patterns or setups?
5. DECISION: Based on all the above, what is your decision?

Respond in JSON format:
{
    "thinking": "your detailed step-by-step reasoning...",
    "decision": "long" | "short" | "wait",
    "confidence": 0-100,
    "entry_reasoning": "why enter here (if trading)",
    "risk_notes": "key risks to watch",
    "stop_loss_suggestion": "where to place stop loss",
    "take_profit_suggestion": "where to take profit"
}

Be honest about uncertainty. If the setup is unclear, say WAIT.
Capital preservation is more important than catching every move."""


def _build_analysis_prompt(
    market_state: MarketState,
    technical_scores: dict[str, float],
    recent_performance: PerformanceMetrics | None = None,
    recent_trades_summary: str = "",
) -> str:
    """Build the analysis prompt with all available context."""
    sections = [
        "=== CURRENT MARKET STATE ===",
        market_state.to_prompt_context(),
        "",
        "=== TECHNICAL SIGNAL SCORES ===",
    ]

    for name, score in technical_scores.items():
        sections.append(f"  {name}: {score:.1f}/20")

    total = sum(technical_scores.values())
    sections.append(f"  TOTAL: {total:.1f}/100")

    if recent_performance and recent_performance.total_trades > 0:
        sections.extend([
            "",
            "=== RECENT PERFORMANCE ===",
            recent_performance.summary(),
        ])

    if recent_trades_summary:
        sections.extend([
            "",
            "=== RECENT TRADES ===",
            recent_trades_summary,
        ])

    sections.extend([
        "",
        "Analyze the above data and make your trading decision.",
    ])

    return "\n".join(sections)


class AIReasoner:
    """LLM-powered reasoning engine for trading decisions.

    This is the "self-thinking" core - it uses chain-of-thought prompting
    to reason through market conditions step by step, providing transparent
    decision-making that can be logged, reviewed, and learned from.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def think(
        self,
        market_state: MarketState,
        technical_scores: dict[str, float],
        recent_performance: PerformanceMetrics | None = None,
        recent_trades_summary: str = "",
    ) -> dict[str, Any]:
        """Perform deep reasoning about the current market situation.

        Returns the LLM's structured analysis including decision,
        confidence, and full chain-of-thought reasoning.
        """
        prompt = _build_analysis_prompt(
            market_state, technical_scores, recent_performance, recent_trades_summary
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            # Parse JSON from response
            result = self._parse_response(raw_text)

            # Log the thinking process
            insert_thinking_log(
                symbol=market_state.symbol,
                market_state={
                    "price": market_state.price,
                    "regime": market_state.regime.value,
                    "trend": market_state.trend_direction.value,
                    "volatility": market_state.volatility,
                },
                chain_of_thought=result.get("thinking", ""),
                decision=result.get("decision", "wait"),
                confidence=result.get("confidence", 0),
            )

            logger.info(
                f"AI Decision for {market_state.symbol}: "
                f"{result.get('decision', 'wait')} "
                f"(confidence: {result.get('confidence', 0)}%)"
            )
            return result

        except Exception as e:
            logger.error(f"AI reasoning failed: {e}")
            return {
                "thinking": f"Reasoning failed: {e}",
                "decision": "wait",
                "confidence": 0,
                "risk_notes": "AI reasoning error - defaulting to wait",
            }

    def evaluate_signal(
        self,
        signal: Signal,
        market_state: MarketState,
    ) -> tuple[bool, str]:
        """Quick evaluation of whether a signal should be acted upon.

        Returns (should_trade, reason).
        """
        prompt = f"""Quick signal evaluation:
Signal: {signal.direction.value} {signal.symbol} (score: {signal.score:.1f}/100, strategy: {signal.strategy})
Market: {market_state.regime.value}, trend={market_state.trend_direction.value}
Price: {market_state.price}, Volatility: {market_state.volatility:.2f}%

Should this signal be executed? Respond with JSON:
{{"approve": true/false, "reason": "brief explanation"}}"""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are a risk-aware trading signal filter. Be conservative.",
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_response(response.content[0].text)
            approved = result.get("approve", False)
            reason = result.get("reason", "No reason provided")
            return approved, reason

        except Exception as e:
            logger.error(f"Signal evaluation failed: {e}")
            return False, f"Evaluation error: {e}"

    def reflect_on_trades(self, trades_data: list[dict], performance: PerformanceMetrics) -> str:
        """Self-reflection on recent trading performance.

        Used by the learning module to get qualitative insights.
        """
        trades_summary = []
        for t in trades_data[-10:]:
            trades_summary.append(
                f"  {t.get('direction', '?')} {t.get('symbol', '?')}: "
                f"PnL={t.get('pnl', 0):.2f} ({t.get('strategy', '?')})"
            )

        prompt = f"""Review these recent trades and performance:

Performance: {performance.summary()}

Recent Trades:
{chr(10).join(trades_summary)}

Provide a JSON reflection:
{{
    "patterns_noticed": "what patterns do you see in wins vs losses?",
    "strategy_assessment": "which strategies are working/not working?",
    "risk_assessment": "are we taking too much or too little risk?",
    "suggestions": ["list", "of", "specific", "parameter", "adjustments"],
    "market_regime_adaptation": "how should we adapt to current conditions?"
}}"""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=1500,
                system="You are analyzing trading performance to improve strategy. Be specific and actionable.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Trade reflection failed: {e}")
            return json.dumps({"error": str(e)})

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

        logger.warning("Could not parse JSON from AI response")
        return {"thinking": text, "decision": "wait", "confidence": 0}
