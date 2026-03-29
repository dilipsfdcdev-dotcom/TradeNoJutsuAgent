"""Signal combination engine - blends technical, ML, and LLM signals."""

from __future__ import annotations

from tradenojutsu.data.models import Direction, Signal, SignalStrength, MarketState
from tradenojutsu.infra.logger import get_logger

logger = get_logger("brain.combiner")


class SignalCombiner:
    """Combines multiple signal sources into a unified trading decision.

    Weights are tunable by the self-learning module.
    """

    def __init__(
        self,
        technical_weight: float = 0.30,
        ml_weight: float = 0.35,
        llm_weight: float = 0.35,
        entry_threshold: float = 65.0,
        min_confluence: int = 2,
    ):
        self.technical_weight = technical_weight
        self.ml_weight = ml_weight
        self.llm_weight = llm_weight
        self.entry_threshold = entry_threshold
        self.min_confluence = min_confluence

    def combine(
        self,
        symbol: str,
        technical_score: float,
        technical_direction: Direction,
        technical_components: dict[str, float],
        ml_score: float,
        ml_direction: Direction,
        llm_decision: dict,
    ) -> Signal:
        """Combine all signal sources into a final Signal.

        Args:
            technical_score: 0-100 from technical analysis
            technical_direction: direction from technicals
            technical_components: breakdown of technical scores
            ml_score: 0-100 from ML ensemble
            ml_direction: direction from ML model
            llm_decision: dict from AIReasoner.think()
        """
        llm_direction = self._parse_llm_direction(llm_decision.get("decision", "wait"))
        llm_confidence = float(llm_decision.get("confidence", 0))

        # Determine consensus direction
        directions = [
            (technical_direction, self.technical_weight),
            (ml_direction, self.ml_weight),
            (llm_direction, self.llm_weight),
        ]

        long_weight = sum(w for d, w in directions if d == Direction.LONG)
        short_weight = sum(w for d, w in directions if d == Direction.SHORT)

        if long_weight > short_weight and long_weight > 0.4:
            direction = Direction.LONG
        elif short_weight > long_weight and short_weight > 0.4:
            direction = Direction.SHORT
        else:
            direction = Direction.FLAT

        # Weighted composite score
        composite = (
            technical_score * self.technical_weight
            + ml_score * self.ml_weight
            + llm_confidence * self.llm_weight
        )

        # Count agreeing sources (confluence)
        active_directions = [d for d, _ in directions if d != Direction.FLAT]
        confluence = sum(1 for d in active_directions if d == direction)

        # Determine strength
        if composite >= 80 and confluence >= 3:
            strength = SignalStrength.STRONG
        elif composite >= self.entry_threshold and confluence >= self.min_confluence:
            strength = SignalStrength.MODERATE
        elif composite >= 50:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE
            direction = Direction.FLAT

        # Build reasoning
        reasoning_parts = [
            f"Technical: {technical_direction.value} ({technical_score:.0f}/100)",
            f"ML Model: {ml_direction.value} ({ml_score:.0f}/100)",
            f"AI Brain: {llm_direction.value} ({llm_confidence:.0f}% confidence)",
            f"Confluence: {confluence}/{len(active_directions)} sources agree",
            f"Composite score: {composite:.1f}/100 (threshold: {self.entry_threshold})",
        ]
        if llm_decision.get("thinking"):
            reasoning_parts.append(f"AI reasoning: {llm_decision['thinking'][:200]}...")

        components = {
            "technical": technical_score,
            "ml_ensemble": ml_score,
            "llm_confidence": llm_confidence,
            "confluence": float(confluence),
            **{f"tech_{k}": v for k, v in technical_components.items()},
        }

        return Signal(
            symbol=symbol,
            direction=direction,
            score=composite,
            strength=strength,
            strategy="combined",
            reasoning="\n".join(reasoning_parts),
            components=components,
        )

    def _parse_llm_direction(self, decision: str) -> Direction:
        decision = decision.lower().strip()
        if decision in ("long", "buy"):
            return Direction.LONG
        if decision in ("short", "sell"):
            return Direction.SHORT
        return Direction.FLAT

    def update_weights(self, technical: float, ml: float, llm: float) -> None:
        """Update combination weights (called by learning module)."""
        total = technical + ml + llm
        if total > 0:
            self.technical_weight = technical / total
            self.ml_weight = ml / total
            self.llm_weight = llm / total
        logger.info(
            f"Updated weights: tech={self.technical_weight:.2f} "
            f"ml={self.ml_weight:.2f} llm={self.llm_weight:.2f}"
        )
