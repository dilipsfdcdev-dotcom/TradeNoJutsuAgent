"""
Final signal combiner.

Merges outputs from the rule-based signal engine, PPO reinforcement-learning
agent, LightGBM classifier, LLM reasoner, and news filter into a single
actionable decision: **BUY**, **SELL**, or **HOLD** with an associated
confidence and human-readable reason.
"""

from __future__ import annotations

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


def combine(
    signal_engine_result: dict,
    ppo_action: int,
    ppo_confidence: float,
    lgbm_prob: float,
    llm_result: dict,
    news_blackout: bool,
    daily_drawdown_pct: float,
    open_positions: int,
    max_positions: int,
    regime: str,
    settings: dict,
) -> tuple[str, float, str]:
    """
    Combine all model outputs into a final trade decision.

    Parameters
    ----------
    signal_engine_result : dict
        Output of :meth:`M3M5SignalEngine.get_signal`.
    ppo_action : int
        PPO agent action: ``-1`` (sell), ``0`` (hold), ``1`` (buy).
    ppo_confidence : float
        PPO agent confidence score (0-100 scale).
    lgbm_prob : float
        LightGBM predicted probability of a profitable *buy* (0.0-1.0).
    llm_result : dict
        Output of :func:`llm_reasoner.check` with keys ``action`` and ``reason``.
    news_blackout : bool
        ``True`` if a high-impact event blackout is active.
    daily_drawdown_pct : float
        Current session drawdown as a positive percentage (e.g., 2.5 = 2.5 %).
    open_positions : int
        Number of currently open positions.
    max_positions : int
        Maximum allowed concurrent positions.
    regime : str
        Current volatility regime label (e.g., ``"HIGH_VOLATILE"``).
    settings : dict
        Tuneable weights and thresholds.

    Returns
    -------
    tuple[str, float, str]
        ``(action, confidence, reason)`` where action is ``"BUY"``,
        ``"SELL"``, or ``"HOLD"``.
    """

    buy_score = float(signal_engine_result.get("buy_score", 0.0))
    sell_score = float(signal_engine_result.get("sell_score", 0.0))
    entry_threshold = float(settings.get("entry_threshold", 62))

    # ------------------------------------------------------------------ #
    # Hard stops  (always block)                                          #
    # ------------------------------------------------------------------ #

    if news_blackout:
        reason = "HOLD: news blackout active"
        logger.info(reason)
        return "HOLD", 0.0, reason

    if daily_drawdown_pct >= 3.0:
        reason = f"HOLD: daily drawdown {daily_drawdown_pct:.1f}% >= 3.0% limit"
        logger.info(reason)
        return "HOLD", 0.0, reason

    if open_positions >= max_positions:
        reason = f"HOLD: {open_positions} open positions >= max {max_positions}"
        logger.info(reason)
        return "HOLD", 0.0, reason

    llm_action = str(llm_result.get("action", "CONFIRM")).upper()
    if llm_action == "VETO":
        reason = f"HOLD: LLM VETO – {llm_result.get('reason', 'no reason')}"
        logger.info(reason)
        return "HOLD", 0.0, reason

    if regime == "HIGH_VOLATILE" and max(buy_score, sell_score) < 80:
        reason = (
            f"HOLD: HIGH_VOLATILE regime requires score >= 80 "
            f"(buy={buy_score:.1f}, sell={sell_score:.1f})"
        )
        logger.info(reason)
        return "HOLD", 0.0, reason

    # ------------------------------------------------------------------ #
    # Weighted combination                                                #
    # ------------------------------------------------------------------ #

    score_weight = float(settings.get("score_weight", 0.35))
    ppo_weight = float(settings.get("ppo_weight", 0.35))
    lgbm_weight = float(settings.get("lgbm_weight", 0.30))

    # Map PPO action to directional confidence
    ppo_conf_buy = ppo_confidence if ppo_action == 1 else 0.0
    ppo_conf_sell = ppo_confidence if ppo_action == -1 else 0.0

    combined_buy = (
        buy_score * score_weight
        + ppo_conf_buy * ppo_weight
        + lgbm_prob * 100.0 * lgbm_weight
    )
    combined_sell = (
        sell_score * score_weight
        + ppo_conf_sell * ppo_weight
        + (1.0 - lgbm_prob) * 100.0 * lgbm_weight
    )

    # Apply LLM REDUCE modifier (scale down by 20 %)
    if llm_action == "REDUCE":
        combined_buy *= 0.80
        combined_sell *= 0.80

    # ------------------------------------------------------------------ #
    # Decision                                                            #
    # ------------------------------------------------------------------ #

    buy_fires = combined_buy >= entry_threshold
    sell_fires = combined_sell >= entry_threshold

    if buy_fires and sell_fires:
        if combined_buy >= combined_sell:
            action, confidence = "BUY", combined_buy
        else:
            action, confidence = "SELL", combined_sell
    elif buy_fires:
        action, confidence = "BUY", combined_buy
    elif sell_fires:
        action, confidence = "SELL", combined_sell
    else:
        reason = (
            f"HOLD: combined scores below threshold "
            f"(buy={combined_buy:.1f}, sell={combined_sell:.1f}, thr={entry_threshold})"
        )
        logger.info(reason)
        return "HOLD", max(combined_buy, combined_sell), reason

    # Cap confidence at 100
    confidence = min(confidence, 100.0)

    reason = (
        f"{action}: combined={confidence:.1f} "
        f"(score={buy_score if action == 'BUY' else sell_score:.1f}*{score_weight} "
        f"+ ppo={ppo_conf_buy if action == 'BUY' else ppo_conf_sell:.1f}*{ppo_weight} "
        f"+ lgbm={'%.2f' % (lgbm_prob if action == 'BUY' else 1 - lgbm_prob)}*{lgbm_weight})"
    )

    logger.info(
        "Signal combined",
        extra={
            "action": action,
            "confidence": round(confidence, 2),
            "combined_buy": round(combined_buy, 2),
            "combined_sell": round(combined_sell, 2),
            "regime": regime,
            "llm_action": llm_action,
        },
    )
    return action, confidence, reason
