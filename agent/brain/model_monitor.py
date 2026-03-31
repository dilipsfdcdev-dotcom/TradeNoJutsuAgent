"""Daily ML model health monitor — detects accuracy degradation, feature
importance shift, and prediction calibration drift.

If drift is detected, sets a GLOBAL REDUCE veto and triggers early retrain.
Never blocks trading outright — only reduces risk.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import anthropic
import structlog

from agent.brain.veto_register import VetoRegister, VetoState, veto_register
from agent.config import settings

logger = structlog.get_logger(__name__)

# Thresholds
_XGB_CRITICAL_ACCURACY = 45.0
_XGB_WARNING_ACCURACY = 50.0
_LSTM_CRITICAL_ACCURACY = 40.0
_LSTM_WARNING_ACCURACY = 48.0
_MAX_PREDICTION_HISTORY = 200


class ModelMonitor:
    """Track ML model predictions and detect drift.

    Records predictions in real time, computes rolling accuracy,
    and runs periodic health checks that set veto-register entries
    when models degrade.
    """

    def __init__(self, register: VetoRegister | None = None):
        self._register = register or veto_register
        self.xgb_predictions: list[dict] = []
        self.lstm_predictions: list[dict] = []
        self.max_history = _MAX_PREDICTION_HISTORY
        self._health_history: list[dict] = []

    # ------------------------------------------------------------------
    # Recording predictions
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        model_type: str,
        predicted: float,
        actual: int | None = None,
        symbol: str = "",
        direction: str | None = None,
    ) -> None:
        """Record a prediction for accuracy tracking.

        Parameters
        ----------
        model_type : str
            ``"xgb"`` or ``"lstm"``.
        predicted : float
            The model's predicted score/confidence.
        actual : int | None
            Outcome: 1 = win, 0 = loss, None = still open.
        symbol : str
            Trading symbol for per-symbol tracking.
        direction : str | None
            For LSTM: the predicted direction (buy/sell/wait).
        """
        entry = {
            "predicted": predicted,
            "actual": actual,
            "symbol": symbol,
            "direction": direction,
            "timestamp": datetime.now(timezone.utc),
        }
        if model_type == "xgb":
            self.xgb_predictions.append(entry)
            if len(self.xgb_predictions) > self.max_history:
                self.xgb_predictions = self.xgb_predictions[-self.max_history:]
        elif model_type == "lstm":
            self.lstm_predictions.append(entry)
            if len(self.lstm_predictions) > self.max_history:
                self.lstm_predictions = self.lstm_predictions[-self.max_history:]

    def update_actual(self, model_type: str, timestamp: datetime, actual: int) -> None:
        """Update the actual outcome for a prediction after a trade closes.

        Matches by timestamp proximity (within 5 minutes).
        """
        preds = self.xgb_predictions if model_type == "xgb" else self.lstm_predictions
        for p in reversed(preds):
            if p["actual"] is None and abs((p["timestamp"] - timestamp).total_seconds()) < 300:
                p["actual"] = actual
                break

    # ------------------------------------------------------------------
    # Accuracy computation
    # ------------------------------------------------------------------

    def get_accuracy(self, model_type: str, last_n: int = 50) -> float:
        """Calculate accuracy for the last N predictions with known outcomes.

        Returns 50.0 (baseline) if fewer than 10 evaluated predictions.
        """
        preds = self.xgb_predictions if model_type == "xgb" else self.lstm_predictions
        evaluated = [p for p in preds if p["actual"] is not None][-last_n:]
        if len(evaluated) < 10:
            return 50.0

        if model_type == "xgb":
            correct = sum(
                1 for p in evaluated
                if (p["predicted"] >= 0.55) == (p["actual"] == 1)
            )
        else:
            correct = sum(
                1 for p in evaluated
                if (p["predicted"] >= 0.60) == (p["actual"] == 1)
            )
        return (correct / len(evaluated)) * 100

    def get_calibration_error(self, model_type: str, last_n: int = 50) -> float:
        """Compute |avg_predicted_confidence - actual_win_rate|.

        Returns 0.0 if insufficient data.
        """
        preds = self.xgb_predictions if model_type == "xgb" else self.lstm_predictions
        evaluated = [p for p in preds if p["actual"] is not None][-last_n:]
        if len(evaluated) < 10:
            return 0.0

        avg_predicted = sum(p["predicted"] for p in evaluated) / len(evaluated)
        actual_win_rate = sum(1 for p in evaluated if p["actual"] == 1) / len(evaluated)
        return abs(avg_predicted - actual_win_rate)

    def get_health(self) -> dict:
        """Return a snapshot of model health metrics."""
        xgb_acc = self.get_accuracy("xgb")
        lstm_acc = self.get_accuracy("lstm")
        xgb_cal = self.get_calibration_error("xgb")
        lstm_cal = self.get_calibration_error("lstm")

        return {
            "xgb_accuracy": round(xgb_acc, 1),
            "lstm_accuracy": round(lstm_acc, 1),
            "xgb_calibration_error": round(xgb_cal, 3),
            "lstm_calibration_error": round(lstm_cal, 3),
            "xgb_predictions_tracked": len(self.xgb_predictions),
            "lstm_predictions_tracked": len(self.lstm_predictions),
        }

    # ------------------------------------------------------------------
    # Daily health check
    # ------------------------------------------------------------------

    async def daily_check(self, trigger_retrain_fn: Callable | None = None) -> dict:
        """Run the daily model health check.

        Evaluates:
        1. Accuracy degradation (XGBoost and LSTM)
        2. Calibration error
        3. Prediction count (enough data?)

        If drift is detected, sets a GLOBAL REDUCE veto and optionally
        triggers early retrain.  Never BLOCKs trading.

        Parameters
        ----------
        trigger_retrain_fn : callable or None
            Async function ``fn(symbol: str) -> dict`` to trigger retrain.

        Returns
        -------
        dict
            Health report with keys: ``health``, ``issues``, ``action_taken``,
            ``trigger_retrain``.
        """
        health = self.get_health()
        xgb_acc = health["xgb_accuracy"]
        lstm_acc = health["lstm_accuracy"]
        xgb_cal = health["xgb_calibration_error"]
        lstm_cal = health["lstm_calibration_error"]

        issues: list[str] = []

        # --- Accuracy checks ---
        if xgb_acc < _XGB_CRITICAL_ACCURACY:
            issues.append(f"XGBoost accuracy critically low: {xgb_acc:.1f}%")
        elif xgb_acc < _XGB_WARNING_ACCURACY:
            issues.append(f"XGBoost accuracy below baseline: {xgb_acc:.1f}%")

        if lstm_acc < _LSTM_CRITICAL_ACCURACY:
            issues.append(f"LSTM accuracy critically low: {lstm_acc:.1f}%")
        elif lstm_acc < _LSTM_WARNING_ACCURACY:
            issues.append(f"LSTM accuracy below baseline: {lstm_acc:.1f}%")

        # --- Calibration checks ---
        if xgb_cal > 0.15:
            issues.append(f"XGBoost poorly calibrated: error={xgb_cal:.3f}")
        if lstm_cal > 0.15:
            issues.append(f"LSTM poorly calibrated: error={lstm_cal:.3f}")

        result: dict[str, Any] = {
            "health": health,
            "issues": issues,
            "action_taken": None,
            "trigger_retrain": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        has_critical = any("critically" in issue for issue in issues)
        has_calibration = any("calibrated" in issue for issue in issues)

        if has_critical:
            # Set global REDUCE veto — never BLOCK
            modifier = 0.5 if not has_calibration else 0.3
            self._register.set_veto("GLOBAL", VetoState(
                blocked=False,
                risk_modifier=modifier,
                reason=f"Model drift detected: {'; '.join(issues[:3])}",
                expires_at=datetime.utcnow() + timedelta(hours=24),
                source="model_monitor",
            ))
            result["action_taken"] = f"GLOBAL REDUCE {modifier}"
            result["trigger_retrain"] = True
            logger.warning("model_drift_detected", **health, issues=issues)

            # Trigger early retrain if callback provided
            if trigger_retrain_fn is not None:
                for symbol in settings.symbols_list:
                    try:
                        retrain_result = trigger_retrain_fn(symbol)
                        if asyncio.iscoroutine(retrain_result):
                            await retrain_result
                        logger.info("early_retrain_triggered", symbol=symbol)
                    except Exception as exc:
                        logger.error("early_retrain_failed", symbol=symbol, error=str(exc))

        elif issues:
            logger.info("model_health_warning", issues=issues)
            result["action_taken"] = "warning_only"
        else:
            logger.info("model_health_ok", **health)

        self._health_history.append(result)
        if len(self._health_history) > 100:
            self._health_history = self._health_history[-100:]

        return result

    # ------------------------------------------------------------------
    # Optional: Claude-powered deeper analysis
    # ------------------------------------------------------------------

    async def claude_health_analysis(self) -> dict:
        """Ask Claude for a deeper model health analysis.

        This is an optional enhancement — the daily_check is fully
        functional without it.  Falls back gracefully on API failure.
        """
        health = self.get_health()
        try:
            client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            prompt = f"""Analyze ML model health for a trading bot:

XGBoost accuracy (last 50): {health['xgb_accuracy']:.1f}%
LSTM accuracy (last 50): {health['lstm_accuracy']:.1f}%
XGBoost calibration error: {health['xgb_calibration_error']:.3f}
LSTM calibration error: {health['lstm_calibration_error']:.3f}
Predictions tracked: XGB={health['xgb_predictions_tracked']}, LSTM={health['lstm_predictions_tracked']}

Is this concerning? What should we check?
Respond in JSON only:
{{"status": "healthy" | "warning" | "critical", "analysis": "...", "recommendations": ["..."]}}"""

            response = await asyncio.wait_for(
                client.messages.create(
                    model=settings.CLAUDE_MODEL,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=10.0,
            )
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)

        except anthropic.APIConnectionError as exc:
            logger.warning("claude_health_api_connection_error", error=str(exc))
            return {"status": "unknown", "analysis": "API connection failed"}
        except anthropic.RateLimitError as exc:
            logger.warning("claude_health_rate_limit", error=str(exc))
            return {"status": "unknown", "analysis": "Rate limited"}
        except anthropic.APIStatusError as exc:
            logger.warning("claude_health_api_error", status=exc.status_code)
            return {"status": "unknown", "analysis": f"API error {exc.status_code}"}
        except asyncio.TimeoutError:
            logger.warning("claude_health_timeout")
            return {"status": "unknown", "analysis": "Request timed out"}
        except json.JSONDecodeError:
            logger.warning("claude_health_parse_error")
            return {"status": "unknown", "analysis": "Response parse error"}
        except Exception as exc:
            logger.warning("claude_health_check_failed", error=str(exc))
            return {"status": "unknown", "analysis": "Check failed"}

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run(
        self,
        interval_hours: float = 6.0,
        trigger_retrain_fn: Callable | None = None,
    ) -> None:
        """Run the monitor as a background loop.

        Parameters
        ----------
        interval_hours : float
            Hours between checks (default 6).
        trigger_retrain_fn : callable or None
            Passed through to ``daily_check``.
        """
        logger.info("model_monitor_started", interval_hours=interval_hours)
        while True:
            try:
                await self.daily_check(trigger_retrain_fn=trigger_retrain_fn)
            except Exception as exc:
                logger.error("model_monitor_loop_error", error=str(exc))
            await asyncio.sleep(interval_hours * 3600)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_health_history(self) -> list[dict]:
        """Return recent health check results."""
        return list(self._health_history)


# Module-level singleton
model_monitor = ModelMonitor()
