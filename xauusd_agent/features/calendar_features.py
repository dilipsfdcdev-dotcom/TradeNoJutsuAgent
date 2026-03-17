"""
Economic calendar feature engineering for XAUUSD.

Transforms upcoming event data into actionable features:
  - Time-to-event
  - Impact scoring
  - Blackout / pre-event flags
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ── Impact scores for major events ────────────────────────────────────────

_EVENT_IMPACT: Dict[str, float] = {
    # Central bank
    "FOMC": 1.0,
    "FED": 1.0,
    "INTEREST RATE": 1.0,
    "RATE DECISION": 1.0,
    "MONETARY POLICY": 0.95,
    "ECB": 0.85,
    "BOE": 0.80,
    "BOJ": 0.75,
    # Employment
    "NFP": 1.0,
    "NON-FARM": 1.0,
    "NONFARM": 1.0,
    "EMPLOYMENT": 0.80,
    "UNEMPLOYMENT": 0.80,
    "JOBLESS CLAIMS": 0.60,
    "ADP": 0.65,
    # Inflation
    "CPI": 0.90,
    "CONSUMER PRICE": 0.90,
    "PPI": 0.70,
    "PRODUCER PRICE": 0.70,
    "PCE": 0.85,
    "CORE PCE": 0.90,
    "INFLATION": 0.85,
    # GDP
    "GDP": 0.70,
    "GROSS DOMESTIC": 0.70,
    # Trade / manufacturing
    "ISM": 0.65,
    "PMI": 0.60,
    "RETAIL SALES": 0.65,
    "TRADE BALANCE": 0.50,
    "DURABLE GOODS": 0.55,
    # Speeches
    "POWELL": 0.90,
    "FED CHAIR": 0.90,
    "YELLEN": 0.70,
    "LAGARDE": 0.65,
    # Housing
    "HOUSING": 0.45,
    "HOME SALES": 0.45,
    "BUILDING PERMITS": 0.40,
    # Consumer confidence
    "CONSUMER CONFIDENCE": 0.55,
    "MICHIGAN": 0.55,
    "SENTIMENT": 0.50,
}

# Threshold (hours) for pre-event flag
_PRE_EVENT_HOURS = 2.0
_NO_EVENT_MINUTES = 999


def _score_event(event: dict) -> float:
    """Compute impact score for a single event.

    Checks the event ``"title"`` (and optionally ``"impact"`` field)
    against known high-impact keywords.
    """
    title = str(event.get("title", "")).upper()
    explicit_impact = str(event.get("impact", "")).upper()

    # If the source already tags it as LOW, cap the score
    if explicit_impact == "LOW":
        return 0.2

    best = 0.0
    for keyword, score in _EVENT_IMPACT.items():
        if keyword in title:
            best = max(best, score)

    # If source says HIGH but we didn't match a keyword, give a floor
    if best == 0.0 and explicit_impact == "HIGH":
        best = 0.7
    elif best == 0.0 and explicit_impact == "MEDIUM":
        best = 0.4
    elif best == 0.0:
        best = 0.3  # unknown event default

    return best


def _minutes_until(event_time: Any) -> int:
    """Calculate minutes from now until the event time."""
    now = datetime.now(tz=timezone.utc)

    if isinstance(event_time, datetime):
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        delta = (event_time - now).total_seconds() / 60.0
        return max(0, int(delta))

    if isinstance(event_time, str):
        # Try fromisoformat first (handles +00:00 and Z suffixes)
        try:
            dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - now).total_seconds() / 60.0
            return max(0, int(delta))
        except (ValueError, TypeError):
            pass

        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                dt = datetime.strptime(event_time, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delta = (dt - now).total_seconds() / 60.0
                return max(0, int(delta))
            except ValueError:
                continue

    return _NO_EVENT_MINUTES


# ── Public API ────────────────────────────────────────────────────────────

def compute_calendar_features(
    upcoming_events: List[Dict[str, Any]],
    blackout: bool = False,
) -> Dict[str, Any]:
    """Derive calendar-aware features from upcoming economic events.

    Parameters
    ----------
    upcoming_events : list[dict]
        Each dict should contain at minimum:
        - ``"title"`` : str – event name (e.g. ``"US CPI MoM"``)
        - ``"time"``  : str | datetime – scheduled time (UTC)
        Optional:
        - ``"impact"`` : str – ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``

    blackout : bool
        External flag indicating whether a trading blackout is active
        (e.g., around known no-trade windows).

    Returns
    -------
    dict
        Calendar feature dictionary.
    """
    logger.info(
        "Computing calendar features: %d events, blackout=%s",
        len(upcoming_events),
        blackout,
    )

    features: Dict[str, Any] = {
        "is_blackout": blackout,
        "next_event_minutes": _NO_EVENT_MINUTES,
        "event_impact_score": 0.0,
        "pre_event_volatility_expected": False,
    }

    if not upcoming_events:
        logger.debug("No upcoming events – returning defaults")
        return features

    # Find the nearest future event and its impact
    best_minutes = _NO_EVENT_MINUTES
    best_score = 0.0
    pre_event = False

    for event in upcoming_events:
        event_time = event.get("time") or event.get("datetime")
        if event_time is None:
            continue

        mins = _minutes_until(event_time)
        score = _score_event(event)

        if mins < best_minutes:
            best_minutes = mins
            best_score = score

        # Pre-event flag: within 2 hours of a high-impact event
        if mins <= _PRE_EVENT_HOURS * 60 and score >= 0.7:
            pre_event = True

    features["next_event_minutes"] = best_minutes
    features["event_impact_score"] = round(best_score, 2)
    features["pre_event_volatility_expected"] = pre_event

    # Auto-blackout: if a score-1.0 event is within 15 minutes
    if best_minutes <= 15 and best_score >= 0.95:
        features["is_blackout"] = True

    logger.info(
        "Calendar features: next_event=%d min, impact=%.2f, pre_event=%s, blackout=%s",
        features["next_event_minutes"],
        features["event_impact_score"],
        features["pre_event_volatility_expected"],
        features["is_blackout"],
    )

    return features
