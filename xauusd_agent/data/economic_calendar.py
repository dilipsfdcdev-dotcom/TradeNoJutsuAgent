"""
Economic calendar event detection.

Provides awareness of upcoming high-impact events so the agent can
avoid trading during volatile blackout windows (e.g. FOMC, NFP, CPI).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# High-impact event keywords
# ---------------------------------------------------------------------------

HIGH_IMPACT_EVENTS: list[str] = [
    "FOMC",
    "NFP",
    "CPI",
    "PPI",
    "Fed Chair",
    "ECB Rate",
    "GDP",
    "Core PCE",
    "Unemployment",
]

# ---------------------------------------------------------------------------
# Static fallback calendar (manually maintained)
# ---------------------------------------------------------------------------

# Each entry: (weekday 0=Mon, hour UTC, event_name)
# Represents recurring monthly/weekly slots for the most common releases.
_STATIC_EVENTS: list[dict] = [
    # US NFP -- first Friday of the month, 13:30 UTC
    {"name": "NFP", "weekday": 4, "hour": 13, "minute": 30, "recurring": "monthly_first"},
    # US CPI -- ~10th-14th of month, 13:30 UTC (Tuesday/Wednesday)
    {"name": "CPI", "weekday": None, "hour": 13, "minute": 30, "recurring": "monthly_mid"},
    # FOMC -- 8 times/year, 19:00 UTC (Wednesday)
    {"name": "FOMC", "weekday": 2, "hour": 19, "minute": 0, "recurring": "fomc"},
    # Core PCE -- last Friday of month, 13:30 UTC
    {"name": "Core PCE", "weekday": 4, "hour": 13, "minute": 30, "recurring": "monthly_last"},
    # US GDP -- end of month, 13:30 UTC
    {"name": "GDP", "weekday": None, "hour": 13, "minute": 30, "recurring": "monthly_end"},
    # PPI -- ~14th-16th of month, 13:30 UTC
    {"name": "PPI", "weekday": None, "hour": 13, "minute": 30, "recurring": "monthly_mid"},
    # Unemployment Claims -- every Thursday 13:30 UTC
    {"name": "Unemployment", "weekday": 3, "hour": 13, "minute": 30, "recurring": "weekly"},
]

_FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class EconomicCalendar:
    """Detects upcoming high-impact economic events and blackout windows."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._cache_ts: float = 0.0
        self._cache_ttl: int = 3600  # 1 hour

    # ------------------------------------------------------------------
    # Calendar loading
    # ------------------------------------------------------------------

    def load_calendar(self) -> None:
        """Fetch this week's calendar from ForexFactory (via proxy API).

        Falls back to a static list of recurring high-impact events if the
        remote source is unavailable.
        """
        if self._events and (time.time() - self._cache_ts) < self._cache_ttl:
            logger.debug("Calendar cache still valid")
            return

        try:
            resp = requests.get(_FOREX_FACTORY_URL, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            self._events = self._parse_remote_events(raw)
            self._cache_ts = time.time()
            logger.info(
                "Loaded %d events from remote calendar", len(self._events)
            )
        except Exception:
            logger.warning(
                "Remote calendar unavailable, using static fallback",
                exc_info=True,
            )
            self._events = self._generate_static_events()
            self._cache_ts = time.time()

    @staticmethod
    def _parse_remote_events(raw: list[dict]) -> list[dict]:
        """Parse ForexFactory JSON into normalised event dicts."""
        events: list[dict] = []
        for item in raw:
            title = item.get("title", "")
            impact = item.get("impact", "").lower()
            date_str = item.get("date", "")

            # Only keep high-impact events.
            if impact not in ("high", "red"):
                is_known = any(
                    kw.lower() in title.lower() for kw in HIGH_IMPACT_EVENTS
                )
                if not is_known:
                    continue

            try:
                event_time = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            events.append(
                {
                    "name": title,
                    "time": event_time,
                    "impact": impact or "high",
                    "currency": item.get("country", "USD"),
                }
            )

        return events

    @staticmethod
    def _generate_static_events() -> list[dict]:
        """Generate approximate event times for the current week from
        the static fallback table."""
        now = datetime.now(tz=timezone.utc)
        week_start = now - timedelta(days=now.weekday())  # Monday 00:00
        events: list[dict] = []

        for tmpl in _STATIC_EVENTS:
            if tmpl["recurring"] == "weekly":
                # Every week on the given weekday.
                event_dt = week_start.replace(
                    hour=tmpl["hour"], minute=tmpl["minute"], second=0, microsecond=0
                ) + timedelta(days=tmpl["weekday"])
                events.append(
                    {
                        "name": tmpl["name"],
                        "time": event_dt,
                        "impact": "high",
                        "currency": "USD",
                    }
                )
            elif tmpl.get("weekday") is not None:
                # Specific weekday this week (rough approximation).
                event_dt = week_start.replace(
                    hour=tmpl["hour"], minute=tmpl["minute"], second=0, microsecond=0
                ) + timedelta(days=tmpl["weekday"])
                events.append(
                    {
                        "name": tmpl["name"],
                        "time": event_dt,
                        "impact": "high",
                        "currency": "USD",
                    }
                )

        return events

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_upcoming_events(
        self, hours_ahead: int = 2
    ) -> list[dict]:
        """Return high-impact events occurring within *hours_ahead* hours.

        Automatically refreshes the calendar if the cache has expired.
        """
        self.load_calendar()

        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = [
            evt for evt in self._events if now <= evt["time"] <= cutoff
        ]

        if upcoming:
            logger.info(
                "%d upcoming events in next %dh: %s",
                len(upcoming),
                hours_ahead,
                [e["name"] for e in upcoming],
            )
        return upcoming

    def is_blackout(
        self,
        blackout_before_min: int = 30,
        blackout_after_min: int = 30,
    ) -> bool:
        """Return *True* if we are within a blackout window of any
        high-impact event.

        Parameters
        ----------
        blackout_before_min:
            Minutes before the event to start the blackout.
        blackout_after_min:
            Minutes after the event to end the blackout.
        """
        self.load_calendar()

        now = datetime.now(tz=timezone.utc)

        for evt in self._events:
            window_start = evt["time"] - timedelta(minutes=blackout_before_min)
            window_end = evt["time"] + timedelta(minutes=blackout_after_min)
            if window_start <= now <= window_end:
                logger.warning(
                    "BLACKOUT active for '%s' at %s (window %s - %s)",
                    evt["name"],
                    evt["time"].isoformat(),
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
                return True

        return False
