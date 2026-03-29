"""Economic calendar -- fetches high-impact events and enforces blackout windows.

Primary source: ForexFactory RSS feed.
Fallback: a static list of recurring high-impact releases so the agent can
still avoid news even when the feed is unreachable.

All times are UTC.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from tradenojutsu.infra.logger import get_logger

logger = get_logger("data.economic_calendar")

# ---------------------------------------------------------------------------
# High-impact event keywords (used for filtering + static fallback)
# ---------------------------------------------------------------------------
HIGH_IMPACT_EVENTS: list[str] = [
    "Non-Farm Payrolls",
    "NFP",
    "FOMC",
    "Fed Interest Rate Decision",
    "ECB Interest Rate Decision",
    "BOE Interest Rate Decision",
    "BOJ Interest Rate Decision",
    "CPI",
    "Core CPI",
    "GDP",
    "Retail Sales",
    "ISM Manufacturing PMI",
    "ISM Services PMI",
    "Unemployment Rate",
    "Average Hourly Earnings",
    "PCE Price Index",
    "Core PCE",
    "PPI",
    "Trade Balance",
    "Initial Jobless Claims",
    "Consumer Confidence",
    "Michigan Consumer Sentiment",
    "Durable Goods Orders",
    "Housing Starts",
    "Existing Home Sales",
    "New Home Sales",
    "Jackson Hole",
    "BOC Interest Rate Decision",
    "RBA Interest Rate Decision",
    "RBNZ Interest Rate Decision",
    "SNB Interest Rate Decision",
]

# ForexFactory RSS endpoint
_FF_RSS_URL = "https://www.forexfactory.com/ffcal_week_this.xml"

# Cache duration in seconds
_CACHE_TTL = 3600  # 1 hour


@dataclass
class EconomicEvent:
    """A single economic calendar event."""

    title: str
    currency: str
    impact: str  # "high", "medium", "low"
    dt_utc: datetime
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_impact(self) -> bool:
        if self.impact.lower() == "high":
            return True
        return any(kw.lower() in self.title.lower() for kw in HIGH_IMPACT_EVENTS)


class EconomicCalendar:
    """Fetches and caches economic calendar events.

    Usage::

        cal = EconomicCalendar()
        if cal.is_blackout():
            logger.warning("Inside news blackout -- skipping trade entry")

        events = cal.get_upcoming_events(hours_ahead=4)
    """

    def __init__(
        self,
        blackout_minutes_before: int = 30,
        blackout_minutes_after: int = 30,
        cache_ttl: int = _CACHE_TTL,
    ) -> None:
        self.blackout_before = timedelta(minutes=blackout_minutes_before)
        self.blackout_after = timedelta(minutes=blackout_minutes_after)
        self.cache_ttl = cache_ttl

        self._cache: list[EconomicEvent] = []
        self._cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        high_impact_only: bool = False,
    ) -> list[EconomicEvent]:
        """Return events occurring within *hours_ahead* from now.

        Parameters
        ----------
        hours_ahead : int
            Look-ahead window in hours.
        high_impact_only : bool
            If ``True``, return only high-impact events.
        """
        events = self._get_events()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = [e for e in events if now <= e.dt_utc <= cutoff]
        if high_impact_only:
            upcoming = [e for e in upcoming if e.is_high_impact]

        return sorted(upcoming, key=lambda e: e.dt_utc)

    def is_blackout(self, now: datetime | None = None) -> bool:
        """Check if we are inside a news-blackout window.

        A blackout is active when a high-impact event is within
        ``blackout_minutes_before`` in the future or
        ``blackout_minutes_after`` in the past.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        events = self._get_events()
        for event in events:
            if not event.is_high_impact:
                continue
            window_start = event.dt_utc - self.blackout_before
            window_end = event.dt_utc + self.blackout_after
            if window_start <= now <= window_end:
                logger.info(
                    "Blackout active: '%s' at %s (window %s -> %s)",
                    event.title,
                    event.dt_utc.isoformat(),
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Data fetching (RSS + static fallback)
    # ------------------------------------------------------------------

    def _get_events(self) -> list[EconomicEvent]:
        """Return cached events, refreshing if stale."""
        if self._cache and (time.time() - self._cache_time) < self.cache_ttl:
            return self._cache

        events = self._fetch_rss()
        if not events:
            logger.warning("RSS feed unavailable -- using static fallback calendar")
            events = self._static_fallback()

        self._cache = events
        self._cache_time = time.time()
        logger.info("Economic calendar refreshed: %d events cached", len(events))
        return events

    def _fetch_rss(self) -> list[EconomicEvent]:
        """Parse the ForexFactory weekly RSS feed."""
        try:
            resp = requests.get(_FF_RSS_URL, timeout=10, headers={
                "User-Agent": "TradeNoJutsuAgent/1.0",
            })
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to fetch ForexFactory RSS: %s", exc)
            return []

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse RSS XML: %s", exc)
            return []

        events: list[EconomicEvent] = []
        ns = {"ff": "https://www.forexfactory.com"}

        for item in root.iter("event"):
            title = self._xml_text(item, "title", ns)
            currency = self._xml_text(item, "country", ns)
            impact = self._xml_text(item, "impact", ns).lower()
            date_str = self._xml_text(item, "date", ns)
            time_str = self._xml_text(item, "time", ns)

            dt_utc = self._parse_ff_datetime(date_str, time_str)
            if dt_utc is None:
                continue

            events.append(
                EconomicEvent(
                    title=title,
                    currency=currency,
                    impact=impact,
                    dt_utc=dt_utc,
                    actual=self._xml_text(item, "actual", ns) or None,
                    forecast=self._xml_text(item, "forecast", ns) or None,
                    previous=self._xml_text(item, "previous", ns) or None,
                )
            )

        return events

    @staticmethod
    def _xml_text(parent: ET.Element, tag: str, ns: dict[str, str]) -> str:
        """Safely extract text from an XML element."""
        # Try with namespace, then without
        for prefix in [f"ff:{tag}", tag]:
            el = parent.find(prefix, ns)
            if el is not None and el.text:
                return el.text.strip()
        return ""

    @staticmethod
    def _parse_ff_datetime(date_str: str, time_str: str) -> datetime | None:
        """Parse ForexFactory date/time strings into a UTC datetime."""
        if not date_str:
            return None
        # Common formats: "01-03-2026", time "8:30am" or "Tentative"
        for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return None

        if time_str and time_str.lower() not in ("tentative", "all day", ""):
            time_str = time_str.strip().lower()
            for tfmt in ("%I:%M%p", "%H:%M"):
                try:
                    t = datetime.strptime(time_str, tfmt)
                    dt = dt.replace(hour=t.hour, minute=t.minute)
                    break
                except ValueError:
                    continue

        return dt

    def _static_fallback(self) -> list[EconomicEvent]:
        """Generate a minimal static calendar for the current week.

        This ensures the agent has *something* to work with even when the
        RSS feed is unreachable.  Covers recurring US releases.
        """
        now = datetime.now(timezone.utc)
        monday = now - timedelta(days=now.weekday())
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Typical weekly schedule (day_offset, hour_utc, minute, title, currency)
        static_schedule: list[tuple[int, int, int, str, str]] = [
            # Tuesday
            (1, 15, 0, "Consumer Confidence", "USD"),
            # Wednesday
            (2, 13, 15, "ADP Non-Farm Employment", "USD"),
            (2, 15, 0, "ISM Manufacturing PMI", "USD"),
            (2, 19, 0, "FOMC Statement", "USD"),
            # Thursday
            (3, 13, 30, "Initial Jobless Claims", "USD"),
            (3, 13, 30, "Trade Balance", "USD"),
            # Friday
            (4, 13, 30, "Non-Farm Payrolls", "USD"),
            (4, 13, 30, "Unemployment Rate", "USD"),
            (4, 13, 30, "Average Hourly Earnings", "USD"),
        ]

        events: list[EconomicEvent] = []
        for day_off, hour, minute, title, currency in static_schedule:
            dt = monday + timedelta(days=day_off)
            dt = dt.replace(hour=hour, minute=minute)
            events.append(
                EconomicEvent(
                    title=title,
                    currency=currency,
                    impact="high",
                    dt_utc=dt,
                    metadata={"source": "static_fallback"},
                )
            )

        return events
