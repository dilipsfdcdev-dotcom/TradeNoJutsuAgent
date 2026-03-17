"""
Economic-calendar news blackout filter.

Prevents the agent from opening new positions within a configurable window
around high-impact economic events (e.g., NFP, FOMC, CPI).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class NewsFilter:
    """Gate that blocks trading around high-impact news releases."""

    def __init__(self, settings: dict) -> None:
        self.blackout_before: int = settings.get("blackout_before_min", 30)
        self.blackout_after: int = settings.get("blackout_after_min", 30)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def is_blackout(self, upcoming_events: list[dict]) -> bool:
        """
        Return ``True`` if any high-impact event falls within the blackout
        window (``blackout_before`` minutes ahead **or** ``blackout_after``
        minutes behind the current UTC time).

        Each event dict is expected to carry at least:

        * ``"time"``   – ISO-8601 string **or** :class:`datetime` object
        * ``"impact"`` – ``"high"`` (case-insensitive).  Events without this
          key or with lower impact are ignored.
        """
        return self.get_active_blackout_event(upcoming_events) is not None

    def get_active_blackout_event(
        self, upcoming_events: list[dict]
    ) -> dict | None:
        """
        Return the first event that is currently causing a blackout,
        or ``None`` if trading is clear.
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=self.blackout_after)
        window_end = now + timedelta(minutes=self.blackout_before)

        for event in upcoming_events:
            # Only high-impact events trigger a blackout
            impact = str(event.get("impact", "")).strip().lower()
            if impact != "high":
                continue

            event_time = self._parse_time(event)
            if event_time is None:
                continue

            if window_start <= event_time <= window_end:
                logger.info(
                    "News blackout active",
                    extra={
                        "event": event.get("title", "unknown"),
                        "event_time": event_time.isoformat(),
                    },
                )
                return event

        return None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_time(event: dict) -> datetime | None:
        """Extract a timezone-aware datetime from the event dict."""
        raw = event.get("time")
        if raw is None:
            return None

        if isinstance(raw, datetime):
            # Ensure tz-aware
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)

        if isinstance(raw, str):
            # Try common ISO formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
            ):
                try:
                    dt = datetime.strptime(raw, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

        logger.debug("Could not parse event time", extra={"raw": raw})
        return None
