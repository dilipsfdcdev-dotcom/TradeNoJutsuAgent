"""High-impact economic event tracker."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import httpx
import structlog
from redis.asyncio import Redis

from agent.config import settings  # noqa: F401 – available for future config

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

HIGH_IMPACT_EVENTS: set[str] = {
    "NFP",
    "Non-Farm Payrolls",
    "Nonfarm Payrolls",
    "CPI",
    "Consumer Price Index",
    "FOMC",
    "Federal Funds Rate",
    "Interest Rate Decision",
    "GDP",
    "Gross Domestic Product",
    "Employment Change",
    "Unemployment Rate",
    "Retail Sales",
    "PMI",
    "Manufacturing PMI",
    "Services PMI",
}

CALENDAR_CACHE_KEY = "economic:calendar"
CALENDAR_TTL_SECONDS = 3600  # 1 hour


@dataclass
class EconomicEvent:
    name: str
    currency: str
    impact: str  # high / medium / low
    datetime_utc: datetime
    forecast: str | None
    previous: str | None
    actual: str | None

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datetime_utc"] = self.datetime_utc.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EconomicEvent:
        d = dict(d)  # shallow copy
        dt_raw = d["datetime_utc"]
        if isinstance(dt_raw, str):
            d["datetime_utc"] = datetime.fromisoformat(dt_raw)
        return cls(**d)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


async def _fetch_from_api() -> list[dict]:
    """Fetch raw calendar data from a free Forex-Factory-style JSON feed."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_CALENDAR_URL)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("calendar.fetch_http_error", status=exc.response.status_code)
        return []
    except httpx.RequestError as exc:
        log.error("calendar.fetch_request_error", error=str(exc))
        return []


def _parse_event(raw: dict) -> EconomicEvent | None:
    """Convert a raw JSON entry into an EconomicEvent (or None on failure)."""

    try:
        dt_str = raw.get("date", "")
        dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

    name = raw.get("title", raw.get("event", ""))
    impact = raw.get("impact", "low").lower()
    currency = raw.get("country", raw.get("currency", "USD")).upper()

    return EconomicEvent(
        name=name,
        currency=currency,
        impact=impact if impact in {"high", "medium", "low"} else "low",
        datetime_utc=dt,
        forecast=raw.get("forecast") or None,
        previous=raw.get("previous") or None,
        actual=raw.get("actual") or None,
    )


async def fetch_calendar(redis_client: Redis) -> list[EconomicEvent]:
    """Fetch the daily economic calendar, cache in Redis, and return events."""

    raw_items = await _fetch_from_api()
    events = [e for raw in raw_items if (e := _parse_event(raw)) is not None]

    log.info("calendar.fetched", total_events=len(events))

    # Cache in Redis
    if events:
        payload = json.dumps([ev.to_dict() for ev in events])
        await redis_client.set(CALENDAR_CACHE_KEY, payload, ex=CALENDAR_TTL_SECONDS)
        log.debug("calendar.cached", count=len(events))

    return events


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def _load_cached(redis_client: Redis) -> list[EconomicEvent]:
    """Load events from Redis cache (empty list on miss)."""

    raw: bytes | None = await redis_client.get(CALENDAR_CACHE_KEY)
    if raw is None:
        return []
    items = json.loads(raw)
    return [EconomicEvent.from_dict(d) for d in items]


async def get_upcoming_events(
    redis_client: Redis,
    hours: int = 4,
) -> list[EconomicEvent]:
    """Return events happening within the next *hours* hours."""

    events = await _load_cached(redis_client)
    if not events:
        events = await fetch_calendar(redis_client)

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=hours)

    upcoming = [ev for ev in events if now <= ev.datetime_utc <= horizon]
    log.debug("calendar.upcoming", hours=hours, count=len(upcoming))
    return upcoming


# ---------------------------------------------------------------------------
# High-impact window check
# ---------------------------------------------------------------------------


def is_high_impact_window(
    events: list[EconomicEvent],
    minutes_buffer: int = 15,
) -> bool:
    """Return True if any high-impact event is within *minutes_buffer* minutes.

    Checks both before and after the current time so callers can avoid
    entering trades right before or during a major release.
    """

    now = datetime.now(timezone.utc)
    buffer = timedelta(minutes=minutes_buffer)

    for ev in events:
        if ev.impact != "high":
            continue

        # Check against known high-impact event names as an extra filter
        name_upper = ev.name.upper()
        is_known = any(hi.upper() in name_upper for hi in HIGH_IMPACT_EVENTS)
        if not is_known:
            continue

        if (ev.datetime_utc - buffer) <= now <= (ev.datetime_utc + buffer):
            log.warning(
                "calendar.high_impact_window",
                event=ev.name,
                event_time=ev.datetime_utc.isoformat(),
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Refresh loop
# ---------------------------------------------------------------------------


async def calendar_refresh_loop(redis_client: Redis) -> None:
    """Refresh the economic calendar cache every hour."""

    log.info("calendar.refresh_loop_started")

    while True:
        try:
            await fetch_calendar(redis_client)
        except Exception:
            log.exception("calendar.refresh_error")

        await asyncio.sleep(3600)  # 1 hour
