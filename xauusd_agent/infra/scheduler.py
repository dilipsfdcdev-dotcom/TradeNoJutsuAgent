"""
Async task scheduler for the XAUUSD trading agent.

Provides a lightweight, timezone-aware scheduler built on ``asyncio``.
Tasks are defined declaratively and fire at their next eligible UTC wall-clock
time without drift.

Pre-configured schedules
------------------------
* **Daily summary** -- every day at 22:00 UTC
* **Weekly LightGBM retrain** -- Sunday 22:00 UTC
* **Monthly PPO fine-tune** -- 1st of each month, 00:00 UTC
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable

from xauusd_agent.infra.logger import get_logger

logger = get_logger("scheduler")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AsyncCallback = Callable[..., Awaitable[None]]


@dataclass
class ScheduledTask:
    """Describes a single recurring task."""

    name: str
    callback: AsyncCallback
    callback_args: tuple[Any, ...] = ()
    callback_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Target wall-clock time (UTC) for the task to fire.
    target_time: time = time(22, 0)
    #: Optional weekday constraint (0=Monday .. 6=Sunday). ``None`` = daily.
    weekday: int | None = None
    #: Optional day-of-month constraint. ``None`` = every day / every week.
    day_of_month: int | None = None
    #: Tracks the asyncio.Task handle.
    _task: asyncio.Task | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# AsyncScheduler
# ---------------------------------------------------------------------------


class AsyncScheduler:
    """Timezone-aware async scheduler.

    Usage::

        scheduler = AsyncScheduler()
        scheduler.add_daily("daily_summary", my_summary_coro, target_time=time(22, 0))
        scheduler.add_weekly("lgbm_retrain", retrain_lgbm, weekday=6, target_time=time(22, 0))
        scheduler.add_monthly("ppo_finetune", finetune_ppo, day=1, target_time=time(0, 0))
        await scheduler.start()
        ...
        await scheduler.stop()
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._running: bool = False

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def add_daily(
        self,
        name: str,
        callback: AsyncCallback,
        *args: Any,
        target_time: time = time(22, 0),
        **kwargs: Any,
    ) -> None:
        """Register a task that fires once per day at *target_time* UTC."""
        self._tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            callback_args=args,
            callback_kwargs=kwargs,
            target_time=target_time,
        )

    def add_weekly(
        self,
        name: str,
        callback: AsyncCallback,
        *args: Any,
        weekday: int = 6,
        target_time: time = time(22, 0),
        **kwargs: Any,
    ) -> None:
        """Register a task that fires once per week on *weekday* at *target_time* UTC.

        *weekday* follows Python convention: 0 = Monday, 6 = Sunday.
        """
        self._tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            callback_args=args,
            callback_kwargs=kwargs,
            target_time=target_time,
            weekday=weekday,
        )

    def add_monthly(
        self,
        name: str,
        callback: AsyncCallback,
        *args: Any,
        day: int = 1,
        target_time: time = time(0, 0),
        **kwargs: Any,
    ) -> None:
        """Register a task that fires on day *day* of each month at *target_time* UTC."""
        self._tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            callback_args=args,
            callback_kwargs=kwargs,
            target_time=target_time,
            day_of_month=day,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all registered scheduled tasks."""
        self._running = True
        for st in self._tasks.values():
            st._task = asyncio.create_task(
                self._loop(st), name=f"sched:{st.name}"
            )
            logger.info(
                "Scheduled task registered",
                extra={
                    "task": st.name,
                    "target_time": str(st.target_time),
                    "weekday": st.weekday,
                    "day_of_month": st.day_of_month,
                },
            )
        logger.info("AsyncScheduler started", extra={"task_count": len(self._tasks)})

    async def stop(self) -> None:
        """Cancel all scheduled tasks gracefully."""
        self._running = False
        for st in self._tasks.values():
            if st._task and not st._task.done():
                st._task.cancel()
                try:
                    await st._task
                except asyncio.CancelledError:
                    pass
        logger.info("AsyncScheduler stopped")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_until(target: time, weekday: int | None, day_of_month: int | None) -> float:
        """Compute seconds from *now* (UTC) until the next occurrence."""
        now = datetime.now(timezone.utc)
        candidate = now.replace(
            hour=target.hour,
            minute=target.minute,
            second=target.second,
            microsecond=0,
        )

        if day_of_month is not None:
            # Monthly: advance to the target day (this month or next)
            try:
                candidate = candidate.replace(day=day_of_month)
            except ValueError:
                # e.g. day 31 in a 30-day month -- roll to next month
                candidate = _first_day_next_month(candidate).replace(
                    hour=target.hour, minute=target.minute, second=0, microsecond=0
                )
            if candidate <= now:
                candidate = _first_day_next_month(candidate).replace(
                    day=day_of_month,
                    hour=target.hour,
                    minute=target.minute,
                    second=0,
                    microsecond=0,
                )
        elif weekday is not None:
            # Weekly: advance to the target weekday
            days_ahead = (weekday - now.weekday()) % 7
            candidate += timedelta(days=days_ahead)
            if candidate <= now:
                candidate += timedelta(days=7)
        else:
            # Daily
            if candidate <= now:
                candidate += timedelta(days=1)

        return (candidate - now).total_seconds()

    async def _loop(self, st: ScheduledTask) -> None:
        """Sleep until the next firing time, run the callback, repeat."""
        while self._running:
            delay = self._seconds_until(st.target_time, st.weekday, st.day_of_month)
            logger.debug(
                "Sleeping until next fire",
                extra={"task": st.name, "delay_s": round(delay, 1)},
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            logger.info("Firing scheduled task", extra={"task": st.name})
            try:
                await st.callback(*st.callback_args, **st.callback_kwargs)
                logger.info(
                    "Scheduled task completed",
                    extra={"task": st.name},
                )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "Scheduled task failed",
                    extra={"task": st.name},
                )

            # Brief cooldown to avoid double-fire within the same second
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                return


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _first_day_next_month(dt: datetime) -> datetime:
    """Return the 1st day of the month following *dt*."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)
