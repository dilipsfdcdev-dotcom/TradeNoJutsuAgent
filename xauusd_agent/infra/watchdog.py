"""
Watchdog manager for the XAUUSD trading agent.

Monitors critical async loops (signal generation, ratchet SL management) by
tracking heartbeat timestamps.  If a loop fails to heartbeat within its
configured timeout the watchdog restarts the coroutine and logs the event.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from xauusd_agent.infra.logger import get_logger

logger = get_logger("watchdog")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CoroutineFactory = Callable[..., Awaitable[None]]


@dataclass
class _WatchedLoop:
    """Internal bookkeeping for a single monitored coroutine."""

    name: str
    factory: CoroutineFactory
    factory_args: tuple[Any, ...] = ()
    factory_kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 60.0
    last_heartbeat: float = field(default_factory=time.monotonic)
    task: asyncio.Task | None = None
    restart_count: int = 0


# ---------------------------------------------------------------------------
# WatchdogManager
# ---------------------------------------------------------------------------


class WatchdogManager:
    """Supervises long-running asyncio coroutines via heartbeats.

    Usage::

        wd = WatchdogManager()
        wd.register("signal_loop", signal_loop_coro, timeout_s=60)
        wd.register("ratchet_loop", ratchet_loop_coro, timeout_s=30)
        await wd.start()       # begins the monitor loop
        ...
        await wd.stop()        # graceful shutdown

    Inside each monitored coroutine call ``wd.heartbeat("signal_loop")``
    on every iteration to reset the deadline.
    """

    CHECK_INTERVAL: float = 5.0  # seconds between health checks

    def __init__(self) -> None:
        self._loops: dict[str, _WatchedLoop] = {}
        self._monitor_task: asyncio.Task | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        factory: CoroutineFactory,
        *args: Any,
        timeout_s: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Register a coroutine factory to be monitored.

        Parameters
        ----------
        name : str
            Unique human-readable label (e.g. ``"signal_loop"``).
        factory : callable
            An async function (not a coroutine instance) that will be called
            to create/restart the loop.
        timeout_s : float
            Maximum seconds between heartbeats before the loop is restarted.
        """
        self._loops[name] = _WatchedLoop(
            name=name,
            factory=factory,
            factory_args=args,
            factory_kwargs=kwargs,
            timeout_s=timeout_s,
        )
        logger.info(
            "Registered watched loop",
            extra={"loop": name, "timeout_s": timeout_s},
        )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self, name: str) -> None:
        """Record a heartbeat for the named loop."""
        if name in self._loops:
            self._loops[name].last_heartbeat = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all registered loops and begin the monitoring task."""
        self._running = True
        for wl in self._loops.values():
            wl.last_heartbeat = time.monotonic()
            wl.task = asyncio.create_task(
                self._run_guarded(wl), name=f"wd:{wl.name}"
            )
        self._monitor_task = asyncio.create_task(
            self._monitor(), name="wd:monitor"
        )
        logger.info("WatchdogManager started")

    async def stop(self) -> None:
        """Cancel all watched loops and the monitor task."""
        self._running = False

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        for wl in self._loops.values():
            if wl.task and not wl.task.done():
                wl.task.cancel()
                try:
                    await wl.task
                except asyncio.CancelledError:
                    pass

        logger.info("WatchdogManager stopped")

    # ------------------------------------------------------------------
    # Health status
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return a health-check dict suitable for an HTTP endpoint.

        Example response::

            {
                "status": "ok",
                "loops": {
                    "signal_loop": {"alive": True, "restarts": 0, "age_s": 4.2},
                    "ratchet_loop": {"alive": True, "restarts": 1, "age_s": 1.1}
                }
            }
        """
        now = time.monotonic()
        loops_info: dict[str, Any] = {}
        all_ok = True

        for name, wl in self._loops.items():
            age = now - wl.last_heartbeat
            alive = age <= wl.timeout_s
            if not alive:
                all_ok = False
            loops_info[name] = {
                "alive": alive,
                "restarts": wl.restart_count,
                "age_s": round(age, 2),
                "timeout_s": wl.timeout_s,
            }

        return {
            "status": "ok" if all_ok else "degraded",
            "loops": loops_info,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_guarded(self, wl: _WatchedLoop) -> None:
        """Run the coroutine factory and catch unexpected exits."""
        try:
            await wl.factory(*wl.factory_args, **wl.factory_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Watched loop crashed",
                extra={"loop": wl.name},
            )

    async def _restart(self, wl: _WatchedLoop) -> None:
        """Cancel and restart a watched loop."""
        wl.restart_count += 1
        logger.warning(
            "Restarting loop due to missed heartbeat",
            extra={
                "loop": wl.name,
                "restart_count": wl.restart_count,
                "timeout_s": wl.timeout_s,
            },
        )

        # Cancel the stale task
        if wl.task and not wl.task.done():
            wl.task.cancel()
            try:
                await wl.task
            except asyncio.CancelledError:
                pass

        # Re-launch
        wl.last_heartbeat = time.monotonic()
        wl.task = asyncio.create_task(
            self._run_guarded(wl), name=f"wd:{wl.name}"
        )

    async def _monitor(self) -> None:
        """Periodically check heartbeats and restart stale loops."""
        while self._running:
            try:
                await asyncio.sleep(self.CHECK_INTERVAL)
            except asyncio.CancelledError:
                return

            now = time.monotonic()
            for wl in self._loops.values():
                elapsed = now - wl.last_heartbeat
                if elapsed > wl.timeout_s:
                    await self._restart(wl)
                # Also restart if the task has exited unexpectedly
                elif wl.task is not None and wl.task.done():
                    logger.warning(
                        "Watched loop task exited unexpectedly",
                        extra={"loop": wl.name},
                    )
                    await self._restart(wl)
