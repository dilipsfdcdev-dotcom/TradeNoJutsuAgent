"""
Ratchet stop-loss manager for the XAUUSD trading agent.

**This is the most critical execution file.**  It runs as its own async
loop (every 15 seconds by default), completely independent of the signal
loop.  The sole invariant is:

    *The stop-loss only ever moves toward profit.  Never backward.*

Each ratchet level is expressed as a multiple of the initial risk (R).
Once price reaches a level the SL is moved to lock in a defined
percentage of the unrealised profit.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# Type aliases for the callables injected at run-time.
PositionsGetter = Callable[[], Awaitable[list[dict[str, Any]]]]
SLModifier = Callable[[int, float], Awaitable[bool] | bool]
RatchetEventLogger = Callable[[dict[str, Any]], Awaitable[None] | None]
TelegramAlerter = Callable[[str], Awaitable[None] | None]


class RatchetSLManager:
    """Move stop-losses toward profit as the position advances.

    Ratchet table
    -------------
    Each entry is ``(r_multiple, lock_spec)`` where *lock_spec* is either
    the string ``"breakeven"`` (move SL to entry) or a float representing
    the fraction of the current profit-in-pips to lock.

    ===== ===========
    R     Lock
    ===== ===========
    0.50  breakeven
    1.00  30 %
    1.50  50 %
    2.00  65 %
    2.50  75 %
    3.00  85 %
    ===== ===========
    """

    RATCHET: list[tuple[float, str | float]] = [
        (0.50, "breakeven"),
        (1.00, 0.30),
        (1.50, 0.50),
        (2.00, 0.65),
        (2.50, 0.75),
        (3.00, 0.85),
    ]

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        settings = settings or {}
        self.enabled: bool = settings.get("enabled", True)
        self.check_interval: int = settings.get("check_interval_s", 15)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_new_sl(
        self,
        position: dict[str, Any],
        current_price: float,
    ) -> float | None:
        """Determine whether the SL should be ratcheted.

        Args:
            position: Must contain at minimum:
                ``ticket``, ``direction``, ``open_price``, ``current_sl``,
                ``lot_size``, ``initial_risk_usd``, ``pip_value``,
                ``pip_size``.
            current_price: Latest bid (SELL) or ask (BUY).

        Returns:
            The new SL price, or ``None`` if no move is warranted.
        """
        direction: str = position["direction"].upper()
        open_price: float = position["open_price"]
        current_sl: float = position["current_sl"]
        initial_risk_usd: float = position["initial_risk_usd"]
        pip_value: float = position["pip_value"]
        pip_size: float = position["pip_size"]
        lot_size: float = position["lot_size"]

        # ---- 1.  Profit in USD and R ------------------------------------
        if direction == "BUY":
            profit_pips = (current_price - open_price) / pip_size
        else:
            profit_pips = (open_price - current_price) / pip_size

        profit_usd = profit_pips * pip_value * lot_size
        if initial_risk_usd <= 0:
            return None
        profit_r = profit_usd / initial_risk_usd

        # ---- 2.  Find the highest ratchet level reached -----------------
        active_level: tuple[float, str | float] | None = None
        for r_threshold, lock_spec in self.RATCHET:
            if profit_r >= r_threshold:
                active_level = (r_threshold, lock_spec)

        if active_level is None:
            return None

        _, lock_spec = active_level

        # ---- 3.  Compute target SL --------------------------------------
        if lock_spec == "breakeven":
            target_sl = open_price
        else:
            locked_pips = profit_pips * float(lock_spec)
            if direction == "BUY":
                target_sl = open_price + locked_pips * pip_size
            else:
                target_sl = open_price - locked_pips * pip_size

        # ---- 4.  Enforce monotonic direction ----------------------------
        if direction == "BUY":
            if target_sl <= current_sl:
                return None
        else:
            if target_sl >= current_sl:
                return None

        # Round to a sensible precision (5 decimal places covers all FX
        # and metals).
        target_sl = round(target_sl, 5)

        logger.debug(
            "Ratchet ticket=%d profit_r=%.2f lock=%s target_sl=%.5f",
            position["ticket"],
            profit_r,
            lock_spec,
            target_sl,
        )
        return target_sl

    # ------------------------------------------------------------------
    # Async run loop
    # ------------------------------------------------------------------

    async def run_forever(
        self,
        open_positions_getter: PositionsGetter,
        sl_modifier: SLModifier,
        ratchet_event_logger: RatchetEventLogger,
        telegram_alerter: TelegramAlerter | None = None,
    ) -> None:
        """Run the ratchet check indefinitely.

        Args:
            open_positions_getter: Async callable returning enriched
                position dicts (must include ``initial_risk_usd``,
                ``pip_value``, ``pip_size``).
            sl_modifier:           Callable ``(ticket, new_sl) -> bool``.
            ratchet_event_logger:  Callable to persist ratchet events.
            telegram_alerter:      Optional callable to send Telegram
                                   notifications.
        """
        logger.info(
            "RatchetSLManager started (interval=%ds, enabled=%s)",
            self.check_interval,
            self.enabled,
        )

        while True:
            try:
                if self.enabled:
                    await self._tick(
                        open_positions_getter,
                        sl_modifier,
                        ratchet_event_logger,
                        telegram_alerter,
                    )
            except Exception:
                logger.exception("RatchetSLManager tick error")

            await asyncio.sleep(self.check_interval)

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------

    async def _tick(
        self,
        open_positions_getter: PositionsGetter,
        sl_modifier: SLModifier,
        ratchet_event_logger: RatchetEventLogger,
        telegram_alerter: TelegramAlerter | None,
    ) -> None:
        positions = await open_positions_getter()
        if not positions:
            return

        for pos in positions:
            current_price = pos.get("current_price")
            if current_price is None:
                logger.warning(
                    "No current_price for ticket %d — skipping", pos["ticket"]
                )
                continue

            new_sl = self.compute_new_sl(pos, current_price)
            if new_sl is None:
                continue

            # Modify stop-loss via the injected modifier.
            modify_result = sl_modifier(pos["ticket"], new_sl)
            if asyncio.iscoroutine(modify_result):
                modify_result = await modify_result

            if not modify_result:
                logger.error(
                    "Failed to ratchet SL for ticket %d to %.5f",
                    pos["ticket"],
                    new_sl,
                )
                continue

            # Determine the ratchet label for logging / alerting.
            profit_r = self._profit_r(pos, current_price)
            ratchet_label = self._ratchet_label(profit_r)

            event: dict[str, Any] = {
                "ticket": pos["ticket"],
                "direction": pos["direction"],
                "old_sl": pos["current_sl"],
                "new_sl": new_sl,
                "profit_r": profit_r,
                "ratchet_level": ratchet_label,
            }

            log_result = ratchet_event_logger(event)
            if asyncio.iscoroutine(log_result):
                await log_result

            logger.info(
                "Ratchet applied: ticket=%d SL %.5f → %.5f (%s, %.2fR)",
                pos["ticket"],
                pos["current_sl"],
                new_sl,
                ratchet_label,
                profit_r,
            )

            if telegram_alerter is not None:
                alert_msg = self.format_ratchet_alert(
                    pos, new_sl, profit_r, ratchet_label
                )
                alert_result = telegram_alerter(alert_msg)
                if asyncio.iscoroutine(alert_result):
                    await alert_result

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_ratchet_alert(
        self,
        position: dict[str, Any],
        new_sl: float,
        profit_r: float,
        ratchet_level: str,
    ) -> str:
        """Build a human-readable Telegram alert string."""
        direction = position["direction"].upper()
        ticket = position["ticket"]
        old_sl = position["current_sl"]
        symbol = position.get("symbol", "XAUUSD")

        # Compute profit USD for the message.
        pip_size = position.get("pip_size", 0.1)
        pip_value = position.get("pip_value", 1.0)
        lot_size = position.get("lot_size", 0.0)
        open_price = position["open_price"]
        current_price = position.get("current_price", 0.0)

        if direction == "BUY":
            profit_pips = (current_price - open_price) / pip_size
        else:
            profit_pips = (open_price - current_price) / pip_size
        profit_usd = profit_pips * pip_value * lot_size

        tail = ""
        if ratchet_level == "breakeven":
            tail = "You cannot lose on this trade now."
        elif profit_r >= 2.0:
            tail = "Locking in significant profit."

        lines = [
            f"\U0001f512 SL RATCHET \u2014 {symbol} {direction} #{ticket}",
            f"Profit: +${profit_usd:,.0f} ({profit_r:.2f}R)",
            f"SL: ${old_sl:,.2f} \u2192 ${new_sl:,.2f} ({ratchet_level})",
        ]
        if tail:
            lines.append(tail)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profit_r(self, position: dict[str, Any], current_price: float) -> float:
        pip_size = position.get("pip_size", 0.1)
        pip_value = position.get("pip_value", 1.0)
        lot_size = position.get("lot_size", 0.0)
        initial_risk_usd = position.get("initial_risk_usd", 1.0)
        open_price = position["open_price"]
        direction = position["direction"].upper()

        if direction == "BUY":
            profit_pips = (current_price - open_price) / pip_size
        else:
            profit_pips = (open_price - current_price) / pip_size

        profit_usd = profit_pips * pip_value * lot_size
        if initial_risk_usd <= 0:
            return 0.0
        return profit_usd / initial_risk_usd

    def _ratchet_label(self, profit_r: float) -> str:
        """Return the label of the highest ratchet level reached."""
        label = "none"
        for r_threshold, lock_spec in self.RATCHET:
            if profit_r >= r_threshold:
                if lock_spec == "breakeven":
                    label = "breakeven"
                else:
                    label = f"lock_{int(float(lock_spec) * 100)}pct"
        return label
