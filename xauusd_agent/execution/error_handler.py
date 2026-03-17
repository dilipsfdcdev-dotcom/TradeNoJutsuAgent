"""
Execution error handling, retry logic, and slippage checks.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# MT5 return codes → human-readable descriptions.
_RETCODE_MAP: dict[int, str] = {
    10004: "Requote — price changed during submission",
    10006: "Request rejected by server",
    10007: "Request cancelled by trader",
    10008: "Order placed (pending, not yet filled)",
    10009: "Order executed successfully",
    10010: "Order executed partially",
    10011: "Request processing error",
    10012: "Request timed out",
    10013: "Invalid request parameters",
    10014: "Invalid volume",
    10015: "Invalid price",
    10016: "Invalid stops (SL/TP)",
    10017: "Trade is disabled",
    10018: "Market is closed",
    10019: "Insufficient margin",
    10020: "Prices changed — requote",
    10021: "No quotes available",
    10022: "Invalid order expiration date",
    10023: "Order state changed",
    10024: "Too many trade requests",
    10025: "No changes in request — modification identical to current state",
    10026: "Autotrading disabled by server",
    10027: "Autotrading disabled by client terminal",
    10028: "Request locked for processing",
    10029: "Order or position frozen",
    10030: "Invalid order filling type",
    10031: "No connection to trade server",
    10032: "Operation allowed only for live accounts",
    10033: "Pending orders limit reached",
    10034: "Volume limit for symbol/position reached",
    10035: "Invalid or prohibited order type",
    10036: "Position with specified ticket already closed",
    10038: "Close order already exists for this position",
    10039: "Maximum number of open positions reached",
    10040: "Pending order activation rejected; order cancelled",
    10041: "Request rejected — Only long positions allowed",
    10042: "Request rejected — Only short positions allowed",
    10043: "Request rejected — Only position close allowed",
    10044: "Position close not allowed by FIFO rule",
}


class ExecutionErrorHandler:
    """Utilities for resilient order execution."""

    MAX_RETRIES: int = 3
    RETRY_DELAYS: list[float] = [1, 2, 5]

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    @staticmethod
    async def with_retry(
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* with automatic retries.

        If *func* is a coroutine function it will be awaited; otherwise
        it is called synchronously.

        Raises the last exception after ``MAX_RETRIES`` failures.
        """
        last_exc: Exception | None = None

        for attempt in range(ExecutionErrorHandler.MAX_RETRIES):
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as exc:
                last_exc = exc
                delay = ExecutionErrorHandler.RETRY_DELAYS[
                    min(attempt, len(ExecutionErrorHandler.RETRY_DELAYS) - 1)
                ]
                logger.warning(
                    "Retry %d/%d for %s after error: %s (next delay=%.1fs)",
                    attempt + 1,
                    ExecutionErrorHandler.MAX_RETRIES,
                    getattr(func, "__name__", str(func)),
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        assert last_exc is not None
        logger.error(
            "All %d retries exhausted for %s",
            ExecutionErrorHandler.MAX_RETRIES,
            getattr(func, "__name__", str(func)),
        )
        raise last_exc

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------

    @staticmethod
    def check_slippage(
        requested_price: float,
        filled_price: float,
        max_slippage_pips: float = 3.0,
        pip_size: float = 0.1,
    ) -> bool:
        """Return ``True`` if slippage is within the acceptable range.

        Args:
            requested_price:   The price at which the order was submitted.
            filled_price:      The price at which the order was filled.
            max_slippage_pips: Maximum acceptable slippage in pips.
            pip_size:          Price movement per pip.

        Returns:
            ``True`` if ``|requested - filled| <= max_slippage_pips * pip_size``.
        """
        slippage = abs(requested_price - filled_price)
        slippage_pips = slippage / pip_size if pip_size > 0 else float("inf")
        acceptable = slippage_pips <= max_slippage_pips

        if not acceptable:
            logger.warning(
                "Slippage exceeded: requested=%.5f filled=%.5f "
                "slippage=%.2f pips (max=%.2f)",
                requested_price,
                filled_price,
                slippage_pips,
                max_slippage_pips,
            )

        return acceptable

    # ------------------------------------------------------------------
    # Retcode mapping
    # ------------------------------------------------------------------

    @staticmethod
    def handle_order_error(retcode: int) -> str:
        """Map an MT5 return code to a human-readable message."""
        return _RETCODE_MAP.get(retcode, f"Unknown MT5 retcode: {retcode}")
