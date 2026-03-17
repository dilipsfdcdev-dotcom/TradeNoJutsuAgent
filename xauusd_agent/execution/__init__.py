"""Execution layer for the XAUUSD trading agent."""

from xauusd_agent.execution.error_handler import ExecutionErrorHandler
from xauusd_agent.execution.lot_calculator import calculate_lot, compute_sl_tp
from xauusd_agent.execution.mt5_connector import MT5Connector
from xauusd_agent.execution.mt5_executor import MT5Executor
from xauusd_agent.execution.position_tracker import PositionTracker
from xauusd_agent.execution.sl_manager import RatchetSLManager

__all__ = [
    "ExecutionErrorHandler",
    "MT5Connector",
    "MT5Executor",
    "PositionTracker",
    "RatchetSLManager",
    "calculate_lot",
    "compute_sl_tp",
]
