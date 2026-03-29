"""Structured logging for TradeNoJutsu."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_LOG_DIR = Path("logs")
_initialized = False


def setup_logging(level: str = "INFO") -> None:
    """Initialize logging with rich console and file output."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    _LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger("tradenojutsu")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rich console handler
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(_LOG_DIR / "tradenojutsu.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the tradenojutsu namespace."""
    return logging.getLogger(f"tradenojutsu.{name}")
