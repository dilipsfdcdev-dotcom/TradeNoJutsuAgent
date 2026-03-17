"""
Structured logging infrastructure for the XAUUSD trading agent.

Provides JSON-formatted file logging with rotation and human-readable
console output. All modules should obtain loggers via ``get_logger``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_FILE = LOG_DIR / "xauusd_agent.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }

        # Merge any extra fields passed via ``extra={...}``
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "relativeCreated",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "filename",
                "module",
                "levelname",
                "levelno",
                "pathname",
                "process",
                "processName",
                "thread",
                "threadName",
                "msecs",
                "message",
                "taskName",
            }:
                payload[key] = value

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exception"] = record.exc_text

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Human-readable console formatter
# ---------------------------------------------------------------------------


class ConsoleFormatter(logging.Formatter):
    """Colourless, human-readable formatter for stderr."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_initialised: bool = False


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_root_logger() -> None:
    """Attach console + rotating-file handlers to the root ``xauusd`` logger.

    Called once on first ``get_logger`` invocation.
    """
    global _initialised
    if _initialised:
        return

    _ensure_log_dir()

    root = logging.getLogger("xauusd")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    root.propagate = False

    # Console handler -- human-readable
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    root.addHandler(console_handler)

    # Rotating file handler -- JSON
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    _initialised = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``xauusd`` namespace.

    Usage::

        from xauusd_agent.infra.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Signal generated", extra={"action": "BUY", "score": 0.87})
    """
    _setup_root_logger()
    return logging.getLogger(f"xauusd.{name}")
