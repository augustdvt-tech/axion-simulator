"""
Axion AI - Structured logging module.

Two formats:
  pretty  — human-readable coloured output for local development
  json    — one JSON object per line for log aggregators (Loki, CloudWatch, etc.)

Configuration via env vars:
  AXION_LOG_LEVEL   — DEBUG / INFO / WARNING / ERROR  (default: INFO)
  AXION_LOG_FORMAT  — pretty / json                   (default: pretty)

Usage:
    from axion_logging import get_logger
    logger = get_logger(__name__)
    logger.info("Scenario loaded", extra={"scenario": name, "samples": n})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Standard LogRecord instance attributes to exclude from "extra" display
# ---------------------------------------------------------------------------

_LOG_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


# ---------------------------------------------------------------------------
# Colour helpers (pretty format only — stripped when not a TTY)
# ---------------------------------------------------------------------------

_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # amber
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _PrettyFormatter(logging.Formatter):
    _use_colour: bool

    def __init__(self, use_colour: bool = True):
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        colour = _COLOURS.get(level, "") if self._use_colour else ""
        reset  = _RESET if self._use_colour else ""

        # Base line
        line = f"{colour}{level:<8}{reset} {record.name}  {record.getMessage()}"

        # Append structured extra fields (skip standard LogRecord attrs)
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in _LOG_RECORD_ATTRS}
        if extras:
            kv = "  ".join(f"{k}={v}" for k, v in extras.items())
            line += f"  [{kv}]"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Merge structured extra fields
        for k, v in record.__dict__.items():
            if k not in _LOG_RECORD_ATTRS:
                payload[k] = v

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured = False


def setup_logging(
    level: str | None = None,
    fmt: str | None = None,
) -> None:
    """
    Configure the root logger.  Safe to call multiple times — subsequent calls
    are no-ops unless force=True is added in the future.

    Args:
        level: log level string (DEBUG/INFO/WARNING/ERROR). Falls back to
               AXION_LOG_LEVEL env var, then INFO.
        fmt:   "pretty" or "json". Falls back to AXION_LOG_FORMAT env var,
               then "pretty".
    """
    global _configured
    if _configured:
        return

    level = (level or os.environ.get("AXION_LOG_LEVEL", "INFO")).upper()
    fmt   = (fmt   or os.environ.get("AXION_LOG_FORMAT", "pretty")).lower()

    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        use_colour = sys.stdout.isatty()
        handler.setFormatter(_PrettyFormatter(use_colour=use_colour))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Calls setup_logging() with env-var defaults if
    the root logger has not been configured yet.
    """
    setup_logging()
    return logging.getLogger(name)
