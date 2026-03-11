"""Logging configuration for flexium.

Provides consistent logging across all flexium modules with
configurable log levels and formatting.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}

# Default format for flexium logs
DEFAULT_FORMAT = "[flexium] %(message)s"
DEBUG_FORMAT = "[flexium] %(levelname)s %(name)s: %(message)s"


def _get_level_from_env() -> Optional[int]:
    """Get logging level from FLEXIUM_LOG_LEVEL environment variable.

    Returns:
        Logging level or None if not set.
    """
    env_level = os.environ.get("FLEXIUM_LOG_LEVEL", "").upper()
    if env_level:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(env_level)
    return None


def setup_logging(
    level: int = logging.INFO,
    debug: bool = False,
    stream: Optional[object] = None,
) -> None:
    """Configure logging for flexium.

    Parameters:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
               Can be overridden by FLEXIUM_LOG_LEVEL env var.
        debug: If True, use verbose format with logger names.
        stream: Output stream for logs. Defaults to sys.stderr.
    """
    # Environment variable overrides parameter
    env_level = _get_level_from_env()
    if env_level is not None:
        level = env_level
        if level == logging.DEBUG:
            debug = True

    if stream is None:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(DEBUG_FORMAT if debug else DEFAULT_FORMAT)
    handler.setFormatter(formatter)

    # Configure root flexium logger
    root_logger = logging.getLogger("flexium")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a flexium module.

    Parameters:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    if name not in _loggers:
        # Ensure name is under flexium namespace
        if not name.startswith("flexium"):
            name = f"flexium.{name}"

        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]
