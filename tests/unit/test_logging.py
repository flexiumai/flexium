"""Tests for the logging utilities module."""

from __future__ import annotations

import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest


class TestGetLevelFromEnv:
    """Tests for _get_level_from_env function."""

    def test_returns_none_when_not_set(self) -> None:
        """Test returns None when FLEXIUM_LOG_LEVEL is not set."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {}, clear=True):
            result = _get_level_from_env()

        assert result is None

    def test_returns_debug_level(self) -> None:
        """Test returns DEBUG level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "DEBUG"}):
            result = _get_level_from_env()

        assert result == logging.DEBUG

    def test_returns_info_level(self) -> None:
        """Test returns INFO level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "INFO"}):
            result = _get_level_from_env()

        assert result == logging.INFO

    def test_returns_warning_level(self) -> None:
        """Test returns WARNING level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "WARNING"}):
            result = _get_level_from_env()

        assert result == logging.WARNING

    def test_returns_error_level(self) -> None:
        """Test returns ERROR level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "ERROR"}):
            result = _get_level_from_env()

        assert result == logging.ERROR

    def test_returns_critical_level(self) -> None:
        """Test returns CRITICAL level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "CRITICAL"}):
            result = _get_level_from_env()

        assert result == logging.CRITICAL

    def test_case_insensitive(self) -> None:
        """Test level matching is case insensitive."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "debug"}):
            result = _get_level_from_env()

        assert result == logging.DEBUG

    def test_returns_none_for_invalid_level(self) -> None:
        """Test returns None for invalid level."""
        from flexium.utils.logging import _get_level_from_env

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "INVALID"}):
            result = _get_level_from_env()

        assert result is None


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_setup(self) -> None:
        """Test default logging setup."""
        from flexium.utils.logging import setup_logging

        stream = StringIO()

        with patch.dict(os.environ, {}, clear=True):
            setup_logging(stream=stream)

        # Get root flexium logger and verify configuration
        logger = logging.getLogger("flexium")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

    def test_debug_mode_uses_verbose_format(self) -> None:
        """Test debug mode uses verbose format."""
        from flexium.utils.logging import setup_logging, DEBUG_FORMAT

        stream = StringIO()

        with patch.dict(os.environ, {}, clear=True):
            setup_logging(debug=True, stream=stream)

        logger = logging.getLogger("flexium")
        handler = logger.handlers[0]
        assert "%(levelname)s" in handler.formatter._fmt

    def test_env_var_overrides_parameter(self) -> None:
        """Test environment variable overrides parameter."""
        from flexium.utils.logging import setup_logging

        stream = StringIO()

        with patch.dict(os.environ, {"FLEXIUM_LOG_LEVEL": "DEBUG"}):
            setup_logging(level=logging.ERROR, stream=stream)

        logger = logging.getLogger("flexium")
        assert logger.level == logging.DEBUG

    def test_custom_level(self) -> None:
        """Test custom logging level."""
        from flexium.utils.logging import setup_logging

        stream = StringIO()

        with patch.dict(os.environ, {}, clear=True):
            setup_logging(level=logging.WARNING, stream=stream)

        logger = logging.getLogger("flexium")
        assert logger.level == logging.WARNING

    def test_default_stream_is_stderr(self) -> None:
        """Test default stream is sys.stderr."""
        from flexium.utils.logging import setup_logging
        import sys

        with patch.dict(os.environ, {}, clear=True):
            setup_logging()

        logger = logging.getLogger("flexium")
        handler = logger.handlers[0]
        assert handler.stream == sys.stderr


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self) -> None:
        """Test get_logger returns a logger."""
        from flexium.utils.logging import get_logger

        logger = get_logger("flexium.test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "flexium.test_module"

    def test_caches_logger(self) -> None:
        """Test get_logger caches loggers."""
        from flexium.utils.logging import get_logger

        logger1 = get_logger("flexium.cache_test")
        logger2 = get_logger("flexium.cache_test")

        assert logger1 is logger2

    def test_adds_flexium_prefix(self) -> None:
        """Test get_logger adds flexium prefix if missing."""
        from flexium.utils.logging import get_logger

        logger = get_logger("some_module")

        assert logger.name == "flexium.some_module"

    def test_does_not_add_prefix_if_present(self) -> None:
        """Test get_logger doesn't add prefix if already present."""
        from flexium.utils.logging import get_logger

        logger = get_logger("flexium.already_prefixed")

        assert logger.name == "flexium.already_prefixed"


class TestLoggingFormats:
    """Tests for logging format constants."""

    def test_default_format_defined(self) -> None:
        """Test DEFAULT_FORMAT is defined."""
        from flexium.utils.logging import DEFAULT_FORMAT

        assert "[flexium]" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT

    def test_debug_format_defined(self) -> None:
        """Test DEBUG_FORMAT is defined."""
        from flexium.utils.logging import DEBUG_FORMAT

        assert "[flexium]" in DEBUG_FORMAT
        assert "%(levelname)s" in DEBUG_FORMAT
        assert "%(name)s" in DEBUG_FORMAT
        assert "%(message)s" in DEBUG_FORMAT
