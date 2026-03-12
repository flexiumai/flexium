"""Tests for timing constants module."""

import pytest


class TestTimingConstants:
    """Tests for timing module constants."""

    def test_heartbeat_interval_exists(self) -> None:
        """Test HEARTBEAT_INTERVAL constant exists and is reasonable."""
        from flexium.timing import HEARTBEAT_INTERVAL

        assert isinstance(HEARTBEAT_INTERVAL, (int, float))
        assert HEARTBEAT_INTERVAL > 0
        # Heartbeat should be between 1 and 60 seconds
        assert 1 <= HEARTBEAT_INTERVAL <= 60

    def test_default_max_retries_exists(self) -> None:
        """Test DEFAULT_MAX_RETRIES constant exists and is reasonable."""
        from flexium.timing import DEFAULT_MAX_RETRIES

        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert DEFAULT_MAX_RETRIES > 0
        assert DEFAULT_MAX_RETRIES <= 10  # Reasonable upper bound

    def test_default_retry_delay_exists(self) -> None:
        """Test DEFAULT_RETRY_DELAY constant exists and is reasonable."""
        from flexium.timing import DEFAULT_RETRY_DELAY

        assert isinstance(DEFAULT_RETRY_DELAY, (int, float))
        assert DEFAULT_RETRY_DELAY > 0
        assert DEFAULT_RETRY_DELAY <= 30  # Reasonable upper bound

    def test_default_reconnect_interval_exists(self) -> None:
        """Test DEFAULT_RECONNECT_INTERVAL constant exists and is reasonable."""
        from flexium.timing import DEFAULT_RECONNECT_INTERVAL

        assert isinstance(DEFAULT_RECONNECT_INTERVAL, (int, float))
        assert DEFAULT_RECONNECT_INTERVAL > 0

    def test_default_backoff_multiplier_exists(self) -> None:
        """Test DEFAULT_BACKOFF_MULTIPLIER constant exists and is reasonable."""
        from flexium.timing import DEFAULT_BACKOFF_MULTIPLIER

        assert isinstance(DEFAULT_BACKOFF_MULTIPLIER, (int, float))
        # Backoff multiplier should be > 1 for exponential growth
        assert DEFAULT_BACKOFF_MULTIPLIER >= 1.0
        assert DEFAULT_BACKOFF_MULTIPLIER <= 10.0  # Reasonable upper bound

    def test_all_constants_importable(self) -> None:
        """Test all expected constants can be imported."""
        from flexium.timing import (
            HEARTBEAT_INTERVAL,
            DEFAULT_MAX_RETRIES,
            DEFAULT_RETRY_DELAY,
            DEFAULT_RECONNECT_INTERVAL,
            DEFAULT_BACKOFF_MULTIPLIER,
        )

        # Just verify they all exist
        assert HEARTBEAT_INTERVAL is not None
        assert DEFAULT_MAX_RETRIES is not None
        assert DEFAULT_RETRY_DELAY is not None
        assert DEFAULT_RECONNECT_INTERVAL is not None
        assert DEFAULT_BACKOFF_MULTIPLIER is not None
