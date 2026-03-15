"""Test utilities for timing-dependent tests.

This module provides:
- FAST_TIMING: Fast timing constants for tests
- wait_for_condition: Polling utility with exponential backoff
- assert_completes_within: Context manager for timing assertions
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable


# Fast timing constants for tests (in seconds)
# These values are short enough for fast tests but long enough
# to avoid race conditions on slow CI systems.
FAST_TIMING = {
    "HEARTBEAT_INTERVAL": 0.1,      # Time between heartbeats
    "RECONNECT_TIMEOUT": 0.5,       # Time before giving up reconnection
    "MIGRATION_TIMEOUT": 1.0,       # Time for migration to complete
}


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    poll_interval: float = 0.01,
    backoff_multiplier: float = 1.5,
    max_interval: float = 1.0,
) -> bool:
    """Wait for a condition to become true with exponential backoff.

    Args:
        condition: A callable that returns True when the condition is met.
        timeout: Maximum time to wait in seconds.
        poll_interval: Initial polling interval in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        max_interval: Maximum polling interval.

    Returns:
        True if condition was met, False if timeout expired.
    """
    start = time.time()
    interval = poll_interval

    while time.time() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
        interval = min(interval * backoff_multiplier, max_interval)

    return False


@contextmanager
def assert_completes_within(timeout: float):
    """Context manager that asserts the block completes within timeout.

    Args:
        timeout: Maximum time allowed in seconds.

    Raises:
        AssertionError: If the block takes longer than timeout.

    Example:
        with assert_completes_within(2.0):
            slow_operation()
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    assert elapsed < timeout, f"Operation took {elapsed:.2f}s, expected < {timeout}s"


def wait_for_port_ready(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait for a TCP port to become available.

    Args:
        host: Hostname to connect to.
        port: Port number to check.
        timeout: Maximum time to wait.

    Returns:
        True if port became available, False if timeout expired.
    """
    import socket

    def check_port() -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    return wait_for_condition(check_port, timeout=timeout)
