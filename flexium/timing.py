"""Timing constants for flexium client.

These constants control various timeouts and intervals used by the client
when connecting to the orchestrator.
"""

# How often to send heartbeats to orchestrator (seconds)
HEARTBEAT_INTERVAL = 5.0

# Connection retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RECONNECT_INTERVAL = 30.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
