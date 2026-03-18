"""Orchestrator client module for flexium.

Provides the client for connecting to the orchestrator server from
training processes. Uses a pluggable transport layer for flexibility.

Architecture:
    OrchestratorClient (business logic)
            │
            ▼
    Transport (abstract interface)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  WebSocket  gRPC   Mock
  Transport        Transport
"""

from flexium.orchestrator.client import (
    OrchestratorClient,
    WebSocketClient,
    ConnectionState,
    DEFAULT_HEARTBEAT_INTERVAL,
)
from flexium.orchestrator.transport import (
    Transport,
    MockTransport,
)
from flexium.orchestrator.websocket_transport import WebSocketTransport

__all__ = [
    # Client
    "OrchestratorClient",
    "WebSocketClient",  # Alias for OrchestratorClient
    "ConnectionState",
    "DEFAULT_HEARTBEAT_INTERVAL",
    # Transports
    "Transport",
    "MockTransport",
    "WebSocketTransport",
]
