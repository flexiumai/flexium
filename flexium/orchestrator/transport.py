"""Transport abstraction layer for orchestrator communication.

This module defines the Transport protocol (interface) that all transport
implementations must follow. This allows swapping between gRPC, WebSocket,
or any other transport without changing the OrchestratorClient business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class Transport(ABC):
    """Abstract base class for orchestrator transports.

    All transport implementations (WebSocket, gRPC, Mock, etc.) must implement
    this interface. The OrchestratorClient uses this interface to communicate
    with the server, remaining agnostic of the underlying transport mechanism.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the server.

        Returns:
            True if connection succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the server."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected to the server.

        Returns:
            True if connected, False otherwise.
        """
        pass

    @abstractmethod
    def send(self, event: str, data: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send a message to the server and wait for response.

        Parameters:
            event: The event/message type (e.g., "register", "heartbeat").
            data: The message payload.
            timeout: Maximum time to wait for response in seconds.

        Returns:
            Response data from server, or None if failed/timeout.
        """
        pass

    @abstractmethod
    def set_event_handler(self, event: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Set a handler for server-initiated events.

        Parameters:
            event: The event type to handle (e.g., "migrate", "pause").
            handler: Callback function that receives the event data.
        """
        pass

    @property
    @abstractmethod
    def server_url(self) -> str:
        """Get the server URL."""
        pass


class MockTransport(Transport):
    """Mock transport for testing.

    This transport doesn't connect to any server. Instead, it uses
    a handler function to simulate server responses. Perfect for unit tests.

    Example:
        def mock_handler(event, data):
            if event == "register":
                return {"success": True, "assigned_device": data["device"]}
            elif event == "heartbeat":
                return {"success": True, "should_migrate": False}
            return {"success": False}

        transport = MockTransport(mock_handler)
        client = OrchestratorClient(transport=transport)
    """

    def __init__(
        self,
        response_handler: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        server_url: str = "mock://localhost/test",
    ):
        """Initialize mock transport.

        Parameters:
            response_handler: Function that receives (event, data) and returns response dict.
            server_url: Mock server URL for display purposes.
        """
        self._response_handler = response_handler or self._default_handler
        self._server_url = server_url
        self._connected = False
        self._event_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    @staticmethod
    def _default_handler(event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler that returns success for all events."""
        if event == "register":
            return {"success": True, "assigned_device": data.get("device", "cuda:0")}
        elif event == "heartbeat":
            return {"success": True, "should_migrate": False}
        elif event == "complete_migration":
            return {"success": True}
        elif event == "complete_pause":
            return {"success": True}
        elif event == "unregister":
            return {"success": True}
        return {"success": True}

    def connect(self) -> bool:
        """Simulate connection."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False

    def is_connected(self) -> bool:
        """Check mock connection state."""
        return self._connected

    def send(self, event: str, data: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send message and get mock response."""
        if not self._connected:
            return None
        return self._response_handler(event, data)

    def set_event_handler(self, event: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Set event handler."""
        self._event_handlers[event] = handler

    @property
    def server_url(self) -> str:
        """Get mock server URL."""
        return self._server_url

    def simulate_server_event(self, event: str, data: Dict[str, Any]) -> None:
        """Simulate a server-initiated event (for testing).

        Parameters:
            event: The event type (e.g., "migrate", "pause").
            data: The event data.
        """
        if event in self._event_handlers:
            self._event_handlers[event](data)
