"""WebSocket transport implementation.

This module provides a WebSocket-based transport for communicating with
the orchestrator server. Uses Socket.IO for reliable real-time communication.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

import socketio

from flexium.orchestrator.transport import Transport
from flexium.utils.logging import get_logger

logger = get_logger(__name__)


class WebSocketTransport(Transport):
    """WebSocket transport using Socket.IO.

    Handles the low-level WebSocket communication with the orchestrator server.
    Supports both request/response patterns and server-initiated events.
    """

    def __init__(
        self,
        server_url: str,
        workspace: str,
    ):
        """Initialize WebSocket transport.

        Parameters:
            server_url: The WebSocket server URL (e.g., "https://app.flexium.ai").
            workspace: The workspace name for authentication.
        """
        self._server_url = server_url
        self._workspace = workspace

        # Socket.IO client
        self._sio: Optional[socketio.Client] = None
        self._connected = False

        # Response tracking for request/response pattern
        self._pending_responses: Dict[str, Any] = {}
        self._response_events: Dict[str, threading.Event] = {}

        # Event handlers for server-initiated events
        self._event_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def _create_socket(self) -> socketio.Client:
        """Create and configure Socket.IO client."""
        sio = socketio.Client(
            reconnection=False,  # We handle reconnection at a higher level
            logger=False,
            engineio_logger=False,
        )

        @sio.event
        def connect():
            logger.debug("WebSocket connected")
            self._connected = True

        @sio.event
        def disconnect():
            logger.debug("WebSocket disconnected")
            self._connected = False

        @sio.on("connected")
        def on_connected(data):
            logger.debug(f"Server acknowledged connection: {data}")

        @sio.on("error")
        def on_error(data):
            logger.warning(f"Server error: {data.get('message', data)}")

        # Response handlers for request/response pattern
        @sio.on("register_response")
        def on_register_response(data):
            self._handle_response("register", data)

        @sio.on("heartbeat_response")
        def on_heartbeat_response(data):
            self._handle_response("heartbeat", data)
            # Also trigger event handler if migration/pause requested
            if data.get("should_migrate") and data.get("target_device"):
                if data["target_device"] == "__PAUSE__":
                    if "pause" in self._event_handlers:
                        self._event_handlers["pause"](data)
                else:
                    if "migrate" in self._event_handlers:
                        self._event_handlers["migrate"](data)

        @sio.on("complete_migration_response")
        def on_complete_migration_response(data):
            self._handle_response("complete_migration", data)

        @sio.on("complete_pause_response")
        def on_complete_pause_response(data):
            self._handle_response("complete_pause", data)

        @sio.on("unregister_response")
        def on_unregister_response(data):
            self._handle_response("unregister", data)

        # Server-initiated events
        @sio.on("migrate")
        def on_migrate(data):
            if "migrate" in self._event_handlers:
                self._event_handlers["migrate"](data)

        @sio.on("pause")
        def on_pause(data):
            if "pause" in self._event_handlers:
                self._event_handlers["pause"](data)

        return sio

    def _handle_response(self, event_type: str, data: Any) -> None:
        """Handle a response from the server."""
        self._pending_responses[event_type] = data
        if event_type in self._response_events:
            self._response_events[event_type].set()

    def _wait_for_response(self, event_type: str, timeout: float) -> Optional[Dict[str, Any]]:
        """Wait for a response from the server."""
        event = threading.Event()
        self._response_events[event_type] = event

        if event.wait(timeout):
            response = self._pending_responses.pop(event_type, None)
            del self._response_events[event_type]
            return response
        else:
            self._response_events.pop(event_type, None)
            return None

    def connect(self) -> bool:
        """Connect to the WebSocket server.

        Returns:
            True if connected successfully, False otherwise.
        """
        if self._sio is not None:
            try:
                self._sio.disconnect()
            except Exception:
                pass

        self._sio = self._create_socket()

        try:
            # Use polling transport only - more reliable through proxies/Cloudflare
            # and avoids sticky session issues with multiple gunicorn workers.
            # WebSocket upgrade fails when requests hit different workers.
            logger.info(f"[flexium] Connecting to {self._server_url} workspace={self._workspace}")
            self._sio.connect(
                self._server_url,
                socketio_path="/socket.io",
                transports=["polling"],
                wait_timeout=10,
                headers={},
                auth={"workspace": self._workspace},
            )
            logger.info(f"[flexium] Socket.IO connected: {self._connected}")
            return True
        except Exception as e:
            logger.warning(f"[flexium] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._sio is not None:
            try:
                self._sio.disconnect()
            except Exception:
                pass
            self._sio = None
        self._connected = False
        logger.debug("Disconnected from orchestrator")

    def is_connected(self) -> bool:
        """Check if connected to server."""
        connected = self._connected and self._sio is not None
        return connected

    def send(self, event: str, data: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Send a message and wait for response.

        Parameters:
            event: The event type (e.g., "register", "heartbeat").
            data: The message payload.
            timeout: Maximum time to wait for response.

        Returns:
            Response data, or None if failed/timeout.
        """
        if not self.is_connected():
            return None

        try:
            self._sio.emit(event, data)
            response = self._wait_for_response(event, timeout)
            return response
        except Exception as e:
            logger.debug(f"Send failed: {e}")
            self._connected = False
            return None

    def set_event_handler(self, event: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Set handler for server-initiated events.

        Parameters:
            event: Event type ("migrate", "pause").
            handler: Callback function.
        """
        self._event_handlers[event] = handler

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return self._server_url

    @property
    def workspace(self) -> str:
        """Get workspace name."""
        return self._workspace
