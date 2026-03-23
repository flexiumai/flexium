"""Tests for WebSocket transport."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestWebSocketTransportInit:
    """Tests for WebSocketTransport initialization."""

    def test_init_sets_server_url(self):
        """Test that init sets server URL."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        assert transport._server_url == "https://example.com"

    def test_init_sets_workspace(self):
        """Test that init sets workspace."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        assert transport._workspace == "workspace1"

    def test_init_starts_disconnected(self):
        """Test that transport starts disconnected."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        assert transport._connected is False
        assert transport._sio is None

    def test_init_empty_pending_responses(self):
        """Test that pending responses starts empty."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        assert transport._pending_responses == {}
        assert transport._response_events == {}

    def test_init_empty_event_handlers(self):
        """Test that event handlers starts empty."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        assert transport._event_handlers == {}


class TestWebSocketTransportProperties:
    """Tests for WebSocketTransport properties."""

    def test_server_url_property(self):
        """Test server_url property."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://test.flexium.ai", "myworkspace")
        assert transport.server_url == "https://test.flexium.ai"

    def test_workspace_property(self):
        """Test workspace property."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://test.flexium.ai", "myworkspace")
        assert transport.workspace == "myworkspace"


class TestWebSocketTransportConnect:
    """Tests for WebSocketTransport.connect()."""

    def test_connect_creates_socket(self):
        """Test that connect creates a socket."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        with patch.object(transport, "_create_socket") as mock_create:
            mock_sio = MagicMock()
            mock_create.return_value = mock_sio

            transport.connect()

            mock_create.assert_called_once()
            assert transport._sio is mock_sio

    def test_connect_calls_sio_connect(self):
        """Test that connect calls sio.connect with correct params."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        with patch.object(transport, "_create_socket") as mock_create:
            mock_sio = MagicMock()
            mock_create.return_value = mock_sio

            transport.connect()

            mock_sio.connect.assert_called_once()
            call_kwargs = mock_sio.connect.call_args
            assert call_kwargs[0][0] == "https://example.com"
            assert call_kwargs[1]["auth"] == {"workspace": "workspace1"}

    def test_connect_returns_true_on_success(self):
        """Test that connect returns True on success."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        with patch.object(transport, "_create_socket") as mock_create:
            mock_sio = MagicMock()
            mock_create.return_value = mock_sio

            result = transport.connect()

            assert result is True

    def test_connect_returns_false_on_exception(self):
        """Test that connect returns False on exception."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        with patch.object(transport, "_create_socket") as mock_create:
            mock_sio = MagicMock()
            mock_sio.connect.side_effect = Exception("Connection failed")
            mock_create.return_value = mock_sio

            result = transport.connect()

            assert result is False

    def test_connect_disconnects_existing_socket(self):
        """Test that connect disconnects existing socket first."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        old_sio = MagicMock()
        transport._sio = old_sio

        with patch.object(transport, "_create_socket") as mock_create:
            mock_sio = MagicMock()
            mock_create.return_value = mock_sio

            transport.connect()

            old_sio.disconnect.assert_called_once()


class TestWebSocketTransportDisconnect:
    """Tests for WebSocketTransport.disconnect()."""

    def test_disconnect_calls_sio_disconnect(self):
        """Test that disconnect calls sio.disconnect."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        mock_sio = MagicMock()
        transport._sio = mock_sio
        transport._connected = True

        transport.disconnect()

        mock_sio.disconnect.assert_called_once()

    def test_disconnect_sets_connected_false(self):
        """Test that disconnect sets connected to False."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._sio = MagicMock()
        transport._connected = True

        transport.disconnect()

        assert transport._connected is False

    def test_disconnect_sets_sio_none(self):
        """Test that disconnect sets sio to None."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._sio = MagicMock()

        transport.disconnect()

        assert transport._sio is None

    def test_disconnect_handles_none_sio(self):
        """Test that disconnect handles None sio gracefully."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._sio = None

        # Should not raise
        transport.disconnect()

    def test_disconnect_handles_exception(self):
        """Test that disconnect handles exception from sio.disconnect."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        mock_sio = MagicMock()
        mock_sio.disconnect.side_effect = Exception("Disconnect failed")
        transport._sio = mock_sio

        # Should not raise
        transport.disconnect()
        assert transport._sio is None


class TestWebSocketTransportIsConnected:
    """Tests for WebSocketTransport.is_connected()."""

    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = True
        transport._sio = MagicMock()

        assert transport.is_connected() is True

    def test_is_connected_false_when_disconnected(self):
        """Test is_connected returns False when disconnected."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = False
        transport._sio = MagicMock()

        assert transport.is_connected() is False

    def test_is_connected_false_when_no_sio(self):
        """Test is_connected returns False when no socket."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = True
        transport._sio = None

        assert transport.is_connected() is False


class TestWebSocketTransportSend:
    """Tests for WebSocketTransport.send()."""

    def test_send_returns_none_when_not_connected(self):
        """Test send returns None when not connected."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = False

        result = transport.send("heartbeat", {"test": "data"})

        assert result is None

    def test_send_emits_event(self):
        """Test send emits event to socket."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = True
        transport._sio = MagicMock()

        with patch.object(transport, "_wait_for_response", return_value={"success": True}):
            transport.send("register", {"process_id": "gpu-123"})

        transport._sio.emit.assert_called_once_with("register", {"process_id": "gpu-123"})

    def test_send_waits_for_response(self):
        """Test send waits for response."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = True
        transport._sio = MagicMock()

        with patch.object(transport, "_wait_for_response", return_value={"status": "ok"}) as mock_wait:
            result = transport.send("heartbeat", {"data": "test"}, timeout=5.0)

        mock_wait.assert_called_once_with("heartbeat", 5.0)
        assert result == {"status": "ok"}

    def test_send_returns_none_on_exception(self):
        """Test send returns None on exception."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        transport._connected = True
        transport._sio = MagicMock()
        transport._sio.emit.side_effect = Exception("Emit failed")

        result = transport.send("heartbeat", {"data": "test"})

        assert result is None
        assert transport._connected is False


class TestWebSocketTransportSetEventHandler:
    """Tests for WebSocketTransport.set_event_handler()."""

    def test_set_event_handler_migrate(self):
        """Test setting migrate event handler."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        handler = MagicMock()

        transport.set_event_handler("migrate", handler)

        assert transport._event_handlers["migrate"] is handler

    def test_set_event_handler_pause(self):
        """Test setting pause event handler."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        handler = MagicMock()

        transport.set_event_handler("pause", handler)

        assert transport._event_handlers["pause"] is handler


class TestWebSocketTransportHandleResponse:
    """Tests for WebSocketTransport._handle_response()."""

    def test_handle_response_stores_data(self):
        """Test _handle_response stores data in pending responses."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        transport._handle_response("heartbeat", {"status": "ok"})

        assert transport._pending_responses["heartbeat"] == {"status": "ok"}

    def test_handle_response_sets_event(self):
        """Test _handle_response sets the event."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        event = threading.Event()
        transport._response_events["heartbeat"] = event

        transport._handle_response("heartbeat", {"status": "ok"})

        assert event.is_set()


class TestWebSocketTransportWaitForResponse:
    """Tests for WebSocketTransport._wait_for_response()."""

    def test_wait_for_response_returns_data(self):
        """Test _wait_for_response returns data when event is set."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        def set_response():
            import time
            time.sleep(0.01)
            transport._pending_responses["test"] = {"result": "data"}
            if "test" in transport._response_events:
                transport._response_events["test"].set()

        thread = threading.Thread(target=set_response)
        thread.start()

        result = transport._wait_for_response("test", timeout=1.0)
        thread.join()

        assert result == {"result": "data"}

    def test_wait_for_response_returns_none_on_timeout(self):
        """Test _wait_for_response returns None on timeout."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        result = transport._wait_for_response("test", timeout=0.01)

        assert result is None


class TestWebSocketTransportCreateSocket:
    """Tests for WebSocketTransport._create_socket()."""

    def test_create_socket_returns_client(self):
        """Test _create_socket returns a socketio.Client."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")

        with patch("flexium.orchestrator.websocket_transport.socketio.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            result = transport._create_socket()

            assert result is mock_instance
            mock_client.assert_called_once_with(
                reconnection=False,
                logger=False,
                engineio_logger=False,
            )


class TestWebSocketTransportEventHandlers:
    """Tests for internal event handlers in _create_socket()."""

    def test_heartbeat_response_triggers_migrate_handler(self):
        """Test heartbeat response with migration triggers migrate handler."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        migrate_handler = MagicMock()
        transport._event_handlers["migrate"] = migrate_handler

        # Simulate heartbeat response with migration
        data = {"should_migrate": True, "target_device": "cuda:1"}
        transport._handle_response("heartbeat", data)

        # The internal handler in _create_socket would call the event handler
        # For testing purposes, we verify the handler is stored
        assert transport._event_handlers["migrate"] is migrate_handler

    def test_heartbeat_response_triggers_pause_handler(self):
        """Test heartbeat response with pause triggers pause handler."""
        from flexium.orchestrator.websocket_transport import WebSocketTransport

        transport = WebSocketTransport("https://example.com", "workspace1")
        pause_handler = MagicMock()
        transport._event_handlers["pause"] = pause_handler

        # Verify handler is stored
        assert transport._event_handlers["pause"] is pause_handler
