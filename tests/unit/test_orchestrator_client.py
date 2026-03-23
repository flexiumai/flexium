"""Tests for the orchestrator client module.

These tests use MockTransport to test the OrchestratorClient business logic
without any network dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import time

import pytest


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_connection_states_exist(self) -> None:
        """Test all connection states are defined."""
        from flexium.orchestrator.client import ConnectionState

        assert hasattr(ConnectionState, "DISCONNECTED")
        assert hasattr(ConnectionState, "CONNECTING")
        assert hasattr(ConnectionState, "CONNECTED")
        assert hasattr(ConnectionState, "RECONNECTING")
        assert hasattr(ConnectionState, "LOCAL_MODE")

    def test_connection_states_unique(self) -> None:
        """Test connection states have unique values."""
        from flexium.orchestrator.client import ConnectionState

        states = [
            ConnectionState.DISCONNECTED,
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
            ConnectionState.RECONNECTING,
            ConnectionState.LOCAL_MODE,
        ]
        assert len(states) == len(set(states))


class TestTransportInterface:
    """Tests for Transport abstract interface."""

    def test_transport_is_abstract(self) -> None:
        """Test Transport cannot be instantiated directly."""
        from flexium.orchestrator.transport import Transport

        with pytest.raises(TypeError):
            Transport()

    def test_mock_transport_implements_interface(self) -> None:
        """Test MockTransport implements all Transport methods."""
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()

        # All interface methods should exist
        assert hasattr(transport, "connect")
        assert hasattr(transport, "disconnect")
        assert hasattr(transport, "is_connected")
        assert hasattr(transport, "send")
        assert hasattr(transport, "set_event_handler")
        assert hasattr(transport, "server_url")


class TestMockTransport:
    """Tests for MockTransport."""

    def test_mock_transport_default_handler(self) -> None:
        """Test MockTransport with default handler."""
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        transport.connect()

        # Default handler returns success
        response = transport.send("register", {"device": "cuda:0"})
        assert response["success"] is True
        assert response["assigned_device"] == "cuda:0"

    def test_mock_transport_custom_handler(self) -> None:
        """Test MockTransport with custom handler."""
        from flexium.orchestrator.transport import MockTransport

        def custom_handler(event, data):
            if event == "heartbeat":
                return {"success": True, "should_migrate": True, "target_device": "cuda:1"}
            return {"success": False}

        transport = MockTransport(response_handler=custom_handler)
        transport.connect()

        response = transport.send("heartbeat", {})
        assert response["should_migrate"] is True
        assert response["target_device"] == "cuda:1"

    def test_mock_transport_event_handler(self) -> None:
        """Test MockTransport event handler simulation."""
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        received_events = []

        transport.set_event_handler("migrate", lambda data: received_events.append(data))
        transport.simulate_server_event("migrate", {"target_device": "cuda:2"})

        assert len(received_events) == 1
        assert received_events[0]["target_device"] == "cuda:2"

    def test_mock_transport_not_connected_returns_none(self) -> None:
        """Test MockTransport returns None when not connected."""
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        # Don't call connect()

        response = transport.send("register", {"device": "cuda:0"})
        assert response is None


class TestOrchestratorClient:
    """Tests for OrchestratorClient with MockTransport."""

    def test_client_with_address(self) -> None:
        """Test client initialization with address string."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="flexium.ai:80/testworkspace")
        assert "flexium.ai" in client.server_url

    def test_client_with_transport(self) -> None:
        """Test client initialization with transport."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport(server_url="mock://test/workspace")
        client = OrchestratorClient(transport=transport)

        assert client.server_url == "mock://test/workspace"

    def test_client_requires_address_or_transport(self) -> None:
        """Test client requires either address or transport."""
        from flexium.orchestrator.client import OrchestratorClient

        with pytest.raises(ValueError, match="Either 'address' or 'transport' must be provided"):
            OrchestratorClient()

    def test_client_init_with_heartbeat_interval(self) -> None:
        """Test client with custom heartbeat interval."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport, heartbeat_interval=5.0)

        assert client._heartbeat_interval == 5.0


class TestOrchestratorClientAddressParsing:
    """Tests for address parsing."""

    def test_parse_address_with_workspace(self) -> None:
        """Test parsing address with workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        server_url, workspace = OrchestratorClient._parse_address("flexium.ai:80/myworkspace")
        assert server_url == "http://flexium.ai:80"
        assert workspace == "myworkspace"

    def test_parse_address_nested_workspace(self) -> None:
        """Test parsing address with nested workspace path."""
        from flexium.orchestrator.client import OrchestratorClient

        server_url, workspace = OrchestratorClient._parse_address("flexium.ai:80/org/project")
        assert server_url == "http://flexium.ai:80"
        assert workspace == "org/project"

    def test_parse_address_https_443(self) -> None:
        """Test parsing address uses HTTPS for port 443."""
        from flexium.orchestrator.client import OrchestratorClient

        server_url, workspace = OrchestratorClient._parse_address("flexium.ai:443/workspace")
        assert server_url == "https://flexium.ai:443"
        assert workspace == "workspace"

    def test_parse_address_no_port(self) -> None:
        """Test parsing address without port defaults to HTTPS."""
        from flexium.orchestrator.client import OrchestratorClient

        server_url, workspace = OrchestratorClient._parse_address("flexium.ai/workspace")
        assert server_url == "https://flexium.ai"
        assert workspace == "workspace"

    def test_parse_address_invalid_format(self) -> None:
        """Test parsing invalid address format."""
        from flexium.orchestrator.client import OrchestratorClient

        with pytest.raises(ValueError, match="Invalid address format"):
            OrchestratorClient._parse_address("localhost:50051")

    def test_parse_address_empty_workspace(self) -> None:
        """Test parsing address with empty workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        with pytest.raises(ValueError, match="Workspace is required"):
            OrchestratorClient._parse_address("localhost:50051/")


class TestOrchestratorClientRegister:
    """Tests for register method."""

    def test_register_success(self) -> None:
        """Test register returns device on success."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        result = client.register(
            process_id="test-123",
            device="cuda:0",
        )

        assert result == "cuda:0"

    def test_register_stores_process_info(self) -> None:
        """Test register stores process info for reconnection."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        client.register(
            process_id="test-123",
            device="cuda:0",
            min_gpus=2,
            max_gpus=4,
            priority=80,
        )

        assert client._process_id == "test-123"
        assert client._min_gpus == 2
        assert client._max_gpus == 4
        assert client._priority == 80

    def test_register_rejection_enters_local_mode(self) -> None:
        """Test register enters local mode after max retries on rejection."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        def reject_handler(event, data):
            return {"success": False, "message": "Rejected"}

        transport = MockTransport(response_handler=reject_handler)
        client = OrchestratorClient(
            transport=transport,
            max_retries=1,
            retry_delay=0.01,
        )

        result = client.register(
            process_id="test-123",
            device="cuda:0",
        )

        assert result is None
        assert client._state == ConnectionState.LOCAL_MODE


class TestOrchestratorClientHeartbeat:
    """Tests for heartbeat method."""

    def test_heartbeat_success(self) -> None:
        """Test heartbeat returns response."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        result = client.heartbeat(memory_allocated=1000000)

        assert result is not None
        assert result["success"] is True

    def test_heartbeat_returns_migration(self) -> None:
        """Test heartbeat returns migration directive."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        def migration_handler(event, data):
            if event == "heartbeat":
                return {"success": True, "should_migrate": True, "target_device": "cuda:1"}
            return MockTransport._default_handler(event, data)

        transport = MockTransport(response_handler=migration_handler)
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        result = client.heartbeat()

        assert result["should_migrate"] is True
        assert result["target_device"] == "cuda:1"

    def test_heartbeat_not_connected(self) -> None:
        """Test heartbeat returns None when not connected."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        # Don't connect
        client = OrchestratorClient(transport=transport)

        result = client.heartbeat()

        assert result is None

    def test_heartbeat_connection_lost_enters_local_mode(self) -> None:
        """Test heartbeat enters local mode when connection is lost after registration."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        # Verify initially connected
        assert client._state == ConnectionState.CONNECTED

        # Simulate connection loss - prevent auto-reconnect by making connect fail
        transport._connected = False
        original_connect = transport.connect
        transport.connect = lambda: False

        # Heartbeat should detect disconnect and enter local mode
        result = client.heartbeat()

        # Restore for cleanup
        transport.connect = original_connect

        assert result is None
        assert client._state == ConnectionState.LOCAL_MODE

    def test_heartbeat_reconnection_attempt_in_local_mode(self) -> None:
        """Test heartbeat attempts reconnection when in local mode."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        # Simulate connection loss
        transport._connected = False
        client._state = ConnectionState.LOCAL_MODE

        # First heartbeat won't reconnect (just sets last_reconnect_attempt)
        client.heartbeat()

        # Force reconnect attempt by setting last attempt far in past
        client._last_reconnect_attempt = 0

        # Reconnect transport
        transport._connected = True

        # Next heartbeat should successfully reconnect
        result = client.heartbeat()

        # Should be back to connected state after successful reconnect
        assert client._state == ConnectionState.CONNECTED


class TestOrchestratorClientCompleteMigration:
    """Tests for complete_migration method."""

    def test_complete_migration_success(self) -> None:
        """Test complete_migration returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
            gpu_uuid="GPU-12345678",
        )

        assert result is True

    def test_complete_migration_not_connected(self) -> None:
        """Test complete_migration returns False when not connected."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        # Don't connect

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert result is False


class TestOrchestratorClientUpdateDevice:
    """Tests for update_device method."""

    def test_update_device(self) -> None:
        """Test update_device updates current device."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client._current_device = "cuda:0"

        client.update_device("cuda:1")

        assert client._current_device == "cuda:1"


class TestOrchestratorClientCallbacks:
    """Tests for migration and pause callbacks."""

    def test_set_migration_callback(self) -> None:
        """Test setting migration callback."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        callback_calls = []
        def callback(target: str) -> None:
            callback_calls.append(target)

        client.set_migration_callback(callback)

        # Simulate server event
        transport.simulate_server_event("migrate", {"target_device": "cuda:2"})

        assert len(callback_calls) == 1
        assert callback_calls[0] == "cuda:2"

    def test_set_pause_callback(self) -> None:
        """Test setting pause callback."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        callback_calls = []
        def callback() -> None:
            callback_calls.append(True)

        client.set_pause_callback(callback)

        # Simulate server event
        transport.simulate_server_event("pause", {})

        assert len(callback_calls) == 1


class TestOrchestratorClientReconnection:
    """Tests for reconnection logic."""

    def test_should_attempt_reconnect_first_time(self) -> None:
        """Test should_attempt_reconnect returns True on first attempt."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client._last_reconnect_attempt = None

        assert client._should_attempt_reconnect() is True

    def test_should_attempt_reconnect_after_interval(self) -> None:
        """Test should_attempt_reconnect returns True after interval."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport, reconnect_interval=0.0)
        client._last_reconnect_attempt = time.time() - 1

        assert client._should_attempt_reconnect() is True

    def test_should_attempt_reconnect_too_soon(self) -> None:
        """Test should_attempt_reconnect returns False if too soon."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport, reconnect_interval=60.0)
        client._last_reconnect_attempt = time.time()

        assert client._should_attempt_reconnect() is False


class TestOrchestratorClientUnregister:
    """Tests for unregister method."""

    def test_unregister_success(self) -> None:
        """Test unregister returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        result = client.unregister("test-123")

        assert result is True

    def test_unregister_uses_stored_process_id(self) -> None:
        """Test unregister uses stored process ID."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        client.register(process_id="test-123", device="cuda:0")

        result = client.unregister()  # No process_id argument

        assert result is True

    def test_unregister_no_process_id(self) -> None:
        """Test unregister returns False with no process ID."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)
        # Don't register

        result = client.unregister()

        assert result is False


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_heartbeat_interval(self) -> None:
        """Test DEFAULT_HEARTBEAT_INTERVAL is defined."""
        from flexium.orchestrator.client import DEFAULT_HEARTBEAT_INTERVAL

        assert DEFAULT_HEARTBEAT_INTERVAL > 0


class TestOrchestratorClientAlias:
    """Tests for WebSocketClient alias."""

    def test_websocket_client_is_orchestrator_client(self) -> None:
        """Test WebSocketClient is aliased to OrchestratorClient."""
        from flexium.orchestrator.client import OrchestratorClient, WebSocketClient

        assert WebSocketClient is OrchestratorClient

    def test_can_instantiate_websocket_client(self) -> None:
        """Test can instantiate WebSocketClient."""
        from flexium.orchestrator.client import WebSocketClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = WebSocketClient(transport=transport)
        assert client is not None


class TestPausedReconnection:
    """Tests for reconnection while paused.

    When a client is paused and the server restarts, the client should:
    1. Reconnect with device="__PAUSED__"
    2. Preserve the migratable flag (False for driver 550-579)
    3. Preserve the cached memory value
    """

    def test_set_paused_stores_state(self) -> None:
        """Test set_paused stores paused state and memory."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        # Initially not paused
        assert client._is_paused is False
        assert client._cached_memory_reserved == 0

        # Set paused with memory
        client.set_paused(True, memory_reserved=1024 * 1024 * 1024)

        assert client._is_paused is True
        assert client._cached_memory_reserved == 1024 * 1024 * 1024

    def test_set_paused_clear(self) -> None:
        """Test set_paused(False) clears paused state."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        client.set_paused(True, memory_reserved=1024)
        assert client._is_paused is True

        client.set_paused(False)
        assert client._is_paused is False
        # Memory should be preserved (not cleared)
        assert client._cached_memory_reserved == 1024

    def test_try_reconnect_uses_paused_device(self) -> None:
        """Test _try_reconnect uses __PAUSED__ device when paused."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        # Track what device is sent during registration
        registered_data = {}

        def capture_handler(event, data):
            if event == "register":
                registered_data.update(data)
                return {"success": True, "assigned_device": data.get("device", "cuda:0")}
            return {"success": True}

        transport = MockTransport(response_handler=capture_handler)
        client = OrchestratorClient(transport=transport)

        # Initial registration
        client.register(process_id="test-123", device="cuda:0", migratable=False)
        assert registered_data["device"] == "cuda:0"
        assert registered_data["migratable"] is False

        # Simulate pause
        client.set_paused(True, memory_reserved=2048)

        # Simulate disconnect
        transport.disconnect()
        client._state = ConnectionState.LOCAL_MODE

        # Clear registered_data for reconnection
        registered_data.clear()

        # Reconnect
        result = client._try_reconnect()

        assert result is True
        assert registered_data["device"] == "__PAUSED__"
        assert registered_data["migratable"] is False
        assert registered_data["memory_reserved"] == 2048

    def test_try_reconnect_preserves_migratable_false(self) -> None:
        """Test _try_reconnect preserves migratable=False for non-migratable drivers."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        registered_migratable = []

        def capture_handler(event, data):
            if event == "register":
                registered_migratable.append(data.get("migratable", True))
                return {"success": True, "assigned_device": data.get("device", "cuda:0")}
            return {"success": True}

        transport = MockTransport(response_handler=capture_handler)
        client = OrchestratorClient(transport=transport)

        # Initial registration with migratable=False (driver 550-579)
        client.register(process_id="test-123", device="cuda:0", migratable=False)
        assert registered_migratable[-1] is False

        # Simulate pause and disconnect
        client.set_paused(True)
        transport.disconnect()
        client._state = ConnectionState.LOCAL_MODE

        # Reconnect
        client._try_reconnect()

        # migratable should still be False
        assert registered_migratable[-1] is False

    def test_try_reconnect_not_paused_uses_original_device(self) -> None:
        """Test _try_reconnect uses original device when not paused."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        from flexium.orchestrator.transport import MockTransport

        registered_devices = []

        def capture_handler(event, data):
            if event == "register":
                registered_devices.append(data.get("device"))
                return {"success": True, "assigned_device": data.get("device", "cuda:0")}
            return {"success": True}

        transport = MockTransport(response_handler=capture_handler)
        client = OrchestratorClient(transport=transport)

        # Initial registration
        client.register(process_id="test-123", device="cuda:0")
        assert registered_devices[-1] == "cuda:0"

        # Simulate disconnect without pause
        transport.disconnect()
        client._state = ConnectionState.LOCAL_MODE

        # Reconnect (not paused)
        client._try_reconnect()

        # Should use original device, not __PAUSED__
        assert registered_devices[-1] == "cuda:0"
