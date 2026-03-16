"""Tests for the orchestrator client module."""

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


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_init_default(self) -> None:
        """Test ConnectionManager initialization with defaults."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        assert cm.state == ConnectionState.DISCONNECTED
        assert cm.is_healthy is False
        assert cm.is_local_mode is False

    def test_init_with_params(self) -> None:
        """Test ConnectionManager initialization with parameters."""
        from flexium.orchestrator.client import ConnectionManager

        cm = ConnectionManager(
            max_retries=5,
            retry_delay=2.0,
            reconnect_interval=30.0,
            backoff_multiplier=1.5,
        )
        assert cm._max_retries == 5
        assert cm._retry_delay == 2.0
        assert cm._reconnect_interval == 30.0
        assert cm._backoff_multiplier == 1.5

    def test_is_healthy_when_connected(self) -> None:
        """Test is_healthy returns True when connected."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state = ConnectionState.CONNECTED
        assert cm.is_healthy is True

    def test_is_healthy_when_not_connected(self) -> None:
        """Test is_healthy returns False when not connected."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        for state in [
            ConnectionState.DISCONNECTED,
            ConnectionState.CONNECTING,
            ConnectionState.RECONNECTING,
            ConnectionState.LOCAL_MODE,
        ]:
            cm._state = state
            assert cm.is_healthy is False

    def test_is_local_mode(self) -> None:
        """Test is_local_mode property."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()

        cm._state = ConnectionState.LOCAL_MODE
        assert cm.is_local_mode is True

        cm._state = ConnectionState.CONNECTED
        assert cm.is_local_mode is False

    def test_on_success(self) -> None:
        """Test on_success updates state correctly."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state = ConnectionState.RECONNECTING
        cm._consecutive_failures = 3

        cm.on_success()

        assert cm._state == ConnectionState.CONNECTED
        assert cm._consecutive_failures == 0
        assert cm._last_success_time is not None

    def test_on_failure_increments_count(self) -> None:
        """Test on_failure increments failure count."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager(max_retries=5)
        cm._state = ConnectionState.CONNECTED
        cm._consecutive_failures = 0

        should_retry = cm.on_failure()

        assert cm._consecutive_failures == 1
        assert cm._state == ConnectionState.RECONNECTING
        assert should_retry is True

    def test_on_failure_switches_to_local_mode(self) -> None:
        """Test on_failure switches to local mode after max retries."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager(max_retries=3)
        cm._state = ConnectionState.CONNECTED
        cm._consecutive_failures = 2  # One more failure will exceed max

        should_retry = cm.on_failure()

        assert cm._consecutive_failures == 3
        assert cm._state == ConnectionState.LOCAL_MODE
        assert should_retry is False

    def test_should_attempt_reconnect_when_connected(self) -> None:
        """Test should_attempt_reconnect returns False when connected."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state = ConnectionState.CONNECTED

        assert cm.should_attempt_reconnect() is False

    def test_should_attempt_reconnect_in_local_mode(self) -> None:
        """Test should_attempt_reconnect respects interval in local mode."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager(reconnect_interval=60.0)
        cm._state = ConnectionState.LOCAL_MODE
        cm._last_reconnect_attempt = None

        # First attempt should be allowed
        assert cm.should_attempt_reconnect() is True

        # Record attempt
        cm._last_reconnect_attempt = time.time()

        # Immediate second attempt should not be allowed
        assert cm.should_attempt_reconnect() is False

    def test_set_state_callback(self) -> None:
        """Test state change callback is called."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        callback_calls = []

        def callback(state: ConnectionState) -> None:
            callback_calls.append(state)

        cm.set_state_callback(callback)
        cm._set_state(ConnectionState.CONNECTED)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ConnectionState.CONNECTED

    def test_get_retry_delay_with_backoff(self) -> None:
        """Test get_retry_delay applies exponential backoff."""
        from flexium.orchestrator.client import ConnectionManager

        cm = ConnectionManager(retry_delay=1.0, backoff_multiplier=2.0)

        cm._consecutive_failures = 0
        delay0 = cm.get_retry_delay()

        cm._consecutive_failures = 1
        delay1 = cm.get_retry_delay()

        cm._consecutive_failures = 2
        delay2 = cm.get_retry_delay()

        assert delay1 > delay0
        assert delay2 > delay1


class TestOrchestratorClient:
    """Tests for OrchestratorClient class."""

    def test_client_init(self) -> None:
        """Test OrchestratorClient initialization."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        assert client.address == "localhost:50051"
        assert client.connection_manager is not None

    def test_client_init_with_heartbeat_interval(self) -> None:
        """Test OrchestratorClient with custom heartbeat interval."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(
            address="localhost:50051",
            heartbeat_interval=5.0,
        )
        assert client._heartbeat_interval == 5.0

    def test_client_connect_creates_channel(self) -> None:
        """Test connect creates gRPC channel."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")

        with patch("grpc.insecure_channel") as mock_channel:
            mock_channel.return_value = MagicMock()
            client.connect()

        mock_channel.assert_called_once()
        assert client._channel is not None

    def test_client_disconnect(self) -> None:
        """Test disconnect closes channel."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_channel = MagicMock()
        client._channel = mock_channel

        client.disconnect()

        mock_channel.close.assert_called_once()
        assert client._channel is None

    def test_client_disconnect_no_channel(self) -> None:
        """Test disconnect handles no channel gracefully."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        client._channel = None

        # Should not raise
        client.disconnect()


class TestOrchestratorClientMethods:
    """Tests for OrchestratorClient RPC methods."""

    def test_register_success(self) -> None:
        """Test register returns device on success."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.assigned_device = "cuda:0"
        mock_stub.Register.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        result = client.register(
            process_id="test-123",
            device="cuda:0",
        )

        assert result == "cuda:0"
        mock_stub.Register.assert_called_once()

    def test_register_failure(self) -> None:
        """Test register raises on rejection."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.message = "Registration rejected"
        mock_stub.Register.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        with pytest.raises(RuntimeError, match="Registration rejected"):
            client.register(
                process_id="test-123",
                device="cuda:0",
            )

    def test_unregister_success(self) -> None:
        """Test unregister returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_stub.Unregister.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        result = client.unregister("test-123")

        assert result is True
        mock_stub.Unregister.assert_called_once()

    def test_unregister_no_stub(self) -> None:
        """Test unregister handles no stub gracefully."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        client._stub = None

        result = client.unregister("test-123")

        assert result is False


class TestOrchestratorClientErrorRecovery:
    """Tests for error recovery methods."""

    def test_request_error_recovery_success(self) -> None:
        """Test request_error_recovery returns target on success."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.target_device = "cuda:1"
        mock_response.target_gpu_uuid = "GPU-12345678"
        mock_response.message = "Recovery target found"
        mock_stub.RequestErrorRecovery.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        result = client.request_error_recovery(
            process_id="test-123",
            error_type="OOM",
            current_device="cuda:0",
            memory_needed=2000000000,
        )

        assert result is not None
        assert result["target_device"] == "cuda:1"
        assert result["target_gpu_uuid"] == "GPU-12345678"

    def test_request_error_recovery_failure(self) -> None:
        """Test request_error_recovery returns None on failure."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_stub.RequestErrorRecovery.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        result = client.request_error_recovery(
            process_id="test-123",
            error_type="OOM",
            current_device="cuda:0",
        )

        assert result is None

    def test_request_error_recovery_no_stub(self) -> None:
        """Test request_error_recovery handles no stub."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        client._stub = None

        result = client.request_error_recovery(
            process_id="test-123",
            error_type="OOM",
            current_device="cuda:0",
        )

        assert result is None

    def test_request_error_recovery_local_mode(self) -> None:
        """Test request_error_recovery returns None in local mode."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient(address="localhost:50051")
        client._stub = MagicMock()
        client.connection_manager._state = ConnectionState.LOCAL_MODE

        result = client.request_error_recovery(
            process_id="test-123",
            error_type="OOM",
            current_device="cuda:0",
        )

        assert result is None


class TestConnectionManagerAdditional:
    """Additional tests for ConnectionManager class."""

    def test_start_connecting(self) -> None:
        """Test start_connecting sets state to CONNECTING."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm.start_connecting()

        assert cm._state == ConnectionState.CONNECTING

    def test_reset_for_reconnect(self) -> None:
        """Test reset_for_reconnect resets failure count."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state = ConnectionState.LOCAL_MODE
        cm._consecutive_failures = 5

        cm.reset_for_reconnect()

        assert cm._consecutive_failures == 0
        assert cm._state == ConnectionState.RECONNECTING

    def test_on_success_from_local_mode(self) -> None:
        """Test on_success logs reconnection from local mode."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state = ConnectionState.LOCAL_MODE
        cm._consecutive_failures = 5

        cm.on_success()

        assert cm._state == ConnectionState.CONNECTED
        assert cm._consecutive_failures == 0

    def test_on_failure_with_error(self) -> None:
        """Test on_failure handles error parameter."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager(max_retries=5)
        cm._state = ConnectionState.CONNECTED

        error = Exception("Test error")
        should_retry = cm.on_failure(error)

        assert should_retry is True
        assert cm._consecutive_failures == 1

    def test_set_state_no_callback(self) -> None:
        """Test _set_state works without callback."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        cm._state_change_callback = None

        # Should not raise
        cm._set_state(ConnectionState.CONNECTED)
        assert cm._state == ConnectionState.CONNECTED

    def test_set_state_callback_exception(self) -> None:
        """Test _set_state handles callback exception gracefully."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()

        def bad_callback(state: ConnectionState) -> None:
            raise RuntimeError("Callback error")

        cm.set_state_callback(bad_callback)

        # Should not raise
        cm._set_state(ConnectionState.CONNECTED)
        assert cm._state == ConnectionState.CONNECTED

    def test_set_state_same_state(self) -> None:
        """Test _set_state does not call callback if state unchanged."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager()
        callback_calls = []

        def callback(state: ConnectionState) -> None:
            callback_calls.append(state)

        cm.set_state_callback(callback)
        cm._state = ConnectionState.CONNECTED

        # Set to same state
        cm._set_state(ConnectionState.CONNECTED)

        # Callback should not be called
        assert len(callback_calls) == 0

    def test_should_attempt_reconnect_after_interval(self) -> None:
        """Test should_attempt_reconnect allows after interval passes."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        cm = ConnectionManager(reconnect_interval=0.0)  # Zero interval
        cm._state = ConnectionState.LOCAL_MODE
        cm._last_reconnect_attempt = time.time() - 1  # 1 second ago

        assert cm.should_attempt_reconnect() is True


class TestOrchestratorClientAddress:
    """Tests for address parsing."""

    def test_parse_address_simple(self) -> None:
        """Test parsing simple address."""
        from flexium.orchestrator.client import OrchestratorClient

        addr, workspace = OrchestratorClient._parse_address("localhost:50051")
        assert addr == "localhost:50051"
        assert workspace is None

    def test_parse_address_with_workspace(self) -> None:
        """Test parsing address with workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        addr, workspace = OrchestratorClient._parse_address("flexium.ai:80/myworkspace")
        assert addr == "flexium.ai:80"
        assert workspace == "myworkspace"

    def test_client_init_with_workspace(self) -> None:
        """Test client initialization with workspace in address."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="flexium.ai:80/testworkspace")
        assert client.address == "flexium.ai:80"
        assert client._workspace == "testworkspace"

    def test_get_grpc_metadata_with_workspace(self) -> None:
        """Test gRPC metadata includes workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="flexium.ai:80/myworkspace")
        metadata = client._get_grpc_metadata()

        assert metadata is not None
        assert ("workspace", "myworkspace") in metadata

    def test_get_grpc_metadata_no_workspace(self) -> None:
        """Test gRPC metadata is None without workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        metadata = client._get_grpc_metadata()

        assert metadata is None


class TestOrchestratorClientConnect:
    """Tests for connect/disconnect methods."""

    def test_connect_closes_existing_channel(self) -> None:
        """Test connect closes existing channel before creating new one."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        old_channel = MagicMock()
        client._channel = old_channel

        with patch("grpc.insecure_channel") as mock_channel:
            mock_channel.return_value = MagicMock()
            client.connect()

        old_channel.close.assert_called_once()

    def test_connect_closes_existing_channel_exception(self) -> None:
        """Test connect handles exception when closing existing channel."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        old_channel = MagicMock()
        old_channel.close.side_effect = Exception("Close error")
        client._channel = old_channel

        with patch("grpc.insecure_channel") as mock_channel:
            mock_channel.return_value = MagicMock()
            # Should not raise
            client.connect()

        assert client._channel is not None


class TestOrchestratorClientRPCMethods:
    """Tests for additional RPC methods."""

    def test_update_device(self) -> None:
        """Test update_device updates current device."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        client._current_device = "cuda:0"

        client.update_device("cuda:1")

        assert client._current_device == "cuda:1"

    def test_complete_migration_success(self) -> None:
        """Test complete_migration returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_stub.CompleteMigration.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
            gpu_uuid="GPU-12345678",
        )

        assert result is True
        mock_stub.CompleteMigration.assert_called_once()

    def test_complete_migration_failure(self) -> None:
        """Test complete_migration returns False on failure."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_stub.CompleteMigration.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert result is False

    def test_complete_migration_no_stub(self) -> None:
        """Test complete_migration returns False when not connected."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        client._stub = None

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert result is False

    def test_complete_migration_exception(self) -> None:
        """Test complete_migration handles exception gracefully."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.CompleteMigration.side_effect = Exception("RPC error")

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert result is False

    def test_unregister_with_stored_process_id(self) -> None:
        """Test unregister uses stored process_id."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_stub.Unregister.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()
        client._process_id = "stored-123"

        result = client.unregister()  # No process_id argument

        assert result is True
        mock_stub.Unregister.assert_called_once()

    def test_unregister_no_process_id(self) -> None:
        """Test unregister returns False with no process_id."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        client._stub = mock_stub
        client._process_id = None

        result = client.unregister()

        assert result is False


class TestOrchestratorClientPauseResume:
    """Tests for pause/resume methods."""

    def test_pause_success(self) -> None:
        """Test pause returns success dict."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.message = "Paused"
        mock_response.checkpoint_path = "/tmp/checkpoint"
        mock_stub.Pause.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.pause("test-123")

        assert result["success"] is True
        assert result["checkpoint_path"] == "/tmp/checkpoint"

    def test_pause_failure(self) -> None:
        """Test pause returns failure dict."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.message = "Process not found"
        mock_response.checkpoint_path = ""
        mock_stub.Pause.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.pause("test-123")

        assert result["success"] is False
        assert "checkpoint_path" not in result

    def test_pause_rpc_error(self) -> None:
        """Test pause handles RPC error."""
        from flexium.orchestrator.client import OrchestratorClient
        import grpc

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.Pause.side_effect = grpc.RpcError()

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.pause("test-123")

        assert result["success"] is False

    def test_resume_success(self) -> None:
        """Test resume returns success dict."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.message = "Resumed"
        mock_response.assigned_device = "cuda:1"
        mock_stub.Resume.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.resume("test-123", target_device="cuda:1")

        assert result["success"] is True
        assert result["assigned_device"] == "cuda:1"

    def test_resume_failure(self) -> None:
        """Test resume returns failure dict."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.message = "No GPU available"
        mock_response.assigned_device = ""
        mock_stub.Resume.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.resume("test-123")

        assert result["success"] is False
        assert "assigned_device" not in result

    def test_resume_rpc_error(self) -> None:
        """Test resume handles RPC error."""
        from flexium.orchestrator.client import OrchestratorClient
        import grpc

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.Resume.side_effect = grpc.RpcError()

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.resume("test-123")

        assert result["success"] is False


class TestOrchestratorClientRequestErrorRecoveryRPC:
    """Additional tests for request_error_recovery."""

    def test_request_error_recovery_rpc_error(self) -> None:
        """Test request_error_recovery handles RPC error."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState
        import grpc

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.RequestErrorRecovery.side_effect = grpc.RpcError()

        client._stub = mock_stub
        client._channel = MagicMock()
        client.connection_manager._state = ConnectionState.CONNECTED

        result = client.request_error_recovery(
            process_id="test-123",
            error_type="OOM",
            current_device="cuda:0",
        )

        assert result is None


class TestOrchestratorClientListAndStatus:
    """Tests for list and status methods."""

    def test_list_processes(self) -> None:
        """Test list_processes returns process list."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_process = MagicMock()
        mock_process.process_id = "test-123"
        mock_process.device = "cuda:0"
        mock_process.hostname = "testhost"
        mock_process.status = "RUNNING"
        mock_process.memory_allocated = 1000000
        mock_process.memory_reserved = 2000000
        mock_process.last_heartbeat = 1234567890.0
        mock_process.gpu_uuid = "GPU-12345678"
        mock_process.gpu_name = "Tesla V100"
        mock_process.min_gpus = 1
        mock_process.max_gpus = 1
        mock_process.max_vram = 0
        mock_process.can_share = True
        mock_process.priority = 50
        mock_process.preemptible = True
        mock_process.migratable = True

        mock_response = MagicMock()
        mock_response.processes = [mock_process]
        mock_stub.ListProcesses.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.list_processes()

        assert len(result) == 1
        assert result[0]["process_id"] == "test-123"
        assert result[0]["device"] == "cuda:0"

    def test_list_processes_with_filter(self) -> None:
        """Test list_processes with device filter."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.processes = []
        mock_stub.ListProcesses.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.list_processes(device_filter="cuda:0")

        assert result == []
        mock_stub.ListProcesses.assert_called_once()

    def test_get_process_status_found(self) -> None:
        """Test get_process_status when process is found."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_process = MagicMock()
        mock_process.process_id = "test-123"
        mock_process.device = "cuda:0"
        mock_process.hostname = "testhost"
        mock_process.status = "RUNNING"
        mock_process.memory_allocated = 1000000
        mock_process.memory_reserved = 2000000
        mock_process.last_heartbeat = 1234567890.0
        mock_process.gpu_uuid = "GPU-12345678"
        mock_process.gpu_name = "Tesla V100"
        mock_process.min_gpus = 1
        mock_process.max_gpus = 1
        mock_process.max_vram = 0
        mock_process.can_share = True
        mock_process.priority = 50
        mock_process.preemptible = True
        mock_process.migratable = True

        mock_response = MagicMock()
        mock_response.found = True
        mock_response.process = mock_process
        mock_stub.GetProcessStatus.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.get_process_status("test-123")

        assert result is not None
        assert result["process_id"] == "test-123"

    def test_get_process_status_not_found(self) -> None:
        """Test get_process_status when process is not found."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.found = False
        mock_stub.GetProcessStatus.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.get_process_status("nonexistent")

        assert result is None

    def test_request_migration_success(self) -> None:
        """Test request_migration returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_stub.Migrate.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.request_migration("test-123", "cuda:1")

        assert result is True

    def test_request_migration_failure(self) -> None:
        """Test request_migration returns False on failure."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.message = "No capacity"
        mock_stub.Migrate.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.request_migration("test-123", "cuda:1")

        assert result is False

    def test_get_device_status(self) -> None:
        """Test get_device_status returns device list."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_device = MagicMock()
        mock_device.device = "cuda:0"
        mock_device.process_count = 2
        mock_device.total_memory = 32000000000
        mock_device.used_memory = 16000000000
        mock_device.process_ids = ["p1", "p2"]

        mock_response = MagicMock()
        mock_response.devices = [mock_device]
        mock_stub.GetDeviceStatus.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.get_device_status()

        assert len(result) == 1
        assert result[0]["device"] == "cuda:0"
        assert result[0]["process_count"] == 2


class TestOrchestratorClientGPUHealth:
    """Tests for GPU health methods."""

    def test_mark_gpu_healthy_success(self) -> None:
        """Test mark_gpu_healthy returns True on success."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_stub.MarkGPUHealthy.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.mark_gpu_healthy("GPU-12345678")

        assert result is True

    def test_mark_gpu_healthy_failure(self) -> None:
        """Test mark_gpu_healthy returns False on failure."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_stub.MarkGPUHealthy.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.mark_gpu_healthy("GPU-12345678")

        assert result is False

    def test_mark_gpu_healthy_rpc_error(self) -> None:
        """Test mark_gpu_healthy handles RPC error."""
        from flexium.orchestrator.client import OrchestratorClient
        import grpc

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.MarkGPUHealthy.side_effect = grpc.RpcError()

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.mark_gpu_healthy("GPU-12345678")

        assert result is False

    def test_get_unhealthy_gpus(self) -> None:
        """Test get_unhealthy_gpus returns GPU list."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_gpu = MagicMock()
        mock_gpu.gpu_uuid = "GPU-12345678"
        mock_gpu.reason = "OOM"
        mock_gpu.marked_at = 1234567890.0
        mock_gpu.recovers_at = 1234567950.0

        mock_response = MagicMock()
        mock_response.gpus = [mock_gpu]
        mock_stub.GetUnhealthyGPUs.return_value = mock_response

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.get_unhealthy_gpus()

        assert len(result) == 1
        assert result[0]["gpu_uuid"] == "GPU-12345678"
        assert result[0]["reason"] == "OOM"

    def test_get_unhealthy_gpus_rpc_error(self) -> None:
        """Test get_unhealthy_gpus handles RPC error."""
        from flexium.orchestrator.client import OrchestratorClient
        import grpc

        client = OrchestratorClient(address="localhost:50051")
        mock_stub = MagicMock()
        mock_stub.GetUnhealthyGPUs.side_effect = grpc.RpcError()

        client._stub = mock_stub
        client._channel = MagicMock()

        result = client.get_unhealthy_gpus()

        assert result == []


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_heartbeat_interval(self) -> None:
        """Test DEFAULT_HEARTBEAT_INTERVAL is defined."""
        from flexium.orchestrator.client import DEFAULT_HEARTBEAT_INTERVAL

        assert DEFAULT_HEARTBEAT_INTERVAL > 0

    def test_timing_constants_used(self) -> None:
        """Test timing constants are imported correctly."""
        from flexium.orchestrator.client import (
            DEFAULT_BACKOFF_MULTIPLIER,
            DEFAULT_MAX_RETRIES,
            DEFAULT_RECONNECT_INTERVAL,
            DEFAULT_RETRY_DELAY,
        )

        assert DEFAULT_MAX_RETRIES > 0
        assert DEFAULT_RETRY_DELAY > 0
        assert DEFAULT_RECONNECT_INTERVAL > 0
        assert DEFAULT_BACKOFF_MULTIPLIER >= 1.0


class TestOrchestratorInit:
    """Tests for orchestrator module __init__."""

    def test_import_from_init(self) -> None:
        """Test classes can be imported from flexium.orchestrator."""
        from flexium.orchestrator import OrchestratorClient, ConnectionState, ConnectionManager

        assert OrchestratorClient is not None
        assert ConnectionState is not None
        assert ConnectionManager is not None

    def test_all_list(self) -> None:
        """Test __all__ contains expected exports."""
        import flexium.orchestrator

        assert "OrchestratorClient" in flexium.orchestrator.__all__
        assert "ConnectionState" in flexium.orchestrator.__all__
        assert "ConnectionManager" in flexium.orchestrator.__all__
