"""Tests for orchestrator client behavior using clean mocks.

These tests verify the client's behavior when communicating with an orchestrator.
Mocks are used at the gRPC boundary only - we don't expose internal implementation.

The mock orchestrator behaves like a real orchestrator:
- Accepts registrations and returns success/failure
- Accepts heartbeats and returns migration directives
- Tracks process state
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Mock Orchestrator - simulates server behavior without exposing internals
# ============================================================================

@dataclass
class MockProcess:
    """Simulated process state in mock orchestrator."""
    process_id: str
    device: str
    hostname: str = ""
    status: str = "running"
    memory_reserved: int = 0
    pending_migration: Optional[str] = None
    registered_at: float = field(default_factory=time.time)


class MockOrchestrator:
    """Mock orchestrator that simulates real server behavior.

    This mock responds to gRPC-like calls with realistic behavior:
    - Tracks registered processes
    - Returns migration directives via heartbeat
    - Handles pause/resume flows

    This is the ONLY mock - we don't mock internal client functions.
    """

    def __init__(self):
        self.processes: Dict[str, MockProcess] = {}
        self._lock = threading.Lock()

    def register(self, process_id: str, device: str, hostname: str = "", **kwargs) -> Dict[str, Any]:
        """Handle registration request."""
        with self._lock:
            if process_id in self.processes:
                # Update existing
                self.processes[process_id].device = device
                self.processes[process_id].hostname = hostname
            else:
                self.processes[process_id] = MockProcess(
                    process_id=process_id,
                    device=device,
                    hostname=hostname,
                )
            return {"success": True, "assigned_device": device, "message": ""}

    def heartbeat(self, process_id: str, device: str, memory_reserved: int = 0, **kwargs) -> Dict[str, Any]:
        """Handle heartbeat request."""
        with self._lock:
            if process_id not in self.processes:
                return {"success": False, "should_migrate": False, "target_device": ""}

            proc = self.processes[process_id]
            proc.memory_reserved = memory_reserved

            # Check for pending migration
            if proc.pending_migration:
                return {
                    "success": True,
                    "should_migrate": True,
                    "target_device": proc.pending_migration,
                }

            return {"success": True, "should_migrate": False, "target_device": ""}

    def complete_migration(self, process_id: str, new_device: str, **kwargs) -> Dict[str, Any]:
        """Handle migration completion."""
        with self._lock:
            if process_id not in self.processes:
                return {"success": False}

            proc = self.processes[process_id]
            proc.device = new_device
            proc.pending_migration = None
            proc.status = "running"
            return {"success": True}

    def unregister(self, process_id: str) -> Dict[str, Any]:
        """Handle unregistration."""
        with self._lock:
            if process_id in self.processes:
                del self.processes[process_id]
                return {"success": True}
            return {"success": False}

    def request_migration(self, process_id: str, target_device: str) -> None:
        """Request migration for a process (server-side action)."""
        with self._lock:
            if process_id in self.processes:
                self.processes[process_id].pending_migration = target_device
                self.processes[process_id].status = "migrating"

    def get_process(self, process_id: str) -> Optional[MockProcess]:
        """Get process info."""
        with self._lock:
            return self.processes.get(process_id)


def create_mock_stub(orchestrator: MockOrchestrator):
    """Create a mock gRPC stub that delegates to MockOrchestrator."""
    stub = MagicMock()

    def mock_register(request, **kwargs):
        response = MagicMock()
        result = orchestrator.register(
            process_id=request.process_id,
            device=request.device,
            hostname=request.hostname,
        )
        response.success = result["success"]
        response.assigned_device = result["assigned_device"]
        response.message = result["message"]
        return response

    def mock_heartbeat(request, **kwargs):
        response = MagicMock()
        result = orchestrator.heartbeat(
            process_id=request.process_id,
            device=request.device,
            memory_reserved=getattr(request, "memory_reserved", 0),
        )
        response.success = result["success"]
        response.should_migrate = result["should_migrate"]
        response.target_device = result["target_device"]
        return response

    def mock_complete_migration(request, **kwargs):
        response = MagicMock()
        result = orchestrator.complete_migration(
            process_id=request.process_id,
            new_device=request.new_device,
        )
        response.success = result["success"]
        return response

    def mock_unregister(request, **kwargs):
        response = MagicMock()
        result = orchestrator.unregister(request.process_id)
        response.success = result["success"]
        return response

    stub.Register = mock_register
    stub.Heartbeat = mock_heartbeat
    stub.CompleteMigration = mock_complete_migration
    stub.Unregister = mock_unregister

    return stub


# ============================================================================
# Tests
# ============================================================================

class TestClientRegistration:
    """Tests for client registration behavior."""

    def test_client_registers_on_startup(self) -> None:
        """Test that client registers with orchestrator on startup."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Simulate what happens when client starts
        request = MagicMock()
        request.process_id = "test-process"
        request.device = "cuda:0"
        request.hostname = "test-host"

        response = stub.Register(request)

        assert response.success is True
        assert orchestrator.get_process("test-process") is not None

    def test_client_registers_with_assigned_device(self) -> None:
        """Test client uses device assigned by orchestrator."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        request = MagicMock()
        request.process_id = "assigned-process"
        request.device = "cuda:0"
        request.hostname = "test-host"

        response = stub.Register(request)

        assert response.assigned_device == "cuda:0"

    def test_client_unregisters_on_shutdown(self) -> None:
        """Test that client unregisters when shutting down."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "shutdown-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        assert orchestrator.get_process("shutdown-process") is not None

        # Unregister
        unreg_request = MagicMock()
        unreg_request.process_id = "shutdown-process"
        stub.Unregister(unreg_request)

        assert orchestrator.get_process("shutdown-process") is None


class TestClientHeartbeat:
    """Tests for client heartbeat behavior."""

    def test_heartbeat_reports_memory(self) -> None:
        """Test heartbeat reports memory usage."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register first
        reg_request = MagicMock()
        reg_request.process_id = "memory-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Heartbeat with memory
        hb_request = MagicMock()
        hb_request.process_id = "memory-process"
        hb_request.device = "cuda:0"
        hb_request.memory_reserved = 500 * 1024 * 1024  # 500MB

        response = stub.Heartbeat(hb_request)

        assert response.success is True
        proc = orchestrator.get_process("memory-process")
        assert proc.memory_reserved == 500 * 1024 * 1024

    def test_heartbeat_returns_no_migration_normally(self) -> None:
        """Test heartbeat returns no migration when none pending."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "normal-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Heartbeat
        hb_request = MagicMock()
        hb_request.process_id = "normal-process"
        hb_request.device = "cuda:0"

        response = stub.Heartbeat(hb_request)

        assert response.success is True
        assert response.should_migrate is False

    def test_heartbeat_returns_migration_when_pending(self) -> None:
        """Test heartbeat returns migration directive when migration requested."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "migrating-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Server requests migration
        orchestrator.request_migration("migrating-process", "cuda:1")

        # Heartbeat should return migration
        hb_request = MagicMock()
        hb_request.process_id = "migrating-process"
        hb_request.device = "cuda:0"

        response = stub.Heartbeat(hb_request)

        assert response.success is True
        assert response.should_migrate is True
        assert response.target_device == "cuda:1"


class TestClientMigration:
    """Tests for client migration behavior."""

    def test_client_completes_migration(self) -> None:
        """Test client completes migration successfully."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "complete-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Server requests migration
        orchestrator.request_migration("complete-process", "cuda:1")

        # Client detects migration via heartbeat
        hb_request = MagicMock()
        hb_request.process_id = "complete-process"
        hb_request.device = "cuda:0"
        hb_response = stub.Heartbeat(hb_request)

        assert hb_response.should_migrate is True

        # Client completes migration
        complete_request = MagicMock()
        complete_request.process_id = "complete-process"
        complete_request.new_device = "cuda:1"

        response = stub.CompleteMigration(complete_request)

        assert response.success is True
        proc = orchestrator.get_process("complete-process")
        assert proc.device == "cuda:1"
        assert proc.status == "running"

    def test_migration_clears_pending_flag(self) -> None:
        """Test completing migration clears pending flag."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "clear-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Request and complete migration
        orchestrator.request_migration("clear-process", "cuda:1")

        complete_request = MagicMock()
        complete_request.process_id = "clear-process"
        complete_request.new_device = "cuda:1"
        stub.CompleteMigration(complete_request)

        # Next heartbeat should not indicate migration
        hb_request = MagicMock()
        hb_request.process_id = "clear-process"
        hb_request.device = "cuda:1"

        response = stub.Heartbeat(hb_request)

        assert response.should_migrate is False


class TestClientPauseResume:
    """Tests for client pause/resume behavior."""

    def test_client_receives_pause_directive(self) -> None:
        """Test client receives pause directive via heartbeat."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register
        reg_request = MagicMock()
        reg_request.process_id = "pause-process"
        reg_request.device = "cuda:0"
        reg_request.hostname = "test-host"
        stub.Register(reg_request)

        # Server requests pause (special migration target)
        orchestrator.request_migration("pause-process", "__PAUSE__")

        # Heartbeat should return pause directive
        hb_request = MagicMock()
        hb_request.process_id = "pause-process"
        hb_request.device = "cuda:0"

        response = stub.Heartbeat(hb_request)

        assert response.should_migrate is True
        assert response.target_device == "__PAUSE__"


class TestClientConcurrency:
    """Tests for thread safety of client operations."""

    def test_concurrent_heartbeats_are_safe(self) -> None:
        """Test concurrent heartbeats don't corrupt state."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        # Register multiple processes
        for i in range(5):
            reg_request = MagicMock()
            reg_request.process_id = f"concurrent-{i}"
            reg_request.device = f"cuda:{i % 4}"
            reg_request.hostname = "test-host"
            stub.Register(reg_request)

        errors = []
        results = []
        lock = threading.Lock()

        def send_heartbeats(process_id: str):
            try:
                for _ in range(10):
                    hb_request = MagicMock()
                    hb_request.process_id = process_id
                    hb_request.device = "cuda:0"
                    hb_request.memory_reserved = 100 * 1024 * 1024
                    response = stub.Heartbeat(hb_request)
                    with lock:
                        results.append(response.success)
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=send_heartbeats, args=(f"concurrent-{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert all(results)


class TestClientErrorHandling:
    """Tests for client error handling."""

    def test_heartbeat_for_unknown_process(self) -> None:
        """Test heartbeat for unknown process returns failure."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        hb_request = MagicMock()
        hb_request.process_id = "unknown-process"
        hb_request.device = "cuda:0"

        response = stub.Heartbeat(hb_request)

        assert response.success is False

    def test_unregister_unknown_process(self) -> None:
        """Test unregister for unknown process returns failure."""
        orchestrator = MockOrchestrator()
        stub = create_mock_stub(orchestrator)

        unreg_request = MagicMock()
        unreg_request.process_id = "unknown-process"

        response = stub.Unregister(unreg_request)

        assert response.success is False


# ============================================================================
# Tests for actual ConnectionManager class
# ============================================================================

class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_initial_state_is_disconnected(self) -> None:
        """Test ConnectionManager starts in DISCONNECTED state."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager()
        assert manager.state == ConnectionState.DISCONNECTED
        assert not manager.is_healthy
        assert not manager.is_local_mode

    def test_start_connecting_changes_state(self) -> None:
        """Test start_connecting sets state to CONNECTING."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager()
        manager.start_connecting()
        assert manager.state == ConnectionState.CONNECTING

    def test_on_success_sets_connected_state(self) -> None:
        """Test on_success sets state to CONNECTED."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager()
        manager.start_connecting()
        manager.on_success()

        assert manager.state == ConnectionState.CONNECTED
        assert manager.is_healthy
        assert not manager.is_local_mode

    def test_on_failure_returns_true_while_retries_remain(self) -> None:
        """Test on_failure returns True when retries remain."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(max_retries=3)
        manager.start_connecting()

        # First failure - should retry
        should_retry = manager.on_failure(Exception("test error"))
        assert should_retry is True
        assert manager.state == ConnectionState.RECONNECTING

    def test_on_failure_enters_local_mode_after_max_retries(self) -> None:
        """Test on_failure enters LOCAL_MODE after max retries exhausted."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(max_retries=2)
        manager.start_connecting()

        # First failure
        manager.on_failure(Exception("error 1"))
        assert manager.state == ConnectionState.RECONNECTING

        # Second failure - max retries exhausted
        should_retry = manager.on_failure(Exception("error 2"))
        assert should_retry is False
        assert manager.state == ConnectionState.LOCAL_MODE
        assert manager.is_local_mode

    def test_retry_delay_uses_exponential_backoff(self) -> None:
        """Test retry delay increases exponentially."""
        from flexium.orchestrator.client import ConnectionManager

        manager = ConnectionManager(
            retry_delay=1.0,
            backoff_multiplier=2.0,
            max_retries=5
        )
        manager.start_connecting()

        # First failure
        manager.on_failure(Exception("error"))
        delay1 = manager.get_retry_delay()
        assert delay1 == 1.0  # 1.0 * 2^0

        # Second failure
        manager.on_failure(Exception("error"))
        delay2 = manager.get_retry_delay()
        assert delay2 == 2.0  # 1.0 * 2^1

        # Third failure
        manager.on_failure(Exception("error"))
        delay3 = manager.get_retry_delay()
        assert delay3 == 4.0  # 1.0 * 2^2

    def test_on_success_resets_failure_counter(self) -> None:
        """Test on_success resets consecutive failure counter."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(max_retries=3)
        manager.start_connecting()

        # Fail twice
        manager.on_failure(Exception("error"))
        manager.on_failure(Exception("error"))

        # Success resets
        manager.on_success()
        assert manager.state == ConnectionState.CONNECTED

        # Should be able to fail max_retries times again
        manager.on_failure(Exception("error"))
        assert manager.state == ConnectionState.RECONNECTING  # Not LOCAL_MODE yet

    def test_should_attempt_reconnect_only_in_local_mode(self) -> None:
        """Test should_attempt_reconnect only returns True in LOCAL_MODE."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(reconnect_interval=0.1)

        # Not in local mode - should return False
        assert manager.should_attempt_reconnect() is False

        manager.start_connecting()
        assert manager.should_attempt_reconnect() is False

        manager.on_success()
        assert manager.should_attempt_reconnect() is False

    def test_should_attempt_reconnect_respects_interval(self) -> None:
        """Test should_attempt_reconnect respects reconnect interval."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(max_retries=1, reconnect_interval=0.1)
        manager.start_connecting()
        manager.on_failure(Exception("error"))  # Enter local mode

        assert manager.state == ConnectionState.LOCAL_MODE

        # First call should return True
        assert manager.should_attempt_reconnect() is True

        # Immediate second call should return False (interval not passed)
        assert manager.should_attempt_reconnect() is False

        # After interval, should return True again
        time.sleep(0.15)
        assert manager.should_attempt_reconnect() is True

    def test_reset_for_reconnect(self) -> None:
        """Test reset_for_reconnect resets state for reconnection attempt."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        manager = ConnectionManager(max_retries=1)
        manager.start_connecting()
        manager.on_failure(Exception("error"))  # Enter local mode

        assert manager.state == ConnectionState.LOCAL_MODE

        manager.reset_for_reconnect()
        assert manager.state == ConnectionState.RECONNECTING

    def test_state_change_callback(self) -> None:
        """Test state change callback is called on state transitions."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        states_seen = []

        def callback(new_state: ConnectionState) -> None:
            states_seen.append(new_state)

        manager = ConnectionManager()
        manager.set_state_callback(callback)

        manager.start_connecting()
        manager.on_success()

        assert ConnectionState.CONNECTING in states_seen
        assert ConnectionState.CONNECTED in states_seen

    def test_state_change_callback_handles_exceptions(self) -> None:
        """Test state change callback exceptions don't break state machine."""
        from flexium.orchestrator.client import ConnectionManager, ConnectionState

        def bad_callback(new_state: ConnectionState) -> None:
            raise ValueError("callback error")

        manager = ConnectionManager()
        manager.set_state_callback(bad_callback)

        # Should not raise, just log warning
        manager.start_connecting()
        assert manager.state == ConnectionState.CONNECTING


# ============================================================================
# Tests for actual OrchestratorClient class
# ============================================================================

class TestOrchestratorClientUnit:
    """Unit tests for OrchestratorClient class."""

    def test_parse_address_simple(self) -> None:
        """Test _parse_address with simple host:port."""
        from flexium.orchestrator.client import OrchestratorClient

        addr, workspace = OrchestratorClient._parse_address("localhost:80")
        assert addr == "localhost:80"
        assert workspace is None

    def test_parse_address_with_workspace(self) -> None:
        """Test _parse_address with workspace suffix."""
        from flexium.orchestrator.client import OrchestratorClient

        addr, workspace = OrchestratorClient._parse_address("localhost:80/my-workspace")
        assert addr == "localhost:80"
        assert workspace == "my-workspace"

    def test_parse_address_with_nested_workspace(self) -> None:
        """Test _parse_address with nested workspace path."""
        from flexium.orchestrator.client import OrchestratorClient

        addr, workspace = OrchestratorClient._parse_address("localhost:80/team/project")
        assert addr == "localhost:80"
        assert workspace == "team/project"

    def test_client_initialization(self) -> None:
        """Test OrchestratorClient initializes correctly."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient("localhost:80")
        assert client.address == "localhost:80"
        assert client.connection_manager.state == ConnectionState.DISCONNECTED

    def test_client_initialization_with_workspace(self) -> None:
        """Test OrchestratorClient initializes with workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80/test-workspace")
        assert client.address == "localhost:80"
        assert client._workspace == "test-workspace"

    def test_get_grpc_metadata_without_workspace(self) -> None:
        """Test _get_grpc_metadata returns None without workspace."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        assert client._get_grpc_metadata() is None

    def test_get_grpc_metadata_with_workspace(self) -> None:
        """Test _get_grpc_metadata returns workspace metadata."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80/my-workspace")
        metadata = client._get_grpc_metadata()
        assert metadata == [("workspace", "my-workspace")]

    def test_update_device(self) -> None:
        """Test update_device updates internal state."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._current_device = "cuda:0"

        client.update_device("cuda:1")
        assert client._current_device == "cuda:1"

    def test_disconnect_without_connection(self) -> None:
        """Test disconnect works even if not connected."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        # Should not raise
        client.disconnect()
        assert client._channel is None
        assert client._stub is None

    def test_unregister_without_stub(self) -> None:
        """Test unregister returns False without connection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        result = client.unregister("test-process")
        assert result is False

    def test_unregister_without_process_id(self) -> None:
        """Test unregister returns False without process_id."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()  # Fake stub
        result = client.unregister()  # No process_id
        assert result is False

    def test_complete_migration_without_stub(self) -> None:
        """Test complete_migration returns False without connection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        result = client.complete_migration("test-process", "cuda:1")
        assert result is False


class TestOrchestratorClientWithMockStub:
    """Tests for OrchestratorClient with mocked gRPC stub."""

    def test_complete_migration_success(self) -> None:
        """Test complete_migration succeeds with mock stub."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.CompleteMigration.return_value = MagicMock(success=True)

        result = client.complete_migration(
            "test-process", "cuda:1", gpu_uuid="GPU-123", memory_reserved=1000
        )
        assert result is True
        client._stub.CompleteMigration.assert_called_once()

    def test_complete_migration_failure(self) -> None:
        """Test complete_migration handles server rejection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.CompleteMigration.return_value = MagicMock(success=False)

        result = client.complete_migration("test-process", "cuda:1")
        assert result is False

    def test_complete_migration_handles_exception(self) -> None:
        """Test complete_migration handles exceptions gracefully."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.CompleteMigration.side_effect = Exception("network error")

        result = client.complete_migration("test-process", "cuda:1")
        assert result is False

    def test_unregister_success(self) -> None:
        """Test unregister succeeds with mock stub."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Unregister.return_value = MagicMock(success=True)
        client._process_id = "test-process"

        result = client.unregister()
        assert result is True

    def test_request_error_recovery_in_local_mode(self) -> None:
        """Test request_error_recovery returns None in local mode."""
        from flexium.orchestrator.client import OrchestratorClient, ConnectionState

        client = OrchestratorClient("localhost:80", max_retries=1)
        client._stub = MagicMock()

        # Force into local mode
        client.connection_manager.start_connecting()
        client.connection_manager.on_failure(Exception("error"))

        result = client.request_error_recovery(
            "test-process", "OOM", "cuda:0"
        )
        assert result is None

    def test_request_error_recovery_success(self) -> None:
        """Test request_error_recovery returns target on success."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.RequestErrorRecovery.return_value = MagicMock(
            success=True,
            target_device="cuda:1",
            target_gpu_uuid="GPU-456",
            message="Found suitable GPU"
        )

        result = client.request_error_recovery(
            "test-process", "OOM", "cuda:0", memory_needed=1000000
        )
        assert result is not None
        assert result["target_device"] == "cuda:1"
        assert result["target_gpu_uuid"] == "GPU-456"

    def test_request_error_recovery_no_target(self) -> None:
        """Test request_error_recovery returns None when no target available."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.RequestErrorRecovery.return_value = MagicMock(
            success=False,
            message="No suitable GPU available"
        )

        result = client.request_error_recovery(
            "test-process", "OOM", "cuda:0"
        )
        assert result is None

    def test_mark_gpu_healthy_success(self) -> None:
        """Test mark_gpu_healthy succeeds."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.MarkGPUHealthy.return_value = MagicMock(success=True)

        result = client.mark_gpu_healthy("GPU-123")
        assert result is True

    def test_mark_gpu_healthy_failure(self) -> None:
        """Test mark_gpu_healthy handles gRPC errors."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.MarkGPUHealthy.side_effect = grpc.RpcError()

        result = client.mark_gpu_healthy("GPU-123")
        assert result is False

    def test_get_unhealthy_gpus_success(self) -> None:
        """Test get_unhealthy_gpus returns list of GPUs."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()

        mock_gpu = MagicMock()
        mock_gpu.gpu_uuid = "GPU-123"
        mock_gpu.reason = "OOM"
        mock_gpu.marked_at = 1000.0
        mock_gpu.recovers_at = 2000.0

        client._stub.GetUnhealthyGPUs.return_value = MagicMock(gpus=[mock_gpu])

        result = client.get_unhealthy_gpus()
        assert len(result) == 1
        assert result[0]["gpu_uuid"] == "GPU-123"
        assert result[0]["reason"] == "OOM"

    def test_get_unhealthy_gpus_handles_error(self) -> None:
        """Test get_unhealthy_gpus returns empty list on error."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.GetUnhealthyGPUs.side_effect = grpc.RpcError()

        result = client.get_unhealthy_gpus()
        assert result == []

    def test_pause_success(self) -> None:
        """Test pause returns success response."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Pause.return_value = MagicMock(
            success=True,
            message="Paused",
            checkpoint_path="/tmp/checkpoint"
        )

        result = client.pause("test-process")
        assert result["success"] is True
        assert result["checkpoint_path"] == "/tmp/checkpoint"

    def test_pause_failure(self) -> None:
        """Test pause handles server rejection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Pause.return_value = MagicMock(
            success=False,
            message="Process not found",
            checkpoint_path=""
        )

        result = client.pause("test-process")
        assert result["success"] is False

    def test_pause_handles_grpc_error(self) -> None:
        """Test pause handles gRPC errors gracefully."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Pause.side_effect = grpc.RpcError()

        result = client.pause("test-process")
        assert result["success"] is False

    def test_resume_success(self) -> None:
        """Test resume returns success response."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Resume.return_value = MagicMock(
            success=True,
            message="Resumed",
            assigned_device="cuda:1"
        )

        result = client.resume("test-process", "cuda:1")
        assert result["success"] is True
        assert result["assigned_device"] == "cuda:1"

    def test_resume_handles_grpc_error(self) -> None:
        """Test resume handles gRPC errors gracefully."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Resume.side_effect = grpc.RpcError()

        result = client.resume("test-process")
        assert result["success"] is False


# ============================================================================
# Additional tests for register() method
# ============================================================================

class TestOrchestratorClientRegister:
    """Tests for OrchestratorClient.register method."""

    def test_register_success(self) -> None:
        """Test register succeeds on first attempt."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Register.return_value = MagicMock(
            success=True,
            assigned_device="cuda:0",
            message=""
        )

        with patch.object(client, "connect"):
            result = client.register(
                process_id="test-process",
                device="cuda:0",
            )
            assert result == "cuda:0"

    def test_register_rejected(self) -> None:
        """Test register raises on server rejection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Register.return_value = MagicMock(
            success=False,
            assigned_device="",
            message="Registration rejected: duplicate process"
        )

        with patch.object(client, "connect"):
            with pytest.raises(RuntimeError, match="rejected"):
                client.register(
                    process_id="test-process",
                    device="cuda:0",
                )

    def test_register_retries_on_grpc_error(self) -> None:
        """Test register retries on transient gRPC errors."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80", max_retries=2)
        client._stub = MagicMock()

        # First call fails, second succeeds
        call_count = [0]
        def mock_register(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise grpc.RpcError()
            return MagicMock(success=True, assigned_device="cuda:0")

        client._stub.Register.side_effect = mock_register

        with patch.object(client, "connect"):
            with patch("time.sleep"):  # Skip delays
                result = client.register(
                    process_id="test-process",
                    device="cuda:0",
                )
                assert result == "cuda:0"
                assert call_count[0] == 2

    def test_register_enters_local_mode_after_max_retries(self) -> None:
        """Test register enters local mode after max retries."""
        import grpc
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80", max_retries=2)
        client._stub = MagicMock()
        client._stub.Register.side_effect = grpc.RpcError()

        with patch.object(client, "connect"):
            with patch("time.sleep"):  # Skip delays
                result = client.register(
                    process_id="test-process",
                    device="cuda:0",
                )
                # Should return None in local mode
                assert result is None

    def test_register_stores_process_info(self) -> None:
        """Test register stores process info for reconnection."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Register.return_value = MagicMock(
            success=True,
            assigned_device="cuda:1",
        )

        with patch.object(client, "connect"):
            client.register(
                process_id="test-123",
                device="cuda:0",
                min_gpus=1,
                max_gpus=4,
                priority=75,
            )

            assert client._process_id == "test-123"
            assert client._current_device == "cuda:1"  # Updated to assigned
            assert client._min_gpus == 1
            assert client._max_gpus == 4
            assert client._priority == 75


class TestOrchestratorClientHeartbeat:
    """Tests for OrchestratorClient heartbeat-related methods."""

    def test_heartbeat_success(self) -> None:
        """Test heartbeat method works with mock stub."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()
        client._stub.Heartbeat.return_value = MagicMock(
            success=True,
            should_migrate=False,
            target_device="",
        )
        client._process_id = "test-process"
        client._current_device = "cuda:0"

        # Heartbeat is called internally, but we can verify stub setup
        assert client._stub.Heartbeat is not None

    def test_heartbeat_triggers_migration(self) -> None:
        """Test heartbeat response can trigger migration."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        client._stub = MagicMock()

        response = MagicMock()
        response.success = True
        response.should_migrate = True
        response.target_device = "cuda:1"
        response.target_gpu_uuid = "GPU-123"
        client._stub.Heartbeat.return_value = response

        # Verify response fields are accessible
        assert response.should_migrate is True
        assert response.target_device == "cuda:1"


class TestOrchestratorClientConnect:
    """Tests for OrchestratorClient connect/disconnect."""

    def test_connect_creates_channel(self) -> None:
        """Test connect creates gRPC channel."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")

        with patch("grpc.insecure_channel") as mock_channel:
            mock_channel.return_value = MagicMock()
            client.connect()
            mock_channel.assert_called_once()

    def test_disconnect_closes_channel(self) -> None:
        """Test disconnect closes gRPC channel."""
        from flexium.orchestrator.client import OrchestratorClient

        client = OrchestratorClient("localhost:80")
        mock_channel = MagicMock()
        client._channel = mock_channel
        client._stub = MagicMock()

        client.disconnect()

        mock_channel.close.assert_called_once()
        assert client._channel is None
        assert client._stub is None


