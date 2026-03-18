"""Integration tests for orchestrator client using MockTransport.

These tests verify the client's behavior when communicating with an orchestrator.
MockTransport is used to simulate server behavior without network dependencies.

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

import pytest


# ============================================================================
# Mock Orchestrator - simulates server behavior
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

    This mock responds to events with realistic behavior:
    - Tracks registered processes
    - Returns migration directives via heartbeat
    - Handles pause/resume flows
    """

    def __init__(self):
        self.processes: Dict[str, MockProcess] = {}
        self._lock = threading.Lock()

    def handle_event(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an event from the client."""
        if event == "register":
            return self._handle_register(data)
        elif event == "heartbeat":
            return self._handle_heartbeat(data)
        elif event == "complete_migration":
            return self._handle_complete_migration(data)
        elif event == "complete_pause":
            return self._handle_complete_pause(data)
        elif event == "unregister":
            return self._handle_unregister(data)
        return {"success": False, "message": "Unknown event"}

    def _handle_register(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle registration request."""
        process_id = data.get("process_id", "")
        device = data.get("device", "cuda:0")
        hostname = data.get("hostname", "")

        with self._lock:
            self.processes[process_id] = MockProcess(
                process_id=process_id,
                device=device,
                hostname=hostname,
            )

        return {
            "success": True,
            "assigned_device": device,
            "message": "Registered successfully",
        }

    def _handle_heartbeat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heartbeat request."""
        process_id = data.get("process_id", "")

        with self._lock:
            if process_id not in self.processes:
                return {
                    "success": False,
                    "should_migrate": False,
                    "message": "Process not found",
                }

            process = self.processes[process_id]
            process.memory_reserved = data.get("memory_reserved", 0)

            # Check for pending migration
            if process.pending_migration:
                target = process.pending_migration
                return {
                    "success": True,
                    "should_migrate": True,
                    "target_device": target,
                }

            return {
                "success": True,
                "should_migrate": False,
            }

    def _handle_complete_migration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle migration completion."""
        process_id = data.get("process_id", "")
        new_device = data.get("new_device", "")

        with self._lock:
            if process_id in self.processes:
                self.processes[process_id].device = new_device
                self.processes[process_id].pending_migration = None

        return {"success": True}

    def _handle_complete_pause(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pause completion."""
        return {"success": True}

    def _handle_unregister(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unregistration."""
        process_id = data.get("process_id", "")

        with self._lock:
            if process_id in self.processes:
                del self.processes[process_id]
                return {"success": True}
            return {"success": False, "message": "Process not found"}

    def set_pending_migration(self, process_id: str, target_device: str) -> None:
        """Set a pending migration for testing."""
        with self._lock:
            if process_id in self.processes:
                self.processes[process_id].pending_migration = target_device


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    return MockOrchestrator()


@pytest.fixture
def client_with_mock_orchestrator(mock_orchestrator):
    """Create an OrchestratorClient with a MockTransport wired to MockOrchestrator."""
    from flexium.orchestrator.client import OrchestratorClient
    from flexium.orchestrator.transport import MockTransport

    transport = MockTransport(
        response_handler=mock_orchestrator.handle_event,
        server_url="mock://localhost/testworkspace",
    )
    client = OrchestratorClient(transport=transport)

    return client


# ============================================================================
# Integration Tests - Client Registration
# ============================================================================

class TestClientRegistration:
    """Tests for client registration flow."""

    def test_client_registers_on_startup(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test client registers with orchestrator."""
        result = client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        assert result == "cuda:0"
        assert "test-123" in mock_orchestrator.processes

    def test_client_registers_with_assigned_device(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test client receives assigned device."""
        result = client_with_mock_orchestrator.register(
            process_id="test-456",
            device="cuda:0",
        )

        assert result == "cuda:0"
        assert mock_orchestrator.processes["test-456"].device == "cuda:0"

    def test_client_unregisters_on_shutdown(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test client unregisters on shutdown."""
        # First register
        client_with_mock_orchestrator.register(
            process_id="test-789",
            device="cuda:0",
        )
        assert "test-789" in mock_orchestrator.processes

        # Then unregister
        result = client_with_mock_orchestrator.unregister("test-789")

        assert result is True
        assert "test-789" not in mock_orchestrator.processes


# ============================================================================
# Integration Tests - Client Heartbeat
# ============================================================================

class TestClientHeartbeat:
    """Tests for client heartbeat flow."""

    def test_heartbeat_reports_memory(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test heartbeat reports memory usage."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        client_with_mock_orchestrator.heartbeat(memory_reserved=1000000)

        assert mock_orchestrator.processes["test-123"].memory_reserved == 1000000

    def test_heartbeat_returns_no_migration_normally(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test heartbeat returns no migration directive normally."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        result = client_with_mock_orchestrator.heartbeat()

        assert result is not None
        assert result.get("should_migrate") is False

    def test_heartbeat_returns_migration_when_pending(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test heartbeat returns migration when pending."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        # Set pending migration
        mock_orchestrator.set_pending_migration("test-123", "cuda:1")

        result = client_with_mock_orchestrator.heartbeat()

        assert result is not None
        assert result.get("should_migrate") is True
        assert result.get("target_device") == "cuda:1"


# ============================================================================
# Integration Tests - Client Migration
# ============================================================================

class TestClientMigration:
    """Tests for client migration flow."""

    def test_client_completes_migration(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test client notifies migration completion."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        result = client_with_mock_orchestrator.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert result is True
        assert mock_orchestrator.processes["test-123"].device == "cuda:1"

    def test_migration_clears_pending_flag(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test migration completion clears pending flag."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )
        mock_orchestrator.set_pending_migration("test-123", "cuda:1")

        client_with_mock_orchestrator.complete_migration(
            process_id="test-123",
            new_device="cuda:1",
        )

        assert mock_orchestrator.processes["test-123"].pending_migration is None


# ============================================================================
# Integration Tests - Pause/Resume
# ============================================================================

class TestClientPauseResume:
    """Tests for client pause/resume flow."""

    def test_client_receives_pause_directive(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test client receives pause directive via heartbeat."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        # Set pending pause (using __PAUSE__ target)
        mock_orchestrator.set_pending_migration("test-123", "__PAUSE__")

        result = client_with_mock_orchestrator.heartbeat()

        assert result is not None
        assert result.get("should_migrate") is True
        assert result.get("target_device") == "__PAUSE__"


# ============================================================================
# Integration Tests - Concurrency
# ============================================================================

class TestClientConcurrency:
    """Tests for concurrent client operations."""

    def test_concurrent_heartbeats_are_safe(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test concurrent heartbeats don't cause race conditions."""
        client_with_mock_orchestrator.register(
            process_id="test-123",
            device="cuda:0",
        )

        results = []
        errors = []

        def send_heartbeat():
            try:
                result = client_with_mock_orchestrator.heartbeat(memory_reserved=1000000)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_heartbeat) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# ============================================================================
# Integration Tests - Error Handling
# ============================================================================

class TestClientErrorHandling:
    """Tests for client error handling."""

    def test_heartbeat_for_unknown_process(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test heartbeat for unknown process returns failure."""
        client_with_mock_orchestrator._process_id = "unknown-123"
        client_with_mock_orchestrator._transport.connect()

        result = client_with_mock_orchestrator.heartbeat()

        assert result is not None
        assert result.get("success") is False

    def test_unregister_unknown_process(self, client_with_mock_orchestrator, mock_orchestrator):
        """Test unregistering unknown process returns failure."""
        client_with_mock_orchestrator._transport.connect()
        result = client_with_mock_orchestrator.unregister("unknown-123")

        # Should return False for unknown process
        assert result is False


# ============================================================================
# Transport Interface Tests
# ============================================================================

class TestTransportInterface:
    """Tests verifying transport abstraction works correctly."""

    def test_client_works_with_mock_transport(self):
        """Test client works correctly with MockTransport."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        transport = MockTransport()
        client = OrchestratorClient(transport=transport)

        # Registration
        result = client.register(process_id="test-123", device="cuda:0")
        assert result == "cuda:0"

        # Heartbeat
        hb_result = client.heartbeat()
        assert hb_result["success"] is True

        # Complete migration
        mig_result = client.complete_migration("test-123", "cuda:1")
        assert mig_result is True

    def test_different_transport_same_client_interface(self):
        """Test client interface stays the same regardless of transport."""
        from flexium.orchestrator.client import OrchestratorClient
        from flexium.orchestrator.transport import MockTransport

        # Create multiple clients with different transports
        transport1 = MockTransport(server_url="mock://server1/ws")
        transport2 = MockTransport(server_url="mock://server2/ws")

        client1 = OrchestratorClient(transport=transport1)
        client2 = OrchestratorClient(transport=transport2)

        # Both clients should have the same interface
        for client in [client1, client2]:
            assert hasattr(client, "register")
            assert hasattr(client, "heartbeat")
            assert hasattr(client, "complete_migration")
            assert hasattr(client, "unregister")
            assert hasattr(client, "connect")
            assert hasattr(client, "disconnect")

    def test_transport_can_be_swapped(self):
        """Test transport can be swapped for different implementations."""
        from flexium.orchestrator.transport import Transport, MockTransport

        # Verify MockTransport implements Transport interface
        transport = MockTransport()
        assert isinstance(transport, Transport)

        # Verify it has all required methods
        assert callable(transport.connect)
        assert callable(transport.disconnect)
        assert callable(transport.is_connected)
        assert callable(transport.send)
        assert callable(transport.set_event_handler)
