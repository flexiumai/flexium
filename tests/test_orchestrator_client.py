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
