"""Orchestrator client for flexium.

Provides a client for connecting to the orchestrator server from
training processes. Includes graceful degradation support for when
the orchestrator becomes unreachable.
"""

from __future__ import annotations

import socket
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import grpc

from flexium.proto import orchestrator_pb2 as pb2
from flexium.proto import orchestrator_pb2_grpc as pb2_grpc
from flexium.timing import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECONNECT_INTERVAL,
    DEFAULT_RETRY_DELAY,
    HEARTBEAT_INTERVAL,
)
from flexium.utils.logging import get_logger

logger = get_logger(__name__)

# Default heartbeat interval in seconds (aliased from timing module)
DEFAULT_HEARTBEAT_INTERVAL = HEARTBEAT_INTERVAL


class ConnectionState(Enum):
    """Connection state for orchestrator client.

    Attributes:
        DISCONNECTED: Never connected or explicitly disconnected.
        CONNECTING: Attempting initial connection.
        CONNECTED: Successfully connected and healthy.
        RECONNECTING: Lost connection, attempting to reconnect.
        LOCAL_MODE: Given up on connection, running standalone.
    """

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    LOCAL_MODE = auto()


class ConnectionManager:
    """Manages connection state and retry logic for orchestrator client.

    Implements graceful degradation: if the orchestrator becomes unreachable,
    training continues in "local mode" with migration disabled. Periodic
    reconnection attempts allow recovery when the orchestrator comes back.

    Attributes:
        state: Current connection state.
        is_healthy: Whether connection is healthy (CONNECTED state).
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    ) -> None:
        """Initialize connection manager.

        Parameters:
            max_retries: Maximum retry attempts before switching to local mode.
            retry_delay: Initial delay between retries in seconds.
            reconnect_interval: Interval for reconnection attempts in local mode.
            backoff_multiplier: Multiplier for exponential backoff.
        """
        self._state = ConnectionState.DISCONNECTED
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._reconnect_interval = reconnect_interval
        self._backoff_multiplier = backoff_multiplier
        self._consecutive_failures = 0
        self._last_success_time: Optional[float] = None
        self._last_reconnect_attempt: Optional[float] = None
        self._state_change_callback: Optional[Callable[[ConnectionState], None]] = None

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self._state == ConnectionState.CONNECTED

    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode (orchestrator unavailable)."""
        return self._state == ConnectionState.LOCAL_MODE

    def set_state_callback(
        self, callback: Optional[Callable[[ConnectionState], None]]
    ) -> None:
        """Set callback for state changes.

        Parameters:
            callback: Function to call on state changes, receives new state.
        """
        self._state_change_callback = callback

    def start_connecting(self) -> None:
        """Mark that a connection attempt is starting."""
        self._set_state(ConnectionState.CONNECTING)

    def on_success(self) -> None:
        """Called when an RPC succeeds. Updates state to CONNECTED."""
        was_disconnected = self._state in (
            ConnectionState.RECONNECTING,
            ConnectionState.LOCAL_MODE,
        )

        self._consecutive_failures = 0
        self._last_success_time = time.time()

        if was_disconnected:
            logger.info("[flexium] Orchestrator reconnected!")
            logger.info("[flexium] Migration re-enabled")

        self._set_state(ConnectionState.CONNECTED)

    def on_failure(self, error: Optional[Exception] = None) -> bool:
        """Called when an RPC fails.

        Parameters:
            error: The exception that caused the failure.

        Returns:
            True if should retry, False if should give up.
        """
        self._consecutive_failures += 1
        error_msg = str(error) if error else "unknown error"

        if self._consecutive_failures < self._max_retries:
            # Still have retries left
            self._set_state(ConnectionState.RECONNECTING)
            delay = self._get_retry_delay()
            logger.warning(
                f"[flexium] Connection failed: {error_msg}"
            )
            logger.warning(
                f"[flexium] Retrying in {delay:.1f}s "
                f"(attempt {self._consecutive_failures}/{self._max_retries})"
            )
            return True
        else:
            # Max retries exhausted
            self._enter_local_mode()
            return False

    def _get_retry_delay(self) -> float:
        """Calculate retry delay with exponential backoff."""
        return self._retry_delay * (
            self._backoff_multiplier ** (self._consecutive_failures - 1)
        )

    def get_retry_delay(self) -> float:
        """Get the current retry delay."""
        return self._get_retry_delay()

    def _enter_local_mode(self) -> None:
        """Switch to local mode after max retries exhausted."""
        if self._state != ConnectionState.LOCAL_MODE:
            logger.warning(
                "[flexium] Switching to local mode (migration disabled)"
            )
            logger.info(
                "[flexium] Training continues - "
                "will attempt reconnection periodically"
            )
            self._set_state(ConnectionState.LOCAL_MODE)

    def should_attempt_reconnect(self) -> bool:
        """Check if we should try reconnecting in local mode.

        Returns:
            True if enough time has passed since last reconnect attempt.
        """
        if self._state != ConnectionState.LOCAL_MODE:
            return False

        now = time.time()
        if self._last_reconnect_attempt is None:
            self._last_reconnect_attempt = now
            return True

        if now - self._last_reconnect_attempt >= self._reconnect_interval:
            self._last_reconnect_attempt = now
            return True

        return False

    def reset_for_reconnect(self) -> None:
        """Reset retry counter for a reconnection attempt."""
        self._consecutive_failures = 0
        self._set_state(ConnectionState.RECONNECTING)

    def _set_state(self, new_state: ConnectionState) -> None:
        """Set state and notify callback if changed."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.debug(f"Connection state: {old_state.name} -> {new_state.name}")
            if self._state_change_callback is not None:
                try:
                    self._state_change_callback(new_state)
                except Exception as e:
                    logger.warning(f"State change callback failed: {e}")


class OrchestratorClient:
    """Client for communicating with the orchestrator server.

    Handles registration, heartbeats, and migration coordination.
    Supports graceful degradation when orchestrator is unreachable.

    Attributes:
        address: Orchestrator server address (host:port).
        connection_manager: Manages connection state and retry logic.
    """

    def __init__(
        self,
        address: str,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
    ) -> None:
        """Initialize the orchestrator client.

        Parameters:
            address: Server address in host:port or host:port/workspace format.
            heartbeat_interval: Interval between heartbeats in seconds.
            max_retries: Maximum connection retry attempts.
            retry_delay: Initial delay between retries in seconds.
            reconnect_interval: Interval for reconnection attempts in local mode.
        """
        # Parse workspace from address if present (e.g., "host:port/workspace")
        self.address, self._workspace = self._parse_address(address)
        self._heartbeat_interval = heartbeat_interval

        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb2_grpc.GPUOrchestratorStub] = None

        self._process_id: Optional[str] = None
        self._current_device: Optional[str] = None
        self._metadata: Optional[Dict[str, str]] = None

        # Connection state management for graceful degradation
        self.connection_manager = ConnectionManager(
            max_retries=max_retries,
            retry_delay=retry_delay,
            reconnect_interval=reconnect_interval,
        )

        if self._workspace:
            logger.debug(f"OrchestratorClient initialized for {self.address} (workspace: {self._workspace})")
        else:
            logger.debug(f"OrchestratorClient initialized for {self.address}")

    @staticmethod
    def _parse_address(address: str) -> tuple[str, Optional[str]]:
        """Parse address and workspace from connection string.

        Supports formats:
        - host:port           -> (host:port, None)
        - host:port/workspace -> (host:port, workspace)

        Parameters:
            address: Connection string.

        Returns:
            Tuple of (host:port, workspace or None).
        """
        if "/" in address:
            # Find the first / after the port
            parts = address.split("/", 1)
            server_addr = parts[0]
            workspace = parts[1] if len(parts) > 1 else None
            return server_addr, workspace
        return address, None

    def _get_grpc_metadata(self) -> Optional[List[tuple]]:
        """Get gRPC metadata to include with requests.

        Returns:
            List of metadata tuples, or None if no metadata needed.
        """
        if self._workspace:
            return [("workspace", self._workspace)]
        return None

    def connect(self) -> None:
        """Connect to the orchestrator server."""
        # Close existing channel if any (important for reconnection)
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._stub = None

        self._channel = grpc.insecure_channel(self.address)
        self._stub = pb2_grpc.GPUOrchestratorStub(self._channel)
        logger.debug(f"Connected to orchestrator at {self.address}")

    def disconnect(self) -> None:
        """Disconnect from the orchestrator server."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

        logger.debug("Disconnected from orchestrator")

    def register(
        self,
        process_id: str,
        device: str,
        metadata: Optional[Dict[str, str]] = None,
        min_gpus: int = 1,
        max_gpus: int = 1,
        max_vram: int = 0,
        can_share: bool = True,
        priority: int = 50,
        preemptible: bool = True,
        migratable: bool = True,
        start_time: float = 0.0,
    ) -> Optional[str]:
        """Register with the orchestrator.

        Implements retry logic with exponential backoff. If all retries fail,
        switches to local mode and returns None (training continues without
        orchestrator coordination).

        Parameters:
            process_id: Unique identifier for this process.
            device: Current device.
            metadata: Additional metadata.
            min_gpus: Minimum GPUs required.
            max_gpus: Maximum GPUs that can be utilized.
            max_vram: Peak VRAM requirement per GPU in bytes (0 = unlimited).
            can_share: Can run alongside other processes on same GPU.
            priority: Job priority 0-100, higher = more important.
            preemptible: Can be paused/migrated by higher priority jobs.
            migratable: Can be migrated at all.
            start_time: Unix timestamp when process started (for runtime tracking).

        Returns:
            Assigned device (may differ from requested), or None if
            registration failed and we're in local mode.
        """
        if self._stub is None:
            self.connect()

        hostname = socket.gethostname()

        # Get GPU UUID for reliable identification
        gpu_uuid = ""
        gpu_name = ""
        try:
            from flexium.utils.gpu_info import get_gpu_info
            gpu_info = get_gpu_info(device)
            if gpu_info:
                gpu_uuid = gpu_info.uuid
                gpu_name = gpu_info.name
        except Exception:
            pass  # GPU info is optional

        # Store for reconnection attempts
        self._process_id = process_id
        self._current_device = device
        self._metadata = metadata
        self._gpu_uuid = gpu_uuid
        self._gpu_name = gpu_name

        # Store resource requirements for re-registration
        self._min_gpus = min_gpus
        self._max_gpus = max_gpus
        self._max_vram = max_vram
        self._can_share = can_share
        self._priority = priority
        self._preemptible = preemptible
        self._migratable = migratable

        # Store start_time for re-registration
        self._start_time = start_time

        request = pb2.RegisterRequest(
            process_id=process_id,
            device=device,
            hostname=hostname,
            metadata=metadata or {},
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            min_gpus=min_gpus,
            max_gpus=max_gpus,
            max_vram=max_vram,
            can_share=can_share,
            priority=priority,
            preemptible=preemptible,
            migratable=migratable,
            start_time=start_time,
        )

        self.connection_manager.start_connecting()
        logger.debug(f"Attempting to register {process_id} at {self.address}")
        grpc_metadata = self._get_grpc_metadata()

        while True:
            try:
                response = self._stub.Register(request, metadata=grpc_metadata)

                if not response.success:
                    raise RuntimeError(f"Registration rejected: {response.message}")

                self._current_device = response.assigned_device
                self.connection_manager.on_success()

                logger.debug(f"Registered with orchestrator as {process_id}")
                return response.assigned_device

            except grpc.RpcError as e:
                should_retry = self.connection_manager.on_failure(e)
                if should_retry:
                    delay = self.connection_manager.get_retry_delay()
                    logger.debug(f"Registration failed ({e}), retrying in {delay}s...")
                    time.sleep(delay)
                    # Reconnect channel for retry
                    self.connect()
                else:
                    # Local mode - training continues without orchestrator
                    logger.warning(
                        f"Failed to register with orchestrator after {self.connection_manager._max_retries} "
                        f"retries, entering local mode.\n"
                        f"  Hint: Check FLEXIUM_SERVER is set correctly: export FLEXIUM_SERVER='flexium.ai:80/yourworkspace'\n"
                        f"  Hint: Verify the address '{self.address}' is reachable\n"
                        f"  Hint: Check for firewall rules blocking outbound connections\n"
                        f"  Training will continue without migration support."
                    )
                    return None

            except RuntimeError:
                # Registration was rejected by server (not a connection issue)
                raise

    def update_device(self, new_device: str) -> None:
        """Update the current device (call after migration).

        This updates the device sent in heartbeats, which tells the
        orchestrator that migration is complete.

        Parameters:
            new_device: The new device after migration.
        """
        old_device = self._current_device
        self._current_device = new_device
        logger.info(f"Device updated: {old_device} -> {new_device}")

    def complete_migration(
        self,
        process_id: str,
        new_device: str,
        gpu_uuid: str = "",
        gpu_name: str = "",
        memory_reserved: int = 0,
    ) -> bool:
        """Notify orchestrator that migration is complete.

        This clears the MIGRATING status on the server side.

        Parameters:
            process_id: Process ID that completed migration.
            new_device: The new device after migration.
            gpu_uuid: GPU UUID of the new device.
            gpu_name: GPU name of the new device.
            memory_reserved: Memory reserved on new device.

        Returns:
            True if notification succeeded.
        """
        if self._stub is None:
            logger.warning("Cannot notify migration complete: not connected to orchestrator")
            return False

        try:
            request = pb2.CompleteMigrationRequest(
                process_id=process_id,
                new_device=new_device,
                gpu_uuid=gpu_uuid,
                gpu_name=gpu_name,
                memory_reserved=memory_reserved,
            )
            logger.debug(f"Sending CompleteMigration RPC for {process_id} -> {new_device} (gpu_uuid={gpu_uuid})")
            response = self._stub.CompleteMigration(request, metadata=self._get_grpc_metadata())
            if response.success:
                logger.debug(f"Notified orchestrator: migration complete to {new_device}")
            else:
                logger.warning(f"Orchestrator rejected migration completion for {process_id}")
            return response.success
        except Exception as e:
            logger.warning(f"Failed to notify migration complete: {e}")
            return False

    def unregister(self, process_id: Optional[str] = None) -> bool:
        """Unregister from the orchestrator.

        Parameters:
            process_id: Process ID to unregister. Uses current if not provided.

        Returns:
            True if unregistration succeeded.
        """
        if self._stub is None:
            return False

        pid = process_id or self._process_id
        if pid is None:
            return False

        request = pb2.UnregisterRequest(process_id=pid)
        response = self._stub.Unregister(request, metadata=self._get_grpc_metadata())

        if response.success:
            logger.info(f"Unregistered from orchestrator: {pid}")

        return response.success

    def list_processes(self, device_filter: str = "") -> List[Dict[str, Any]]:
        """List all registered processes.

        Parameters:
            device_filter: Optional device to filter by.

        Returns:
            List of process info dictionaries.
        """
        if self._stub is None:
            self.connect()

        request = pb2.ListProcessesRequest(device_filter=device_filter)
        response = self._stub.ListProcesses(request, metadata=self._get_grpc_metadata())

        return [
            {
                "process_id": p.process_id,
                "device": p.device,
                "hostname": p.hostname,
                "status": p.status,
                "memory_allocated": p.memory_allocated,
                "memory_reserved": p.memory_reserved,
                "last_heartbeat": p.last_heartbeat,
                "gpu_uuid": p.gpu_uuid,
                "gpu_name": p.gpu_name,
                "min_gpus": p.min_gpus,
                "max_gpus": p.max_gpus,
                "max_vram": p.max_vram,
                "can_share": p.can_share,
                "priority": p.priority,
                "preemptible": p.preemptible,
                "migratable": p.migratable,
            }
            for p in response.processes
        ]

    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific process.

        Parameters:
            process_id: ID of the process.

        Returns:
            Process info dictionary or None if not found.
        """
        if self._stub is None:
            self.connect()

        request = pb2.GetProcessStatusRequest(process_id=process_id)
        response = self._stub.GetProcessStatus(request, metadata=self._get_grpc_metadata())

        if not response.found:
            return None

        p = response.process
        return {
            "process_id": p.process_id,
            "device": p.device,
            "hostname": p.hostname,
            "status": p.status,
            "memory_allocated": p.memory_allocated,
            "memory_reserved": p.memory_reserved,
            "last_heartbeat": p.last_heartbeat,
            "gpu_uuid": p.gpu_uuid,
            "gpu_name": p.gpu_name,
            "min_gpus": p.min_gpus,
            "max_gpus": p.max_gpus,
            "max_vram": p.max_vram,
            "can_share": p.can_share,
            "priority": p.priority,
            "preemptible": p.preemptible,
            "migratable": p.migratable,
        }

    def request_migration(self, process_id: str, target_device: str) -> bool:
        """Request migration for a process.

        Parameters:
            process_id: ID of the process to migrate.
            target_device: Target device.

        Returns:
            True if migration was requested successfully.
        """
        if self._stub is None:
            self.connect()

        request = pb2.MigrateRequest(
            process_id=process_id,
            target_device=target_device,
        )

        response = self._stub.Migrate(request, metadata=self._get_grpc_metadata())

        if response.success:
            logger.info(f"Migration requested for {process_id} to {target_device}")
        else:
            logger.error(f"Migration request failed: {response.message}")

        return response.success

    def get_device_status(self, device: str = "") -> List[Dict[str, Any]]:
        """Get device utilization status.

        Parameters:
            device: Specific device to query, or empty for all.

        Returns:
            List of device info dictionaries.
        """
        if self._stub is None:
            self.connect()

        request = pb2.GetDeviceStatusRequest(device=device)
        response = self._stub.GetDeviceStatus(request, metadata=self._get_grpc_metadata())

        return [
            {
                "device": d.device,
                "process_count": d.process_count,
                "total_memory": d.total_memory,
                "used_memory": d.used_memory,
                "process_ids": list(d.process_ids),
            }
            for d in response.devices
        ]

    def request_error_recovery(
        self,
        process_id: str,
        error_type: str,
        current_device: str,
        memory_needed: int = 0,
        current_gpu_uuid: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Request error recovery assistance from orchestrator.

        Asks the orchestrator to find a suitable GPU for recovery
        after a GPU error.

        Parameters:
            process_id: ID of the process requesting recovery.
            error_type: Type of error ("OOM", "DEVICE_ASSERT", "ECC", "LAUNCH_FAILURE").
            current_device: Current device to avoid.
            memory_needed: For OOM errors, estimated VRAM needed in bytes.
            current_gpu_uuid: UUID of current GPU to avoid.

        Returns:
            Dict with 'target_device' and 'target_gpu_uuid' if successful,
            None if no suitable GPU available.
        """
        if self._stub is None:
            self.connect()

        # Skip if in local mode
        if self.connection_manager.is_local_mode:
            logger.debug("Cannot request error recovery: in local mode")
            return None

        try:
            request = pb2.ErrorRecoveryRequest(
                process_id=process_id,
                error_type=error_type,
                current_device=current_device,
                memory_needed=memory_needed,
                current_gpu_uuid=current_gpu_uuid,
            )

            response = self._stub.RequestErrorRecovery(request, metadata=self._get_grpc_metadata())

            if response.success:
                logger.info(
                    f"Error recovery target: {response.target_device} "
                    f"({response.target_gpu_uuid[:12] if response.target_gpu_uuid else 'N/A'}...)"
                )
                return {
                    "target_device": response.target_device,
                    "target_gpu_uuid": response.target_gpu_uuid,
                    "message": response.message,
                }
            else:
                logger.warning(f"No recovery target available: {response.message}")
                return None

        except grpc.RpcError as e:
            logger.warning(f"Error recovery request failed: {e}")
            return None

    def mark_gpu_healthy(self, gpu_uuid: str) -> bool:
        """Mark a GPU as healthy (manual recovery).

        Parameters:
            gpu_uuid: UUID of the GPU to mark healthy.

        Returns:
            True if successful.
        """
        if self._stub is None:
            self.connect()

        try:
            request = pb2.MarkGPUHealthyRequest(gpu_uuid=gpu_uuid)
            response = self._stub.MarkGPUHealthy(request, metadata=self._get_grpc_metadata())
            if response.success:
                logger.info(f"Marked GPU {gpu_uuid[:12]}... as healthy")
            return response.success
        except grpc.RpcError as e:
            logger.warning(f"Mark GPU healthy failed: {e}")
            return False

    def get_unhealthy_gpus(self) -> List[Dict[str, Any]]:
        """Get list of unhealthy GPUs.

        Returns:
            List of unhealthy GPU info dicts.
        """
        if self._stub is None:
            self.connect()

        try:
            request = pb2.GetUnhealthyGPUsRequest()
            response = self._stub.GetUnhealthyGPUs(request, metadata=self._get_grpc_metadata())

            return [
                {
                    "gpu_uuid": gpu.gpu_uuid,
                    "reason": gpu.reason,
                    "marked_at": gpu.marked_at,
                    "recovers_at": gpu.recovers_at,
                }
                for gpu in response.gpus
            ]
        except grpc.RpcError as e:
            logger.warning(f"Get unhealthy GPUs failed: {e}")
            return []

    def pause(self, process_id: str) -> Dict[str, Any]:
        """Pause a running process.

        Requests the orchestrator to pause the process, which will trigger
        a checkpoint to free the GPU.

        Parameters:
            process_id: ID of the process to pause.

        Returns:
            Dict with 'success', 'message', and optionally 'checkpoint_path'.
        """
        if self._stub is None:
            self.connect()

        try:
            request = pb2.PauseRequest(process_id=process_id)
            response = self._stub.Pause(request, metadata=self._get_grpc_metadata())

            result = {
                "success": response.success,
                "message": response.message,
            }
            if response.checkpoint_path:
                result["checkpoint_path"] = response.checkpoint_path

            if response.success:
                logger.info(f"Pause requested for process {process_id}")
            else:
                logger.warning(f"Pause request failed: {response.message}")

            return result

        except grpc.RpcError as e:
            logger.warning(f"Pause request failed: {e}")
            return {
                "success": False,
                "message": str(e),
            }

    def resume(
        self,
        process_id: str,
        target_device: str = "",
    ) -> Dict[str, Any]:
        """Resume a paused process.

        Requests the orchestrator to resume the process on the specified
        device (or auto-select if not specified).

        Parameters:
            process_id: ID of the process to resume.
            target_device: Target device to resume on. Empty for auto-select.

        Returns:
            Dict with 'success', 'message', and optionally 'assigned_device'.
        """
        if self._stub is None:
            self.connect()

        try:
            request = pb2.ResumeRequest(
                process_id=process_id,
                target_device=target_device,
            )
            response = self._stub.Resume(request, metadata=self._get_grpc_metadata())

            result = {
                "success": response.success,
                "message": response.message,
            }
            if response.assigned_device:
                result["assigned_device"] = response.assigned_device

            if response.success:
                logger.info(
                    f"Resume requested for process {process_id} "
                    f"on {response.assigned_device}"
                )
            else:
                logger.warning(f"Resume request failed: {response.message}")

            return result

        except grpc.RpcError as e:
            logger.warning(f"Resume request failed: {e}")
            return {
                "success": False,
                "message": str(e),
            }
