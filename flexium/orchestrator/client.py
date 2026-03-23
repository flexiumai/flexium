"""Orchestrator client for flexium.

Provides a client for connecting to the orchestrator server from
training processes. Uses a pluggable transport layer for communication,
allowing easy switching between WebSocket, gRPC, or mock transports.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, Dict, Optional, Union

from flexium.orchestrator.transport import Transport, MockTransport
from flexium.orchestrator.websocket_transport import WebSocketTransport
from flexium.timing import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RECONNECT_INTERVAL,
    DEFAULT_RETRY_DELAY,
    HEARTBEAT_INTERVAL,
)
from flexium.utils.logging import get_logger

logger = get_logger(__name__)

# Default heartbeat interval in seconds
DEFAULT_HEARTBEAT_INTERVAL = HEARTBEAT_INTERVAL


class ConnectionState:
    """Connection state constants."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    LOCAL_MODE = "local_mode"


class OrchestratorClient:
    """Client for communicating with the orchestrator server.

    Handles registration, heartbeats, and migration coordination.
    Supports graceful degradation when orchestrator is unreachable.

    The client is transport-agnostic - it works with any transport
    implementing the Transport interface (WebSocket, gRPC, Mock, etc.).

    Example with address string (creates WebSocket transport):
        client = OrchestratorClient(address="app.flexium.ai/workspace")

    Example with explicit transport (for testing):
        transport = MockTransport(my_handler)
        client = OrchestratorClient(transport=transport)
    """

    def __init__(
        self,
        address: Optional[str] = None,
        transport: Optional[Transport] = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
    ) -> None:
        """Initialize the orchestrator client.

        Parameters:
            address: Server address in host:port/workspace format.
                     Creates a WebSocket transport. Ignored if transport is provided.
            transport: Pre-configured transport instance. If provided, address is ignored.
            heartbeat_interval: Interval between heartbeats in seconds.
            max_retries: Maximum connection retry attempts.
            retry_delay: Initial delay between retries in seconds.
            reconnect_interval: Interval for reconnection attempts in local mode.
        """
        # Create transport from address if not provided
        if transport is not None:
            self._transport = transport
            self._workspace = getattr(transport, 'workspace', None)
        elif address is not None:
            server_url, workspace = self._parse_address(address)
            self._transport = WebSocketTransport(server_url, workspace)
            self._workspace = workspace
        else:
            raise ValueError("Either 'address' or 'transport' must be provided")

        self._heartbeat_interval = heartbeat_interval
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._reconnect_interval = reconnect_interval

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._consecutive_failures = 0
        self._last_reconnect_attempt: Optional[float] = None

        # Process info (stored for re-registration)
        self._process_id: Optional[str] = None
        self._current_device: Optional[str] = None
        self._metadata: Optional[Dict[str, str]] = None
        self._gpu_uuid: str = ""
        self._gpu_name: str = ""

        # Resource requirements (stored for re-registration)
        self._min_gpus: int = 1
        self._max_gpus: int = 1
        self._max_vram: int = 0
        self._can_share: bool = True
        self._priority: int = 50
        self._preemptible: bool = True
        self._migratable: bool = True
        self._start_time: float = 0.0

        # Callbacks for migration/pause commands
        self._migration_callback: Optional[Callable[[str], None]] = None
        self._pause_callback: Optional[Callable[[], None]] = None

        logger.debug(f"OrchestratorClient initialized for {self._transport.server_url}")

    @staticmethod
    def _parse_address(address: str) -> tuple:
        """Parse address and workspace from connection string.

        Supports formats:
        - host/workspace         -> (https://host, workspace)
        - host:port/workspace    -> (https://host:port, workspace) if port=443
        - host:port/workspace    -> (http://host:port, workspace) otherwise

        Parameters:
            address: Connection string.

        Returns:
            Tuple of (server_url, workspace).
        """
        if "/" not in address:
            raise ValueError(f"Invalid address format: {address}. Expected host:port/workspace or host/workspace")

        parts = address.split("/", 1)
        host_port = parts[0]
        workspace = parts[1] if len(parts) > 1 else None

        if not workspace:
            raise ValueError(f"Workspace is required in address: {address}")

        # Determine protocol based on port
        if ":" in host_port:
            port = host_port.split(":")[-1]
            if port == "443":
                protocol = "https"
            else:
                protocol = "http"
        else:
            protocol = "https"

        server_url = f"{protocol}://{host_port}"
        return server_url, workspace

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return self._transport.server_url

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._transport.is_connected()

    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self._state == ConnectionState.LOCAL_MODE

    def set_migration_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for migration commands.

        Parameters:
            callback: Function to call with target device when migration is requested.
        """
        self._migration_callback = callback
        self._transport.set_event_handler("migrate", lambda data: callback(data.get("target_device", "")))

    def set_pause_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for pause commands.

        Parameters:
            callback: Function to call when pause is requested.
        """
        self._pause_callback = callback
        self._transport.set_event_handler("pause", lambda data: callback())

    def connect(self) -> bool:
        """Connect to the orchestrator server.

        Returns:
            True if connected successfully.
        """
        self._state = ConnectionState.CONNECTING
        result = self._transport.connect()
        if result:
            self._state = ConnectionState.CONNECTED
        return result

    def disconnect(self) -> None:
        """Disconnect from the orchestrator server."""
        self._transport.disconnect()
        self._state = ConnectionState.DISCONNECTED
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
        _skip_gpu_lookup: bool = False,
    ) -> Optional[str]:
        """Register with the orchestrator.

        Parameters:
            process_id: Unique identifier for this process.
            device: Current device.
            metadata: Additional metadata.
            min_gpus: Minimum GPUs required.
            max_gpus: Maximum GPUs that can be utilized.
            max_vram: Peak VRAM requirement per GPU in bytes.
            can_share: Can run alongside other processes on same GPU.
            priority: Job priority 0-100.
            preemptible: Can be paused/migrated by higher priority jobs.
            migratable: Can be migrated at all.
            start_time: Unix timestamp when process started.

        Returns:
            Assigned device, or None if registration failed.
        """
        hostname = socket.gethostname()
        logger.info(f"[flexium] register() called for {process_id} on {device}")

        # Get GPU UUID for reliable identification
        # Skip if reconnecting (GPU may be paused/checkpointed and pynvml will hang)
        gpu_uuid = ""
        gpu_name = ""
        if _skip_gpu_lookup:
            # Use cached values from previous registration
            gpu_uuid = self._gpu_uuid
            gpu_name = self._gpu_name
            logger.info(f"[flexium] Using cached GPU info: {gpu_uuid}, {gpu_name}")
        else:
            try:
                from flexium.utils.gpu_info import get_gpu_info
                logger.info(f"[flexium] Getting GPU info for {device}...")
                gpu_info = get_gpu_info(device)
                logger.info(f"[flexium] Got GPU info: {gpu_info}")
                if gpu_info:
                    gpu_uuid = gpu_info.uuid
                    gpu_name = gpu_info.name
            except Exception as e:
                logger.warning(f"[flexium] Failed to get GPU info: {e}")

        # Store for reconnection
        self._process_id = process_id
        self._current_device = device
        self._metadata = metadata
        self._gpu_uuid = gpu_uuid
        self._gpu_name = gpu_name
        self._min_gpus = min_gpus
        self._max_gpus = max_gpus
        self._max_vram = max_vram
        self._can_share = can_share
        self._priority = priority
        self._preemptible = preemptible
        self._migratable = migratable
        self._start_time = start_time

        self._consecutive_failures = 0
        retry_count = 0

        while retry_count < self._max_retries:
            # Connect if not connected
            if not self.is_connected:
                if not self.connect():
                    retry_count += 1
                    delay = self._retry_delay * (DEFAULT_BACKOFF_MULTIPLIER ** (retry_count - 1))
                    logger.warning(f"[flexium] Connection failed: could not connect to {self.server_url}")
                    logger.warning(f"[flexium] Retrying in {delay:.1f}s (attempt {retry_count}/{self._max_retries})")
                    time.sleep(delay)
                    continue

            # Send register message
            logger.info(f"[flexium] Sending register for {process_id} on {device}")
            response = self._transport.send("register", {
                "process_id": process_id,
                "device": device,
                "hostname": hostname,
                "metadata": metadata or {},
                "gpu_uuid": gpu_uuid,
                "gpu_name": gpu_name,
                "min_gpus": min_gpus,
                "max_gpus": max_gpus,
                "max_vram": max_vram,
                "can_share": can_share,
                "priority": priority,
                "preemptible": preemptible,
                "migratable": migratable,
                "start_time": start_time,
            })
            logger.info(f"[flexium] Register response: {response}")

            if response and response.get("success"):
                self._current_device = response.get("assigned_device", device)
                self._state = ConnectionState.CONNECTED
                logger.debug(f"Registered with orchestrator as {process_id}")
                return self._current_device
            else:
                error_msg = response.get("message", "Unknown error") if response else "No response"
                retry_count += 1
                if retry_count < self._max_retries:
                    delay = self._retry_delay * (DEFAULT_BACKOFF_MULTIPLIER ** (retry_count - 1))
                    logger.warning(f"[flexium] Connection failed: Registration rejected: {error_msg}")
                    logger.warning(f"[flexium] Retrying in {delay:.1f}s (attempt {retry_count}/{self._max_retries})")
                    time.sleep(delay)
                    self.disconnect()

        # Max retries exhausted - enter local mode
        self._state = ConnectionState.LOCAL_MODE
        logger.warning("[flexium] Switching to local mode (migration disabled)")
        logger.info("[flexium] Training continues - will attempt reconnection periodically")
        logger.warning(
            f"Failed to register with orchestrator after {self._max_retries} retries, entering local mode.\n"
            f"  Hint: Check FLEXIUM_SERVER is set correctly: export FLEXIUM_SERVER='app.flexium.ai/yourworkspace'\n"
            f"  Hint: Verify the server is reachable\n"
            f"  Training will continue without migration support."
        )
        return None

    def heartbeat(
        self,
        memory_allocated: int = 0,
        memory_reserved: int = 0,
        gpu_pid: int = 0,
        container_pid: int = 0,
        pynvml_gpu_uuid: str = "",
        cuda_device_count: int = 0,
        device: Optional[str] = None,
        gpu_uuid: Optional[str] = None,
        gpu_name: Optional[str] = None,
        visible_devices: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send heartbeat to orchestrator.

        Parameters:
            memory_allocated: Memory allocated by process.
            memory_reserved: Memory reserved by process.
            gpu_pid: Process ID as seen by nvidia-smi.
            container_pid: Container/process PID.
            pynvml_gpu_uuid: GPU UUID from pynvml.
            cuda_device_count: Number of visible CUDA devices.
            device: Current device (overrides stored value).
            gpu_uuid: GPU UUID (overrides stored value).
            gpu_name: GPU name (overrides stored value).
            visible_devices: List of all device reports from this host.

        Returns:
            Response dict with migration/pause commands, or None if failed.
        """
        if not self.is_connected:
            # Connection lost - enter local mode and try reconnection
            if self._state == ConnectionState.CONNECTED:
                logger.warning("[flexium] Connection to orchestrator lost")
                logger.info("[flexium] Switching to local mode - will attempt reconnection periodically")
                self._state = ConnectionState.LOCAL_MODE

            # Try reconnection in local mode
            if self._state == ConnectionState.LOCAL_MODE:
                if self._should_attempt_reconnect():
                    logger.info("[flexium] Attempting reconnection to orchestrator...")
                    if self._try_reconnect():
                        return {"success": True}
            return None

        data = {
            "process_id": self._process_id,
            "device": device if device is not None else self._current_device,
            "memory_allocated": memory_allocated,
            "memory_reserved": memory_reserved,
            "gpu_uuid": gpu_uuid if gpu_uuid is not None else self._gpu_uuid,
            "gpu_name": gpu_name if gpu_name is not None else self._gpu_name,
            "gpu_pid": gpu_pid,
            "container_pid": container_pid,
            "pynvml_gpu_uuid": pynvml_gpu_uuid,
            "cuda_device_count": cuda_device_count,
        }
        if visible_devices:
            data["visible_devices"] = visible_devices

        logger.debug(f"Sending heartbeat: connected={self._transport.is_connected()}, process_id={self._process_id}")
        response = self._transport.send("heartbeat", data, timeout=5.0)
        logger.debug(f"Heartbeat response: {response}")

        return response

    def _should_attempt_reconnect(self) -> bool:
        """Check if we should try reconnecting."""
        now = time.time()
        if self._last_reconnect_attempt is None:
            self._last_reconnect_attempt = now
            return True

        if now - self._last_reconnect_attempt >= self._reconnect_interval:
            self._last_reconnect_attempt = now
            return True

        return False

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect and re-register."""
        try:
            if self.connect():
                logger.info("[flexium] Connected, re-registering...")
                logger.info(f"[flexium] _try_reconnect: process_id={self._process_id}, device={self._current_device}")
                result = self.register(
                    process_id=self._process_id,
                    device=self._current_device,
                    metadata=self._metadata,
                    min_gpus=self._min_gpus,
                    max_gpus=self._max_gpus,
                    max_vram=self._max_vram,
                    can_share=self._can_share,
                    priority=self._priority,
                    preemptible=self._preemptible,
                    migratable=self._migratable,
                    start_time=self._start_time,
                    _skip_gpu_lookup=True,  # GPU may be paused, skip pynvml call
                )
                if result:
                    logger.info("[flexium] Orchestrator reconnected!")
                    logger.info("[flexium] Migration re-enabled")
                    return True
                else:
                    logger.warning("[flexium] Re-registration failed")
            else:
                logger.warning("[flexium] Reconnect failed: could not connect")
        except Exception as e:
            logger.info(f"[flexium] Reconnection failed: {e}")

        # Reconnect failed - stay in local mode
        self._state = ConnectionState.LOCAL_MODE

        return False

    def update_device(self, new_device: str) -> None:
        """Update the current device after migration.

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

        Parameters:
            process_id: Process ID that completed migration.
            new_device: The new device after migration.
            gpu_uuid: GPU UUID of the new device.
            gpu_name: GPU name of the new device.
            memory_reserved: Memory reserved on new device.

        Returns:
            True if notification succeeded.
        """
        if not self.is_connected:
            logger.warning("Cannot notify migration complete: not connected")
            return False

        response = self._transport.send("complete_migration", {
            "process_id": process_id,
            "new_device": new_device,
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "memory_reserved": memory_reserved,
            "success": True,
        }, timeout=5.0)

        if response and response.get("success"):
            logger.debug(f"Notified orchestrator: migration complete to {new_device}")
            return True
        return False

    def complete_pause(
        self,
        process_id: str,
        gpu_uuid: str = "",
        gpu_name: str = "",
        memory_reserved: int = 0,
    ) -> bool:
        """Notify orchestrator that pause is complete.

        Parameters:
            process_id: Process ID that completed pause.
            gpu_uuid: GPU UUID of the paused device.
            gpu_name: GPU name of the paused device.
            memory_reserved: Memory that was reserved before pause.

        Returns:
            True if notification succeeded.
        """
        if not self.is_connected:
            logger.warning("Cannot notify pause complete: not connected")
            return False

        response = self._transport.send("complete_pause", {
            "process_id": process_id,
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "memory_reserved": memory_reserved,
        }, timeout=5.0)

        if response and response.get("success"):
            logger.debug("Notified orchestrator: pause complete")
            return True
        return False

    def unregister(self, process_id: Optional[str] = None) -> bool:
        """Unregister from the orchestrator.

        Parameters:
            process_id: Process ID to unregister. Uses stored ID if not provided.

        Returns:
            True if unregistration succeeded.
        """
        pid = process_id or self._process_id
        if not pid:
            return False

        if not self.is_connected:
            return False

        response = self._transport.send("unregister", {
            "process_id": pid,
        }, timeout=5.0)

        if response and response.get("success"):
            logger.debug(f"Unregistered from orchestrator: {pid}")
            return True
        return False


# Alias for backward compatibility
WebSocketClient = OrchestratorClient

__all__ = [
    "ConnectionState",
    "OrchestratorClient",
    "WebSocketClient",
    "DEFAULT_HEARTBEAT_INTERVAL",
]
