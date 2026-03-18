"""Mock GPU implementation for testing.

Provides a fake GPU implementation that doesn't require real hardware.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flexium.gpu.interface import GPUInterface, GPUInfo, DeviceReport


class MockGPU(GPUInterface):
    """Mock GPU implementation for testing.

    Simulates GPU devices without requiring real hardware.

    Example:
        gpu = MockGPU(num_devices=2, memory_per_device=16 * 1024**3)
        reports = gpu.get_all_device_reports("localhost")
    """

    def __init__(
        self,
        num_devices: int = 1,
        memory_per_device: int = 16 * 1024**3,  # 16GB default
        gpu_name: str = "Mock GPU",
    ):
        """Initialize mock GPU.

        Parameters:
            num_devices: Number of mock GPUs to simulate.
            memory_per_device: Memory per device in bytes.
            gpu_name: Name for mock GPUs.
        """
        self._num_devices = num_devices
        self._memory_per_device = memory_per_device
        self._gpu_name = gpu_name
        self._current_device = 0

        # Track allocated memory per device
        self._allocated: Dict[int, int] = {i: 0 for i in range(num_devices)}
        self._reserved: Dict[int, int] = {i: 0 for i in range(num_devices)}

    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        return self._num_devices

    def get_device_info(self, device_index: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        if device_index < 0 or device_index >= self._num_devices:
            return None

        allocated = self._allocated.get(device_index, 0)
        reserved = self._reserved.get(device_index, 0)
        used = max(allocated, reserved)

        return GPUInfo(
            device_index=device_index,
            uuid=f"GPU-MOCK-{device_index:04d}-0000-0000-000000000000",
            name=f"{self._gpu_name} {device_index}",
            memory_total=self._memory_per_device,
            memory_used=used,
            memory_free=self._memory_per_device - used,
            utilization=min(100, int(used / self._memory_per_device * 100)),
            temperature=50 + device_index * 5,
            power_usage=100 + device_index * 20,
        )

    def get_all_device_reports(self, hostname: str) -> List[DeviceReport]:
        """Get reports for all devices on this host."""
        reports = []

        for i in range(self._num_devices):
            info = self.get_device_info(i)
            if info:
                reports.append(DeviceReport(
                    gpu_uuid=info.uuid,
                    gpu_name=info.name,
                    hostname=hostname,
                    memory_total=info.memory_total,
                    memory_used=info.memory_used,
                    memory_free=info.memory_free,
                    gpu_utilization=info.utilization,
                    temperature=info.temperature,
                    power_usage=info.power_usage,
                    process_count=1 if self._allocated.get(i, 0) > 0 else 0,
                ))

        return reports

    def get_memory_info(self, device_index: int) -> tuple:
        """Get memory info for a device."""
        if device_index < 0 or device_index >= self._num_devices:
            return (0, 0)
        return (
            self._allocated.get(device_index, 0),
            self._reserved.get(device_index, 0),
        )

    def get_gpu_uuid(self, device_index: int) -> str:
        """Get the UUID of a GPU."""
        info = self.get_device_info(device_index)
        return info.uuid if info else ""

    def get_gpu_name(self, device_index: int) -> str:
        """Get the name of a GPU."""
        info = self.get_device_info(device_index)
        return info.name if info else ""

    def get_current_device(self) -> int:
        """Get the current CUDA device index."""
        return self._current_device

    def set_current_device(self, device_index: int) -> None:
        """Set the current CUDA device."""
        if 0 <= device_index < self._num_devices:
            self._current_device = device_index

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._num_devices > 0

    # Mock-specific methods for testing

    def set_memory_allocated(self, device_index: int, bytes: int) -> None:
        """Set allocated memory for a device (for testing)."""
        if 0 <= device_index < self._num_devices:
            self._allocated[device_index] = bytes

    def set_memory_reserved(self, device_index: int, bytes: int) -> None:
        """Set reserved memory for a device (for testing)."""
        if 0 <= device_index < self._num_devices:
            self._reserved[device_index] = bytes
