"""GPU interface abstraction.

Defines the contract for GPU operations, allowing different implementations
(real NVIDIA, mock for testing, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    device_index: int
    uuid: str
    name: str
    memory_total: int = 0
    memory_used: int = 0
    memory_free: int = 0
    utilization: int = 0
    temperature: int = 0
    power_usage: int = 0


@dataclass
class DeviceReport:
    """Report about a device for sending to orchestrator."""

    gpu_uuid: str
    gpu_name: str
    hostname: str
    memory_total: int = 0
    memory_used: int = 0
    memory_free: int = 0
    gpu_utilization: int = 0
    temperature: int = 0
    power_usage: int = 0
    process_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gpu_uuid": self.gpu_uuid,
            "gpu_name": self.gpu_name,
            "hostname": self.hostname,
            "memory_total": self.memory_total,
            "memory_used": self.memory_used,
            "memory_free": self.memory_free,
            "gpu_utilization": self.gpu_utilization,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "process_count": self.process_count,
        }


class GPUInterface(ABC):
    """Abstract interface for GPU operations.

    This abstraction allows:
    - Testing without real GPUs (MockGPU)
    - Different GPU vendors in the future
    - Clean separation of concerns
    """

    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        pass

    @abstractmethod
    def get_device_info(self, device_index: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU.

        Parameters:
            device_index: The CUDA device index.

        Returns:
            GPUInfo or None if device not found.
        """
        pass

    @abstractmethod
    def get_all_device_reports(self, hostname: str) -> List[DeviceReport]:
        """Get reports for all devices on this host.

        Parameters:
            hostname: The hostname to include in reports.

        Returns:
            List of DeviceReport for all GPUs.
        """
        pass

    @abstractmethod
    def get_memory_info(self, device_index: int) -> tuple:
        """Get memory info for a device.

        Parameters:
            device_index: The CUDA device index.

        Returns:
            Tuple of (allocated, reserved) in bytes.
        """
        pass

    @abstractmethod
    def get_gpu_uuid(self, device_index: int) -> str:
        """Get the UUID of a GPU.

        Parameters:
            device_index: The CUDA device index.

        Returns:
            GPU UUID string.
        """
        pass

    @abstractmethod
    def get_gpu_name(self, device_index: int) -> str:
        """Get the name of a GPU.

        Parameters:
            device_index: The CUDA device index.

        Returns:
            GPU name string.
        """
        pass

    @abstractmethod
    def get_current_device(self) -> int:
        """Get the current CUDA device index."""
        pass

    @abstractmethod
    def set_current_device(self, device_index: int) -> None:
        """Set the current CUDA device."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU is available."""
        pass
