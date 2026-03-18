"""NVIDIA GPU implementation.

Real GPU implementation using PyTorch and pynvml.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flexium.gpu.interface import GPUInterface, GPUInfo, DeviceReport
from flexium.utils.logging import get_logger

logger = get_logger(__name__)


class NvidiaGPU(GPUInterface):
    """NVIDIA GPU implementation using PyTorch and pynvml."""

    def __init__(self):
        """Initialize NVIDIA GPU interface."""
        self._torch = None
        self._pynvml_initialized = False

    def _ensure_torch(self):
        """Lazy load torch."""
        if self._torch is None:
            import torch
            self._torch = torch

    def _ensure_pynvml(self) -> bool:
        """Initialize pynvml if not already done."""
        if self._pynvml_initialized:
            return True
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_initialized = True
            return True
        except Exception as e:
            logger.debug(f"pynvml not available: {e}")
            return False

    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        self._ensure_torch()
        if self._torch.cuda.is_available():
            return self._torch.cuda.device_count()
        return 0

    def get_device_info(self, device_index: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        if not self._ensure_pynvml():
            return None

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
            except Exception:
                gpu_util = 0

            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
            except Exception:
                power = 0

            return GPUInfo(
                device_index=device_index,
                uuid=uuid,
                name=name,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_free=memory.free,
                utilization=gpu_util,
                temperature=temperature,
                power_usage=power,
            )
        except Exception as e:
            logger.debug(f"Failed to get device info for {device_index}: {e}")
            return None

    def get_all_device_reports(self, hostname: str) -> List[DeviceReport]:
        """Get reports for all devices on this host."""
        reports = []

        if not self._ensure_pynvml():
            return reports

        try:
            import pynvml

            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                info = self.get_device_info(i)
                if info:
                    # Get process count
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        process_count = len(procs)
                    except Exception:
                        process_count = 0

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
                        process_count=process_count,
                    ))
        except Exception as e:
            logger.debug(f"Failed to get device reports: {e}")

        return reports

    def get_memory_info(self, device_index: int) -> tuple:
        """Get memory info for a device."""
        self._ensure_torch()
        if not self._torch.cuda.is_available():
            return (0, 0)

        try:
            allocated = self._torch.cuda.memory_allocated(device_index)
            reserved = self._torch.cuda.memory_reserved(device_index)
            return (allocated, reserved)
        except Exception:
            return (0, 0)

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
        self._ensure_torch()
        if self._torch.cuda.is_available():
            return self._torch.cuda.current_device()
        return 0

    def set_current_device(self, device_index: int) -> None:
        """Set the current CUDA device."""
        self._ensure_torch()
        if self._torch.cuda.is_available():
            self._torch.cuda.set_device(device_index)

    def is_available(self) -> bool:
        """Check if GPU is available."""
        self._ensure_torch()
        return self._torch.cuda.is_available()
