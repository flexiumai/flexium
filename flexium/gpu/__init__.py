"""GPU abstraction layer for Flexium.

This module provides a clean interface between the client and GPU operations,
making it easier to test and swap implementations.
"""

from flexium.gpu.interface import GPUInterface, GPUInfo, DeviceReport
from flexium.gpu.nvidia import NvidiaGPU
from flexium.gpu.mock import MockGPU

__all__ = [
    "GPUInterface",
    "GPUInfo",
    "DeviceReport",
    "NvidiaGPU",
    "MockGPU",
]
