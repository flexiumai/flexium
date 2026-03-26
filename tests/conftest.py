"""Pytest configuration and fixtures for flexium tests."""

import os
import subprocess

import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if CUDA is available (only if torch is available)
CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

# Check if NVIDIA driver is available
def _check_nvidia_driver():
    """Check if NVIDIA driver is installed and accessible."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False

NVIDIA_DRIVER_AVAILABLE = _check_nvidia_driver()

# Check if PyYAML is available
try:
    import yaml
    PYYAML_AVAILABLE = True
except ImportError:
    PYYAML_AVAILABLE = False

# Markers for conditional skipping
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available"
)

requires_nvidia_driver = pytest.mark.skipif(
    not NVIDIA_DRIVER_AVAILABLE,
    reason="NVIDIA driver not available"
)

requires_pyyaml = pytest.mark.skipif(
    not PYYAML_AVAILABLE,
    reason="PyYAML not installed"
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "requires_nvidia_driver: mark test as requiring NVIDIA driver"
    )
    config.addinivalue_line(
        "markers", "requires_pyyaml: mark test as requiring PyYAML"
    )
