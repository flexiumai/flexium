"""Pytest configuration and fixtures for flexium tests."""

import pytest
import sys

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if CUDA is available (only if torch is available)
CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

# Markers for conditional skipping
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available"
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: mark test as requiring CUDA"
    )
