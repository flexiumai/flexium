"""Orchestrator module for flexium.

Provides the orchestrator server that manages training processes
across multiple GPUs and machines.
"""

from flexium.orchestrator.registry import ProcessRegistry
from flexium.orchestrator.device_manager import DeviceManager

__all__ = [
    "ProcessRegistry",
    "DeviceManager",
]

# gRPC-dependent modules are optional
try:
    from flexium.orchestrator.server import OrchestratorServer
    from flexium.orchestrator.client import (
        OrchestratorClient,
        ConnectionState,
        ConnectionManager,
    )

    __all__.extend([
        "OrchestratorServer",
        "OrchestratorClient",
        "ConnectionState",
        "ConnectionManager",
    ])
except ImportError:
    # grpc not installed
    pass
