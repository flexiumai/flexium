"""Orchestrator client module for flexium.

Provides the client for connecting to the orchestrator server from
training processes.
"""

__all__ = []

# gRPC-dependent modules are optional
try:
    from flexium.orchestrator.client import (
        OrchestratorClient,
        ConnectionState,
        ConnectionManager,
    )

    __all__.extend([
        "OrchestratorClient",
        "ConnectionState",
        "ConnectionManager",
    ])
except ImportError:
    # grpc not installed
    pass
