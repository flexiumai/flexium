"""gRPC protocol definitions for flexium.

This module contains the generated protobuf code for the
orchestrator service.

The orchestrator_pb2 module contains message definitions and can be
imported without grpc. The orchestrator_pb2_grpc module requires grpc.
"""

# pb2 module doesn't require grpc
from flexium.proto import orchestrator_pb2

__all__ = ["orchestrator_pb2"]

# grpc stubs are optional
try:
    from flexium.proto import orchestrator_pb2_grpc

    __all__.append("orchestrator_pb2_grpc")
except ImportError:
    # grpc not installed
    pass
