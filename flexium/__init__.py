"""Flexium - Dynamic GPU orchestration for PyTorch training.

This package provides GPU orchestration with live migration support using
driver-level migration for zero-residue GPU migration (requires driver 580+).

Simple Usage (recommended):

    import flexium
    flexium.init()

    model = nn.Linear(784, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            data, target = batch[0].cuda(), batch[1].cuda()
            loss = model(data).sum()
            loss.backward()
            optimizer.step()

Explicit Scope (advanced):

    import flexium.auto

    with flexium.auto.run():
        # Training code with explicit scope control
        ...

Configuration:
    - Set FLEXIUM_SERVER environment variable to server address (host:port/workspace)
    - Or pass server= parameter to flexium.init() / flexium.auto.run()

Migration is transparent - training continues in the same process,
same loop iteration, just on a different GPU.
"""

from __future__ import annotations

import atexit
from typing import Optional

__version__ = "0.2.0a3"

# Track if init() has been called
_initialized = False
_init_context = None


def init(
    server: Optional[str] = None,
    device: Optional[str] = None,
    disabled: bool = False,
) -> None:
    """Initialize Flexium for the entire process.

    This is the simplest way to enable GPU migration. Call once at the start
    of your script, and Flexium will manage everything until process exit.

    Parameters:
        server: Flexium server address (host:port/workspace).
                If not provided, uses FLEXIUM_SERVER environment variable.
        device: Initial GPU device (e.g., "cuda:0").
                If not provided, uses GPU_DEVICE env var or "cuda:0".
        disabled: If True, bypass Flexium entirely. Useful for benchmarking
                  or when you want to run without Flexium temporarily.

    Example:
        import flexium
        flexium.init()

        # Your training code - unchanged
        model = Net().cuda()
        for epoch in range(100):
            train(model)

        # To disable Flexium (e.g., for benchmarking):
        flexium.init(disabled=True)
    """
    global _initialized, _init_context

    if _initialized:
        print("[flexium] Already initialized, ignoring duplicate init() call")
        return

    # Import here to avoid circular imports
    from flexium import auto

    # Start the context (this does all the setup)
    _init_context = auto.run(orchestrator=server, device=device, disabled=disabled)
    _init_context.__enter__()

    # Register cleanup on process exit
    atexit.register(_shutdown)

    _initialized = True


def _shutdown() -> None:
    """Clean up Flexium on process exit."""
    global _initialized, _init_context

    if not _initialized or _init_context is None:
        return

    try:
        _init_context.__exit__(None, None, None)
    except Exception:
        pass  # Best effort cleanup

    _initialized = False
    _init_context = None


def shutdown() -> None:
    """Manually shut down Flexium.

    Usually not needed - Flexium cleans up automatically on process exit.
    Use this if you need to explicitly stop Flexium mid-process.
    """
    # Unregister atexit handler to avoid double cleanup
    try:
        atexit.unregister(_shutdown)
    except Exception:
        pass

    _shutdown()


# For checking state
def is_initialized() -> bool:
    """Check if Flexium has been initialized."""
    return _initialized

# GPU layer for testing/mocking
from flexium.gpu import GPUInterface, GPUInfo, DeviceReport, NvidiaGPU, MockGPU
from flexium.utils.gpu_info import set_gpu_backend, get_gpu_backend

__all__ = [
    "__version__",
    # Main API
    "init",
    "shutdown",
    "is_initialized",
    # GPU layer
    "GPUInterface",
    "GPUInfo",
    "DeviceReport",
    "NvidiaGPU",
    "MockGPU",
    "set_gpu_backend",
    "get_gpu_backend",
]
