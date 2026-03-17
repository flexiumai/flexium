"""PyTorch Lightning integration for Flexium.

This module provides seamless integration between Flexium's transparent GPU
migration and PyTorch Lightning training workflows.

Usage:
    from flexium.lightning import FlexiumCallback

    trainer = Trainer(
        callbacks=[FlexiumCallback(orchestrator="localhost:80")],
        max_epochs=100,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, dataloader)

The FlexiumCallback handles:
- Connecting to the Flexium orchestrator during setup
- Transparent GPU migration via heartbeat thread
- Proper cleanup during teardown
"""

from __future__ import annotations

try:
    from flexium.lightning.callback import FlexiumCallback
except ImportError:
    # Provide helpful error if Lightning is not installed
    class FlexiumCallback:  # type: ignore[no-redef]
        """Placeholder that raises ImportError when instantiated."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Lightning is required for FlexiumCallback. "
                "Install it with: pip install pytorch-lightning>=2.0.0 "
                "or: pip install flexium[lightning]"
            )

__all__ = ["FlexiumCallback"]
