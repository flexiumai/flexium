"""Utility functions for Flexium Lightning integration."""

from __future__ import annotations

from typing import Optional

import torch
from pytorch_lightning import Trainer


def get_trainer_device(trainer: Trainer) -> Optional[str]:
    """Get the current device from a Lightning Trainer.

    Parameters:
        trainer: The Lightning Trainer instance.

    Returns:
        Device string (e.g., "cuda:0") or None if not available.
    """
    try:
        root_device = trainer.strategy.root_device
        if root_device is not None:
            return str(root_device)
    except AttributeError:
        pass
    return None


def sync_device_to_trainer(trainer: Trainer, device: str) -> None:
    """Synchronize Flexium's device to the Lightning Trainer.

    After a migration, the trainer's internal device reference may be stale.
    This function updates the trainer to use the new device.

    Parameters:
        trainer: The Lightning Trainer instance.
        device: The new device string (e.g., "cuda:1").
    """
    try:
        new_device = torch.device(device)

        # Update the strategy's root device
        # This is the primary device reference used by Lightning
        if hasattr(trainer.strategy, "root_device"):
            # For SingleDeviceStrategy and similar
            trainer.strategy._root_device = new_device

        print(f"[flexium] Lightning device synced to: {device}")

    except Exception as e:
        # Don't fail training if sync fails - just log
        print(f"[flexium] Warning: Could not sync device to trainer: {e}")


def is_lightning_available() -> bool:
    """Check if PyTorch Lightning is available.

    Returns:
        True if pytorch_lightning can be imported.
    """
    try:
        import pytorch_lightning  # noqa: F401

        return True
    except ImportError:
        return False


def get_lightning_version() -> Optional[str]:
    """Get the installed PyTorch Lightning version.

    Returns:
        Version string or None if not installed.
    """
    try:
        import pytorch_lightning

        return pytorch_lightning.__version__
    except ImportError:
        return None
