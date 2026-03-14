"""Flexium callback for PyTorch Lightning.

This callback integrates Flexium's transparent GPU migration with
PyTorch Lightning's training workflow.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

import flexium.auto
from flexium.lightning.utils import sync_device_to_trainer


class FlexiumCallback(Callback):
    """PyTorch Lightning callback for Flexium GPU migration.

    This callback enables transparent GPU migration for Lightning training.
    It wraps the training in a flexium.auto.run() context and handles
    device synchronization after migrations.

    Parameters:
        orchestrator: Orchestrator address (host:port). If None, uses
            environment variable GPU_ORCHESTRATOR or config file.
        device: Initial device to use. If None, auto-detected.
        disabled: If True, bypass Flexium entirely (useful for debugging).

    Example:
        from flexium.lightning import FlexiumCallback

        trainer = Trainer(
            callbacks=[FlexiumCallback(orchestrator="localhost:80")],
            max_epochs=100,
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(model, train_loader)

    Note:
        - Supports one GPU per training process (covers most use cases)
        - DDP/multi-GPU support planned for future releases
        - Works with: single-GPU training, hyperparameter sweeps,
          multiple independent jobs each on their own GPU
    """

    def __init__(
        self,
        orchestrator: Optional[str] = None,
        device: Optional[str] = None,
        disabled: bool = False,
    ) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.device = device
        self.disabled = disabled
        self._flexium_context: Optional[Any] = None
        self._last_device: Optional[str] = None

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Set up Flexium context when training starts.

        Parameters:
            trainer: The Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        if self.disabled:
            return

        # Enter flexium.auto.run() context
        # This starts heartbeat thread and registers with orchestrator
        self._flexium_context = flexium.auto.run(
            orchestrator=self.orchestrator,
            device=self.device,
            disabled=self.disabled,
        )
        self._flexium_context.__enter__()

        # Track initial device
        self._last_device = flexium.auto.get_device()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check for device changes after each training batch.

        Migration happens transparently via the heartbeat thread.
        This hook detects if a migration occurred and syncs the
        trainer's device reference.

        Parameters:
            trainer: The Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            outputs: Output from the training step.
            batch: The current batch.
            batch_idx: Index of the current batch.
        """
        if self.disabled or self._flexium_context is None:
            return

        # Check if device changed (migration occurred)
        current_device = flexium.auto.get_device()
        if current_device != self._last_device:
            # Migration happened - sync trainer's device reference
            sync_device_to_trainer(trainer, current_device)
            self._last_device = current_device

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Check for device changes after each validation batch."""
        if self.disabled or self._flexium_context is None:
            return

        current_device = flexium.auto.get_device()
        if current_device != self._last_device:
            sync_device_to_trainer(trainer, current_device)
            self._last_device = current_device

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Save Flexium state to checkpoint.

        Parameters:
            trainer: The Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            checkpoint: The checkpoint dictionary to save to.
        """
        if self.disabled or self._flexium_context is None:
            return

        # Save Flexium state for potential resume
        checkpoint["flexium"] = {
            "device": flexium.auto.get_device(),
            "physical_device": flexium.auto.get_physical_device(),
            "process_id": flexium.auto.get_process_id(),
        }

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Restore Flexium state from checkpoint.

        Note: The actual device may differ from the saved state if
        resuming on a different GPU. Flexium handles this transparently.

        Parameters:
            trainer: The Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            checkpoint: The checkpoint dictionary to load from.
        """
        if self.disabled:
            return

        # Log checkpoint info for debugging
        if "flexium" in checkpoint:
            flexium_state = checkpoint["flexium"]
            print(f"[flexium] Checkpoint was saved on: {flexium_state.get('device')}")

    def teardown(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Clean up Flexium context when training ends.

        Parameters:
            trainer: The Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        if self._flexium_context is not None:
            self._flexium_context.__exit__(None, None, None)
            self._flexium_context = None

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing.

        Returns:
            Dictionary containing callback configuration.
        """
        return {
            "orchestrator": self.orchestrator,
            "device": self.device,
            "disabled": self.disabled,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint.

        Parameters:
            state_dict: Dictionary containing callback configuration.
        """
        # Note: We don't restore orchestrator/device since they may
        # have changed. The current settings take precedence.
        pass
