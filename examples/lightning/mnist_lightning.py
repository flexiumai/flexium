#!/usr/bin/env python
"""MNIST training with PyTorch Lightning and Flexium.

This demonstrates Flexium integration with PyTorch Lightning using
the FlexiumCallback. Migration happens transparently via the heartbeat
thread - no changes needed to standard Lightning code!

Compare with examples/simple/mnist_train_auto.py which uses raw PyTorch.

Usage:
    # With orchestrator
    export GPU_ORCHESTRATOR=localhost:80
    python examples/lightning/mnist_lightning.py

    # With orchestrator (inline)
    python examples/lightning/mnist_lightning.py --orchestrator localhost:80

    # Baseline benchmark (no flexium - good for debugging)
    python examples/lightning/mnist_lightning.py --disabled

PyCharm Tips:
    - For debugging, use --disabled flag
    - Set working directory to the project root
    - Add --epochs 1 for quick testing
"""

from __future__ import annotations

import argparse
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from flexium.lightning import FlexiumCallback


class MNISTModel(pl.LightningModule):
    """Simple CNN for MNIST classification."""

    def __init__(self, learning_rate: float = 0.001) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)

        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MNISTDataModule(pl.LightningDataModule):
    """DataModule for MNIST dataset."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self) -> None:
        # Download data
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=self.transform,
        )
        self.val_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=self.transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNIST with PyTorch Lightning and Flexium"
    )
    parser.add_argument(
        "--orchestrator",
        default="localhost:80",
        help="Orchestrator address (e.g., localhost:80)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Initial device (or set GPU_DEVICE env var)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Disable flexium for baseline benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Create model and data module
    model = MNISTModel()
    datamodule = MNISTDataModule(batch_size=args.batch_size)

    # === THIS IS THE ONLY CHANGE FOR FLEXIUM ===
    # Add FlexiumCallback to enable transparent GPU migration
    flexium_callback = FlexiumCallback(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    )

    # Create trainer with Flexium callback
    # Everything else is standard PyTorch Lightning!
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[flexium_callback],
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Train the model
    # Migration happens transparently via heartbeat thread
    trainer.fit(model, datamodule)

    print("\nTraining complete!")
    if not args.disabled:
        import flexium.auto
        print(f"Final device: {flexium.auto.get_device()}")


if __name__ == "__main__":
    main()
