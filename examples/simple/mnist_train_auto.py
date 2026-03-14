#!/usr/bin/env python
"""MNIST training with TRANSPARENT flexium (minimal code changes).

This demonstrates the flexium.auto API - just ONE import and ONE context manager.
Everything else is standard PyTorch code!

Compare with mnist_train.py which uses the explicit API.

Usage:
    # With orchestrator (set via env or config file)
    export FLEXIUM_SERVER=localhost:50051/workspace
    python examples/mnist_train_auto.py

    # With orchestrator (inline)
    python examples/mnist_train_auto.py --orchestrator localhost:50051

    # Baseline benchmark (no flexium - good for PyCharm debugging)
    python examples/mnist_train_auto.py --disabled

    # Local mode (no orchestrator, still get device management)
    python examples/mnist_train_auto.py --orchestrator ""

PyCharm Tips:
    - For debugging, use --disabled flag (runs without flexium)
    - Set working directory to the project root
    - Add --epochs 1 for quick testing
    - Example Run Configuration:
        Script: examples/mnist_train_auto.py
        Parameters: --disabled --epochs 1
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# === THIS IS THE ONLY IMPORT YOU NEED ===
import flexium.auto


class Net(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self) -> None:
        super().__init__()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST with transparent flexium")
    parser.add_argument(
        "--orchestrator",
        default=None,
        help="Orchestrator address (e.g., localhost:50051/workspace). Uses FLEXIUM_SERVER env var if not set.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Initial device (or set GPU_DEVICE env var)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs (default: 10)",
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

    # Set global seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # === THIS IS THE ONLY CHANGE TO YOUR TRAINING CODE ===
    with flexium.auto.run(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    ):
        # =====================================================
        # EVERYTHING BELOW IS 100% STANDARD PYTORCH CODE
        # No ctx.device, no iterate() wrapper, nothing special!
        # =====================================================

        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_data = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        )

        # Create DataLoader with fixed seed for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            generator=generator,
            num_workers=0,  # Use 0 for easier debugging
        )

        print(f"\nDataLoader created:")
        print(f"  Dataset size: {len(train_data)}")
        print(f"  Batch count: {len(train_loader)}")
        print()

        # Model setup - just use .cuda() as normal!
        model = Net().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop - standard PyTorch
        # Migration happens transparently via heartbeat thread
        for epoch in range(args.epochs):
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Just use .cuda() - flexium handles device routing!
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch:2d} | Batch {batch_idx:4d} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Acc: {100.0 * correct / total:.1f}%"
                    )

            epoch_time = time.time() - epoch_start
            if len(train_loader) > 0 and total > 0:
                print(
                    f">>> Epoch {epoch} done | "
                    f"Avg Loss: {total_loss / len(train_loader):.4f} | "
                    f"Acc: {100.0 * correct / total:.1f}% | "
                    f"Time: {epoch_time:.2f}s\n"
                )


if __name__ == "__main__":
    main()
