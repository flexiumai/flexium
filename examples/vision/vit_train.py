#!/usr/bin/env python
"""Vision Transformer (ViT) training with flexium.

A Vision Transformer for CIFAR-10 image classification.
Demonstrates patch embedding, positional encoding,
and transformer encoder for vision tasks.

Usage:
    # With flexium (default)
    python vit_train.py

    # With orchestrator
    python vit_train.py --orchestrator localhost:80

    # Without flexium (baseline)
    python vit_train.py --disabled
"""

from __future__ import annotations

import argparse

import flexium.auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, n_h, n_w) -> (B, n_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    """Standard transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        n_classes: int = 10,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # Learnable class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for block in self.encoder:
            x = block(x)

        # Classification head (use class token)
        x = self.norm(x)
        x = x[:, 0]  # Class token
        x = self.head(x)

        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT training with flexium")
    parser.add_argument(
        "--orchestrator",
        default="localhost:80",
        help="Orchestrator address (e.g., localhost:80). Use empty string for local mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Initial device (or set GPU_DEVICE env var)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--disabled", action="store_true", help="Run without flexium")
    args = parser.parse_args()

    with flexium.auto.run(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    ):
        # Data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010],
            ),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010],
            ),
        ])

        train_dataset = datasets.CIFAR10(
            "./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            "./data", train=False, transform=test_transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Model
        model = ViT(
            img_size=32,
            patch_size=4,
            n_classes=10,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).cuda()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        best_acc = 0.0

        for epoch in range(args.epochs):
            # Training
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"Acc: {100.*correct/total:.2f}%"
                    )

            scheduler.step()

            # Evaluation
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * correct / total
            test_acc = 100.0 * test_correct / test_total
            avg_loss = total_loss / len(train_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                # Save best model
                torch.save(model.state_dict(), "vit_best.pt")

            print(
                f">>> Epoch {epoch} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Best: {best_acc:.2f}%"
            )

        print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
