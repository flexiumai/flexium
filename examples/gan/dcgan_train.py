#!/usr/bin/env python
"""DCGAN training with flexium.

A Deep Convolutional GAN trained on CIFAR-10.
Demonstrates handling of two models (generator + discriminator)
and alternating optimization steps.

Note: This example uses flexium.auto.run() for explicit scope control.
For simpler integration, use flexium.init() instead:
    import flexium
    flexium.init()

Usage:
    # With flexium (default)
    python dcgan_train.py

    # With orchestrator
    python dcgan_train.py --orchestrator localhost:80

    # Without flexium (baseline)
    python dcgan_train.py --disabled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flexium.auto
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Generator(nn.Module):
    """DCGAN Generator network.

    Takes a latent vector and generates an image.
    """

    def __init__(self, latent_dim: int = 100, channels: int = 3, features: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # State: (features*8) x 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # State: (features*4) x 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # State: (features*2) x 16 x 16
            nn.ConvTranspose2d(features * 2, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output: channels x 32 x 32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """DCGAN Discriminator network.

    Takes an image and outputs a probability of it being real.
    """

    def __init__(self, channels: int = 3, features: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: channels x 32 x 32
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: features x 16 x 16
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*2) x 8 x 8
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*4) x 4 x 4
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


def weights_init(m: nn.Module) -> None:
    """Initialize weights as described in the DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DCGAN training with flexium")
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--disabled", action="store_true", help="Run without flexium")
    args = parser.parse_args()

    # Create output directories
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    with flexium.auto.run(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    ):
        # Data
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        # Models
        generator = Generator(args.latent_dim).cuda()
        discriminator = Discriminator().cuda()

        # Initialize weights
        generator.apply(weights_init)
        discriminator.apply(weights_init)

        # Optimizers
        opt_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Loss
        criterion = nn.BCELoss()

        # Fixed noise for visualization
        fixed_noise = torch.randn(64, args.latent_dim).cuda()

        print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
        print("-" * 50)

        for epoch in range(args.epochs):
            g_losses = []
            d_losses = []

            for i, (real_images, _) in enumerate(dataloader):
                batch_size = real_images.size(0)
                real_images = real_images.cuda()

                # Labels
                real_labels = torch.ones(batch_size).cuda()
                fake_labels = torch.zeros(batch_size).cuda()

                # ---------------------
                # Train Discriminator
                # ---------------------
                opt_d.zero_grad()

                # Real images
                output_real = discriminator(real_images)
                loss_d_real = criterion(output_real, real_labels)

                # Fake images
                noise = torch.randn(batch_size, args.latent_dim).cuda()
                fake_images = generator(noise)
                output_fake = discriminator(fake_images.detach())
                loss_d_fake = criterion(output_fake, fake_labels)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                # ---------------------
                # Train Generator
                # ---------------------
                opt_g.zero_grad()

                output = discriminator(fake_images)
                loss_g = criterion(output, real_labels)

                loss_g.backward()
                opt_g.step()

                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}] "
                        f"Batch [{i}/{len(dataloader)}] "
                        f"Loss_D: {loss_d.item():.4f} "
                        f"Loss_G: {loss_g.item():.4f}"
                    )

            # Epoch summary
            avg_g = sum(g_losses) / len(g_losses)
            avg_d = sum(d_losses) / len(d_losses)
            print(f">>> Epoch {epoch} | Avg Loss_D: {avg_d:.4f} | Avg Loss_G: {avg_g:.4f}")

            # Save sample images
            with torch.no_grad():
                fake = generator(fixed_noise)
                save_image(
                    fake,
                    samples_dir / f"epoch_{epoch:03d}.png",
                    normalize=True,
                    nrow=8,
                )

        print("Training complete!")
        print(f"Sample images saved to {samples_dir}/")


if __name__ == "__main__":
    main()
