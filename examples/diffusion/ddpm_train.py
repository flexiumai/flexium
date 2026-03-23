#!/usr/bin/env python
"""DDPM (Denoising Diffusion) training with flexium.

A simplified implementation of DDPM for image generation.
Demonstrates handling of complex training loops with
timestep conditioning and noise scheduling.

Note: This example uses flexium.auto.run() for explicit scope control.
For simpler integration, use flexium.init() instead:
    import flexium
    flexium.init()

Usage:
    # With flexium (default)
    python ddpm_train.py

    # With orchestrator
    python ddpm_train.py --orchestrator localhost:80

    # Without flexium (baseline)
    python ddpm_train.py --disabled
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import flexium.auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timestep conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        # Use min(8, ch) groups to handle small channel counts (e.g., in_ch=1 for grayscale)
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    """Simple UNet for diffusion model noise prediction."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # Middle
        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder
        self.dec3 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final = nn.Conv2d(base_channels, in_channels, 1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)

        # Middle
        m = self.mid(self.down(e3), t_emb)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)

        return self.final(d1)


class DDPM:
    """DDPM noise schedule and sampling utilities."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
    ):
        self.timesteps = timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process - add noise to image."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def p_losses(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        """Calculate training loss (noise prediction MSE)."""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = model(x_t, t.float())
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Single denoising step."""
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x, t_tensor.float())

        # Calculate mean
        alpha = self.alphas[t]
        alpha_cumprod = self.alpha_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t]

        mean = (1 / torch.sqrt(alpha)) * (
            x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * noise_pred
        )

        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t])
            x = mean + variance * noise
        else:
            x = mean

        return x

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples from noise."""
        x = torch.randn(shape, device=self.device)

        timesteps = range(self.timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            x = self.p_sample(model, x, t)

        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM training with flexium")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--disabled", action="store_true", help="Run without flexium")
    args = parser.parse_args()

    # Create output directory
    samples_dir = Path("diffusion_samples")
    samples_dir.mkdir(exist_ok=True)

    with flexium.auto.run(
        orchestrator=args.orchestrator,
        device=args.device,
        disabled=args.disabled,
    ):
        # Data (MNIST for simplicity)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])
        dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        # Model
        model = UNet(in_channels=1, base_channels=64).cuda()
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(dataloader)
        )

        # Diffusion process
        ddpm = DDPM(timesteps=args.timesteps, device="cuda")

        print("-" * 50)

        for epoch in range(args.epochs):
            total_loss = 0

            for i, (images, _) in enumerate(dataloader):
                images = images.cuda()

                optimizer.zero_grad()
                loss = ddpm.p_losses(model, images)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}] "
                        f"Batch [{i}/{len(dataloader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(dataloader)
            print(f">>> Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            # Generate samples every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                samples = ddpm.sample(model, (16, 1, 32, 32), progress=False)
                samples = (samples + 1) / 2  # Scale back to [0, 1]
                save_image(
                    samples,
                    samples_dir / f"epoch_{epoch:03d}.png",
                    nrow=4,
                )
                model.train()

        print("Training complete!")
        print(f"Sample images saved to {samples_dir}/")


if __name__ == "__main__":
    main()
