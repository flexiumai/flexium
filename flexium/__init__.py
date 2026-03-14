"""Flexium - Dynamic GPU orchestration for PyTorch training.

This package provides GPU orchestration with live migration support using
driver-level migration for zero-residue GPU migration (requires driver 580+).

Usage:

    import flexium.auto

    with flexium.auto.run():
        model = nn.Linear(784, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(100):
            for batch in dataloader:
                data, target = batch[0].cuda(), batch[1].cuda()
                loss = model(data).sum()
                loss.backward()
                optimizer.step()

Configuration:
    - Set FLEXIUM_SERVER environment variable to server address (host:port/workspace)
    - Or pass orchestrator= parameter to flexium.auto.run()

Migration is transparent - training continues in the same process,
same loop iteration, just on a different GPU.
"""

from __future__ import annotations

__version__ = "0.2.0a1"

# The main API is flexium.auto.run()
# Import with: import flexium.auto
# Usage: with flexium.auto.run(): ...

__all__ = [
    "__version__",
]
