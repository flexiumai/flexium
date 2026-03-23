# Getting Started with Flexium.AI

This guide will help you set up Flexium.AI and run your first migration-enabled training job.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA support
- NVIDIA Driver:
    - **550+** for pause/resume (same GPU)
    - **580+** for GPU migration (different GPU)
- Linux x86_64 (Windows/macOS not yet supported)

## Installation

### From PyPI

```bash
pip install flexium
```

### From Source

```bash
git clone https://github.com/flexiumai/flexium.git
cd flexium
pip install -e .
```

### PyTorch Installation

Flexium requires PyTorch with CUDA support. Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your system and CUDA version.

### Dependencies

Core dependencies (installed automatically):

- `python-socketio[client]>=5.0.0` - WebSocket communication
- `pynvml>=11.0.0` - NVIDIA Management Library for GPU monitoring

Development dependencies:

- `pytest>=7.0.0` - Testing framework

## Quick Start

### Step 1: Connect to Flexium Server

Flexium is a cloud-hosted service. Set the `FLEXIUM_SERVER` environment variable with your workspace:

```bash
# Format: host:port/workspace
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"
```

Sign up for free at [app.flexium.ai](https://app.flexium.ai) to create your workspace.

### Step 2: Add flexium to Your Training Script

Add just 2 lines to your existing code:

```python
import flexium
flexium.init()  # That's it!

# ... your existing imports ...
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Your model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# Training - no changes needed!
model = Net().cuda()  # Standard PyTorch!
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    for batch in dataloader:
        data, target = batch[0].cuda(), batch[1].cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

That's it! Your training is now migration-enabled.

### Step 3: Run Your Training

```bash
# Set server address with workspace
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"

# Run your script normally
python train.py
```

### Step 4: Monitor and Migrate

Open your workspace dashboard at [app.flexium.ai](https://app.flexium.ai) to:

- See all running training jobs
- Monitor GPU utilization
- Trigger migrations with one click
- Pause and resume training jobs

## Configuration

### Environment Variables (Recommended)

```bash
# Server with workspace (required)
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"

# Optional: default device
export GPU_DEVICE=cuda:0
```

### Config File

Create `~/.flexiumrc`:

```yaml
server: app.flexium.ai/myworkspace
device: cuda:0
```

### Inline Parameters

```python
import flexium
flexium.init(server="app.flexium.ai/myworkspace", device="cuda:0")
```

### Explicit Scope Control (Advanced)

For advanced use cases where you need explicit scope control:

```python
import flexium.auto

with flexium.auto.run(orchestrator="app.flexium.ai/myworkspace"):
    # Training code with explicit scope
    ...
```

## Verification

### Check Driver Version

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# 550+ for pause/resume, 580+ for GPU migration
```

### Check Installation

```bash
python -c "import flexium; print('OK')"
```

### Test Migration

1. Set your server connection:
   ```bash
   export FLEXIUM_SERVER="app.flexium.ai/myworkspace"
   ```

2. Run example:
   ```bash
   python examples/simple/mnist_train_auto.py
   ```

3. Open your workspace dashboard

4. Click "Migrate" to move training to another GPU

5. Watch training continue seamlessly!

6. Verify zero residue with `nvidia-smi` - source GPU should show 0 MB for flexium process

## Connection Resilience

Flexium automatically handles connection issues:

- On connection loss, you'll see: `[flexium] Lost connection, attempting reconnect...`
- On successful reconnection: `[flexium] Reconnected!`
- Training continues uninterrupted during brief outages

## Next Steps

- [Architecture Overview](ARCHITECTURE.md) - Understand how it works
- [API Reference](api.md) - Full API documentation
- [Examples](examples.md) - More code examples
- [Troubleshooting](troubleshooting.md) - Common issues

## Getting Help

- GitHub Issues: Report bugs or request features on your repository
- Documentation: You're reading it!
