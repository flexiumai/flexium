# Flexium.AI

<p align="center">
  <img src="assets/logo_with_text.png" alt="Flexium.AI Logo" width="350">
</p>

**Flexible Resource Allocation** - Dynamic GPU orchestration for PyTorch training with zero VRAM residue migration.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in 5 minutes with just 2 lines of code.

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-cube-outline:{ .lg .middle } __Architecture__

    ---

    Understand how flexium guarantees zero memory residue.

    [:octicons-arrow-right-24: Architecture](ARCHITECTURE.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete documentation of all public APIs.

    [:octicons-arrow-right-24: API Reference](api.md)

-   :material-code-tags:{ .lg .middle } __Examples__

    ---

    Working examples from simple to production-ready.

    [:octicons-arrow-right-24: Examples](examples.md)

</div>

---

## What is Flexium?

Flexium is a GPU orchestration system that enables **dynamic device migration** for PyTorch training jobs. It allows training processes to be moved between GPUs **without leaving any memory traces** on the source device.

### Key Features

- **Zero VRAM Residue**: When a process migrates, the source GPU has **0 MB** used
- **Seamless Migration**: Training continues from the exact batch where it stopped
- **Minimal Code Changes**: As few as 2 lines to integrate
- **Remote Orchestration**: Manage GPUs across your cluster
- **Web Dashboard**: Real-time monitoring and one-click migration
- **GPU Error Recovery**: Automatic recovery from OOM, device assert, and ECC errors
- **Graceful Degradation**: Works standalone without orchestrator (no single point of failure)
- **GPU UUID Support**: Target specific physical GPUs for reproducibility

### The Problem

Traditional approaches to GPU migration leave memory fragments:

```python
# This doesn't fully free memory!
model = model.to("cuda:1")  # Old GPU still has memory residue
torch.cuda.empty_cache()     # Doesn't guarantee cleanup
```

### The Solution

Flexium uses **proprietary migration technology** (requires NVIDIA driver 580+) that guarantees complete memory release:

```
┌─────────────────────────────────────────┐
│         Training on OLD GPU             │
│                                         │
│   Your PyTorch code runs normally       │
│                                         │
└─────────────────────────────────────────┘
                    │
                    │  MIGRATE
                    │  (100% memory freed!)
                    ▼
┌─────────────────────────────────────────┐
│         Training on NEW GPU             │
│                                         │
│   Resumes from exact position           │
│   No progress lost                      │
│                                         │
└─────────────────────────────────────────┘
```

---

## Quick Example

### Before (Standard PyTorch)

```python
import torch

model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    for batch in dataloader:
        data = batch.cuda()
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
```

### After (With Flexium)

```python
import flexium.auto  # Add this line
import torch

with flexium.auto.run():  # Add this line
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            data = batch.cuda()
            loss = model(data).sum()
            loss.backward()
            optimizer.step()
```

**That's it!** Your training is now migration-enabled.

---

## Installation

```bash
pip install flexium
```

Or from source:

```bash
git clone https://github.com/flexiumai/flexium.git
cd flexium
pip install -e .
```

See the [Installation Guide](installation.md) for detailed instructions including:

- System requirements and driver compatibility
- PyTorch with CUDA setup
- Environment configuration
- Troubleshooting common issues

### Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- **NVIDIA Driver 580+** (required for zero-residue migration)
- Linux x86_64

**Note:** Flexium requires PyTorch with CUDA support. Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your system.

---

## How It Works

1. **Start the Orchestrator**: Central server that tracks all training jobs
   ```bash
   flexium-ctl server --dashboard
   ```

2. **Run Your Training**: With flexium enabled
   ```bash
   python train.py
   ```

3. **Monitor & Migrate**: Via web dashboard or CLI
   ```bash
   flexium-ctl list
   flexium-ctl migrate gpu-abc123 cuda:1
   ```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│              (Central server, can be remote)                        │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐    │
│   │ gRPC Server │  │  Registry   │  │    Web Dashboard        │    │
│   │   :50051    │  │             │  │       :8080             │    │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU MACHINE                                   │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                    Training Process                           │ │
│   │  - Your PyTorch training code                                 │ │
│   │  - Managed by flexium for migration                           │ │
│   │  - Communicates with orchestrator                             │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │          │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases

### Dynamic GPU Allocation

Move training jobs between GPUs based on demand:

```bash
# High-priority job needs GPU 0
flexium-ctl migrate low-priority-job cuda:1
```

### Memory Management

Free up a GPU for a larger model:

```bash
# Move small job to make room
flexium-ctl migrate small-job cuda:2
nvidia-smi  # GPU 0 now has more free memory
```

### Fault Tolerance

If a GPU fails, migrate affected jobs:

```bash
flexium-ctl list --device cuda:3  # Find jobs on failing GPU
flexium-ctl migrate job-1 cuda:0
flexium-ctl migrate job-2 cuda:1
```

### Automatic GPU Error Recovery

GPU errors (OOM, device assert, ECC) are automatically detected and trigger recovery:

```python
import flexium.auto

with flexium.auto.run():
    # If OOM occurs, automatically migrates to GPU with more VRAM
    model = Net().cuda()
    ...
```

### Development Workflow

Test on GPU 0, then move to production GPU:

```bash
# Development
python train.py  # Runs on cuda:0

# Move to production GPU without stopping
flexium-ctl migrate my-job cuda:7
```

---

## Why Flexium?

<div class="grid cards" markdown>

-   :material-memory:{ .lg .middle } __Zero VRAM Residue__

    ---

    Unlike `model.to(device)`, migration **guarantees** 100% memory is freed. Flexium's architecture ensures complete GPU release.

-   :material-flash:{ .lg .middle } __Automatic Error Recovery__

    ---

    GPU errors (OOM, device assert, ECC) are detected and handled automatically. Training migrates to a healthy GPU and resumes from checkpoint.

-   :material-shield-check:{ .lg .middle } __No Single Point of Failure__

    ---

    Works standalone without orchestrator. If connection is lost, training continues normally. Reconnects automatically when available.

-   :material-chart-line:{ .lg .middle } __Real-Time Dashboard__

    ---

    Monitor all training jobs, GPU utilization, and memory usage. One-click migration between devices.

-   :material-code-tags:{ .lg .middle } __Minimal Code Changes__

    ---

    Just 2 lines of code to enable. No changes to your training logic, model, or dataloader.

-   :material-target:{ .lg .middle } __GPU UUID Targeting__

    ---

    Target specific physical GPUs by UUID for reproducibility and hardware-specific debugging.

</div>

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Quick start guide |
| [Installation](installation.md) | Detailed installation guide |
| [Architecture](ARCHITECTURE.md) | How flexium works |
| [API Reference](api.md) | Complete API documentation |
| [Examples](examples.md) | Code examples |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

### Feature Documentation

| Feature | Description |
|---------|-------------|
| [Zero-Residue Migration](features/zero-residue-migration.md) | Driver-level migration with zero VRAM residue |
| [Pause/Resume](features/pause-resume.md) | Pause training to free GPU, resume later |
| [GPU Error Recovery](features/gpu-error-recovery.md) | Automatic recovery from OOM, device assert, ECC errors |
| [Graceful Degradation](features/graceful-degradation.md) | Standalone mode without orchestrator |
| [Lightning Integration](features/lightning-integration.md) | PyTorch Lightning support with FlexiumCallback |

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see our [GitHub repository](https://github.com/flexiumai/flexium) to report issues or submit pull requests.
