<p align="center">
  <img src="https://raw.githubusercontent.com/flexiumai/flexium/main/assets/logo_with_text.png" alt="Flexium.AI Logo" width="350">
</p>

<p align="center">
  <strong>Flexible Resource Allocation</strong> - Seamlessly migrate PyTorch training between GPUs with zero interruption.<br>
  Your model continues from exactly where it left off, and the source GPU is completely freed with zero VRAM residue.
</p>

<p align="center">
  <a href="https://pypi.org/project/flexium/"><img src="https://img.shields.io/pypi/v/flexium.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/flexium/"><img src="https://img.shields.io/pypi/pyversions/flexium.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="https://docs.flexium.ai">Documentation</a> •
  <a href="https://app.flexium.ai">Dashboard</a>
</p>

---

## What is Flexium?

Flexium is a GPU orchestration system that enables **dynamic device migration** for PyTorch training jobs. It allows training processes to be moved between GPUs **without leaving any memory traces** on the source device.

### Key Features

- **Seamless Migration**: Training continues from the exact batch where it stopped
- **Zero VRAM Residue**: When a process migrates, the source GPU has **0 MB** used
- **Minimal Code Changes**: Just 2 lines to integrate
- **Web Dashboard**: Real-time monitoring and one-click migration
- **Works Offline**: Training continues even if server connection is lost
- **No Server Installation**: Just `pip install flexium` - no agents or daemons needed
- **Framework Compatibility**: Works with PyTorch Lightning, Hugging Face, timm, FastAI, and more

---

## Quick Start

### 1. Install

```bash
pip install flexium
```

### 2. Sign Up

Create a free account at [app.flexium.ai](https://app.flexium.ai)

### 3. Add Two Lines to Your Training

```python
import flexium
flexium.init()  # That's it!

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

### 4. Run with Your Workspace

```bash
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"
python train.py
```

Your training is now migration-enabled. Monitor and migrate via the [dashboard](https://app.flexium.ai).

---

## The Problem

Traditional approaches to GPU migration leave memory fragments:

```python
# This doesn't fully free memory!
model = model.to("cuda:1")  # Old GPU still has memory residue
torch.cuda.empty_cache()     # Doesn't guarantee cleanup
```

## The Solution

Flexium uses **driver-level migration** that guarantees complete memory release:

```
┌───────────────────────────────────────┐
│        Training on OLD GPU            │
│                                       │
│  Your PyTorch code runs normally      │
│                                       │
└───────────────────────────────────────┘
                   │
                   │  MIGRATE
                   │  (100% memory freed!)
                   ▼
┌───────────────────────────────────────┐
│        Training on NEW GPU            │
│                                       │
│  Resumes from exact position          │
│  No progress lost                     │
│                                       │
└───────────────────────────────────────┘
```

---

## How It Works

```
┌───────────────────────────────────────────────────────────┐
│                      YOUR GPU MACHINE                     │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                  Training Process                   │  │
│  │  - Your PyTorch training code                       │  │
│  │  - Initialized with flexium.init()                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└───────────────────────────────────────────────────────────┘
                            │
                            │ Communicates with
                            ▼
┌───────────────────────────────────────────────────────────┐
│                 FLEXIUM CLOUD (flexium.ai)                │
│                                                           │
│         Web dashboard for monitoring and control          │
└───────────────────────────────────────────────────────────┘
```

1. **Sign Up**: Create a free account at [app.flexium.ai](https://app.flexium.ai)
2. **Connect**: Set `FLEXIUM_SERVER` and run your training
3. **Monitor & Migrate**: Use the web dashboard for one-click migration

---

## Use Cases

### GPU Sharing
```python
import flexium
flexium.init()
# Free up GPUs for teammates without killing your job
train_model()  # Can be migrated via dashboard anytime
```

### Pause for Priority Jobs
```python
import flexium
flexium.init()
# Pause your job to free GPU, resume later on any available GPU
train_model()  # Pause via dashboard, resume when ready
```

### GPU Error Recovery
```python
import flexium
flexium.init()

for batch in dataloader:
    with flexium.auto.recoverable():
        # If OOM occurs, migrates to healthy GPU and continues
        loss = model(batch.cuda()).sum()
        loss.backward()
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Linux x86_64
- NVIDIA Driver:
    - **550+** for pause/resume (same GPU)
    - **580+** for GPU migration (different GPU)

---

## Documentation

- [Getting Started](https://docs.flexium.ai/getting-started)
- [Installation Guide](https://docs.flexium.ai/installation)
- [Architecture](https://docs.flexium.ai/ARCHITECTURE)
- [API Reference](https://docs.flexium.ai/api)
- [Examples](https://docs.flexium.ai/examples)
- [Troubleshooting](https://docs.flexium.ai/troubleshooting)

### Features
- [Zero-Residue Migration](https://docs.flexium.ai/features/zero-residue-migration)
- [GPU Error Recovery](https://docs.flexium.ai/features/gpu-error-recovery)
- [Pause/Resume](https://docs.flexium.ai/features/pause-resume)
- [Framework Compatibility](https://docs.flexium.ai/features/framework-compatibility)

---

## Become a Design Partner

We're looking for **design partners** to explore advanced capabilities:

- Automatic migration based on resource optimization
- Distributed training support (DDP/FSDP)
- Integration with job schedulers (Slurm/Kubernetes)
- Multi-node GPU orchestration

If you're managing multi-GPU servers and want to shape the future of GPU orchestration, [contact us](mailto:flexium.ai@gmail.com?subject=Design%20Partner%20Inquiry).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see our [GitHub repository](https://github.com/flexiumai/flexium) to report issues or submit pull requests.
