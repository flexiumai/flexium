# Flexium Architecture

This document explains how Flexium enables live GPU migration for your training jobs.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Migration Mechanism](#migration-mechanism)
4. [Configuration](#configuration)

---

## Overview

Flexium enables **live GPU migration** for PyTorch training jobs. Your training can be moved between GPUs **without losing progress** and with **zero memory residue** on the source GPU.

### Key Capabilities

- **Zero VRAM Residue**: When a process migrates, ALL memory is freed from the source GPU
- **In-Process Migration**: Training continues in the same process, same loop iteration
- **Transparent Integration**: Just wrap your code with `flexium.auto.run()`
- **Pause/Resume**: Free GPU completely, resume later on any available GPU

### How You Use It

```
┌───────────────────────────────────────────────────────────┐
│                   YOUR TRAINING PROCESS                   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                  flexium.auto.run()                 │  │
│  │                                                     │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │            Your Training Code                 │  │  │
│  │  │  model.cuda(), optimizer.step(), etc.         │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
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
│   Web dashboard for monitoring and triggering migrations  │
└───────────────────────────────────────────────────────────┘
```

---

## How It Works

### Zero VRAM Residue

**Problem**: Traditional approaches to GPU migration (moving tensors with `.to()`) leave memory fragments due to PyTorch's caching allocator.

**Solution**: Flexium captures and restores the complete GPU state at driver level (driver 580+), guaranteeing zero residue.

### In-Process Migration

Unlike traditional approaches, Flexium migrates within the same process:
- No process restart required
- Training continues from the exact same point
- All Python state preserved (variables, loop counters, etc.)

### Minimal Code Changes

```python
import flexium.auto

with flexium.auto.run():
    # 100% standard PyTorch code
    model = Net().cuda()
    optimizer = Adam(model.parameters())
    for batch in dataloader:
        ...
```

---

## Migration Mechanism

When you trigger a migration from the dashboard:

1. **Pause** - Training pauses between batches
2. **Capture** - Complete GPU state is captured at driver level
3. **Release** - Source GPU is completely freed (0 MB)
4. **Restore** - State is restored on target GPU
5. **Resume** - Training continues from the exact same point

Your training code never knows it moved.

---

## Configuration

### Environment Variable (Recommended)

```bash
export FLEXIUM_SERVER="flexium.ai:80/myworkspace"
```

### Inline Parameter

```python
with flexium.auto.run(orchestrator="flexium.ai:80/myworkspace"):
    ...
```

### Config File (`~/.flexiumrc`)

```yaml
server: flexium.ai:80/myworkspace
device: cuda:0
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA 12.4+
- **NVIDIA Driver 580+** (required for zero-residue migration)
- Linux x86_64

---

## Next Steps

- [Getting Started](getting-started.md)
- [API Reference](api.md)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)
