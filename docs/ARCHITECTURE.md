# Flexium Architecture

This document provides an overview of the flexium system architecture.

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Goals](#design-goals)
3. [Core Architecture](#core-architecture)
4. [Migration Mechanism](#migration-mechanism)
5. [Component Overview](#component-overview)
6. [Configuration System](#configuration-system)

---

## System Overview

Flexium is a GPU orchestration system that enables **live GPU migration** for PyTorch training jobs. It allows training processes to be moved between GPUs **without losing progress** and with **zero memory residue** on the source GPU.

### Key Capabilities

- **Zero VRAM Residue**: When a process migrates, ALL memory is freed from the source GPU
- **In-Process Migration**: Training continues in the same process, same loop iteration
- **Transparent Integration**: Just wrap your code with `flexium.auto.run()`
- **Remote Orchestration**: Central server coordinates migrations across a cluster
- **Web Dashboard**: Visual GPU management at `http://localhost:8080`
- **Pause/Resume**: Free GPU completely, resume later on any available GPU

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR SERVER                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────────┐  │
│  │ gRPC Server  │  │ Process       │  │ Device Registry             │  │
│  │ :50051       │  │ Registry      │  │ (tracks GPU health/usage)   │  │
│  └──────────────┘  └───────────────┘  └─────────────────────────────┘  │
│         │                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Web Dashboard (:8080)                          │  │
│  │         Real-time process monitoring and migration control        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC (heartbeats, migration commands)
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PROCESS                                 │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    flexium.auto.run()                              │ │
│  │                                                                    │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │ │
│  │  │ Orchestrator│  │ Heartbeat    │  │ Migration Handler      │   │ │
│  │  │ Client      │  │ Thread       │  │ Migration Handler      │   │ │
│  │  └─────────────┘  └──────────────┘  └────────────────────────┘   │ │
│  │                                                                    │ │
│  │  ┌───────────────────────────────────────────────────────────────┐│ │
│  │  │              User's Training Code                             ││ │
│  │  │  model.cuda(), optimizer.step(), etc.                         ││ │
│  │  └───────────────────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Goals

### 1. Zero VRAM Residue (Primary Goal)

**Problem**: Traditional approaches to GPU migration (moving tensors with `.to()`) leave memory fragments due to PyTorch's caching allocator.

**Solution**: Flexium's proprietary migration technology captures and restores the complete GPU state, guaranteeing zero residue.

### 2. In-Process Migration

Unlike subprocess-based approaches, flexium migrates within the same process:
- No process restart required
- Training continues from the exact same point
- All Python state preserved (variables, loop counters, etc.)

### 3. Minimal User Code Changes

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

## Core Architecture

### Single-Process Model

Flexium uses an in-process architecture:

1. **Training Process**: Runs user's training code with flexium wrapper
2. **Orchestrator Server**: Central coordination (can be remote)
3. **Dashboard**: Web UI for monitoring and control

### Why In-Process?

| Approach | Pros | Cons |
|----------|------|------|
| Subprocess isolation | Simple, guaranteed cleanup | Process restart, state serialization |
| **In-process (Flexium)** | **No restart, seamless** | **Requires driver 580+** |

Flexium uses proprietary migration technology that provides truly seamless migration - the training loop doesn't even know it happened.

---

## Migration Mechanism

### Migration Overview

Flexium's migration technology (requires driver 580+):
1. Captures complete GPU state
2. Restores on a different GPU
3. Guarantees zero residue on source GPU

### Migration Flow

```
Time ─────────────────────────────────────────────────────────────────────►

ORCHESTRATOR                    TRAINING PROCESS
     │                              │
     │  Heartbeat Response          │
     │  (should_migrate=true,       │
     │   target=cuda:1)             │
     │ ─────────────────────────────►
     │                              │
     │                              │  1. Pause training (between batches)
     │                              │
     │                              │  2. Capture GPU state
     │                              │
     │                              │  3. Release source GPU
     │                              │     - Source GPU now at 0 MB
     │                              │
     │                              │  4. Restore on target GPU
     │                              │
     │                              │  5. Resume training
     │                              │     - Same batch, same loop iteration
     │                              │
     │  Heartbeat                   │
     │  (device=cuda:1)             │
     │ ◄─────────────────────────────
```

---

## Component Overview

### 1. flexium.auto (`flexium/auto.py`)

The main user-facing API:

- **`run()`**: Context manager that enables migration
- **`get_device()`**: Returns current device string
- Heartbeat thread for orchestrator communication
- Migration execution via Flexium's migration engine

### 2. Orchestrator (`flexium/orchestrator/`)

Central coordination server:

- **`server.py`**: gRPC server implementation
- **`registry.py`**: Process registry (tracks all processes)
- **`device_registry.py`**: GPU health and utilization tracking
- **`client.py`**: Client for communicating with server

### 3. Dashboard (`flexium/dashboard/`)

Web-based monitoring and control:

- Real-time process list
- GPU utilization display
- One-click migration
- Pause/resume controls

### 4. CLI (`flexium/cli/`)

Command-line tools:

- **`flexium-ctl`**: Server management, process control
  - `flexium-ctl server` - Start orchestrator
  - `flexium-ctl list` - List processes
  - `flexium-ctl migrate <id> <device>` - Trigger migration
  - `flexium-ctl pause/resume` - Pause/resume processes

---

## Configuration System

### Configuration Priority (Highest to Lowest)

1. **Inline Parameters**
   ```python
   flexium.auto.run(orchestrator="host:port", device="cuda:1")
   ```

2. **Environment Variables**
   ```bash
   export GPU_ORCHESTRATOR=host:port
   export GPU_DEVICE=cuda:1
   ```

3. **Config File** (`~/.flexiumrc` or `./.flexiumrc`)
   ```yaml
   orchestrator: host:port
   device: cuda:0
   ```

4. **Defaults**
   - No orchestrator (local mode)
   - Device: cuda:0

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- **NVIDIA Driver 580+** (required for zero-residue migration)
- Linux x86_64

---

## Next Steps

For more information:

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Examples](docs/examples.md)
- [Troubleshooting](docs/troubleshooting.md)
