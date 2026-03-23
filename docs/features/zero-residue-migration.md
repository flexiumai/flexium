# Zero-Residue Migration

Flexium's zero-residue migration ensures that when a training job migrates from one GPU to another, **no memory is left behind** on the source GPU.

**Driver Requirement:** NVIDIA 580+ for GPU migration

## How It Works

Traditional approaches (like `model.to(device)`) leave memory fragments due to PyTorch's caching allocator. Flexium uses driver-level migration:

1. **Capture** - Complete GPU state is captured at driver level
2. **Release** - Source GPU is completely freed (0 MB)
3. **Restore** - State is restored on target GPU

## Benefits

- **Full GPU Reclamation** - The source GPU is immediately available for other workloads
- **No Memory Fragmentation** - Clean memory state on both source and target GPUs
- **Seamless Continuation** - Training resumes exactly where it left off

## Usage

Zero-residue migration is automatic:

```python
import flexium
flexium.init()

# Your training code here
# Migration happens when triggered via dashboard
train_model()
```

**Or with explicit scope control:**
```python
import flexium.auto

with flexium.auto.run():
    train_model()
```

## Verification

You can verify zero-residue behavior by monitoring GPU memory:

```bash
# Before migration
nvidia-smi

# After migration - source GPU should show 0 MB used by the process
nvidia-smi
```
