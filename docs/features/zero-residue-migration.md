# Zero-Residue Migration

Flexium's zero-residue migration ensures that when a training job migrates from one GPU to another, **no memory is left behind** on the source GPU.

## How It Works

Traditional checkpoint-based migration leaves memory artifacts on the original GPU until the process terminates. Flexium uses a different approach:

1. **State Capture** - All model parameters, optimizer states, and training context are serialized
2. **Memory Release** - CUDA memory is explicitly freed before migration
3. **State Restoration** - The training state is restored on the target GPU

## Benefits

- **Full GPU Reclamation** - The source GPU is immediately available for other workloads
- **No Memory Fragmentation** - Clean memory state on both source and target GPUs
- **Seamless Continuation** - Training resumes exactly where it left off

## Usage

Zero-residue migration is enabled by default when using the Flexium driver:

```python
import flexium

flexium.auto.patch()

# Your training code here
# Migration happens automatically when orchestrated
```

## Verification

You can verify zero-residue behavior by monitoring GPU memory:

```bash
# Before migration
nvidia-smi

# After migration - source GPU should show 0 MB used by the process
nvidia-smi
```
