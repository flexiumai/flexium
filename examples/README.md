# Flexium Examples

Example scripts demonstrating flexium's transparent GPU migration capabilities.

## Quick Start

```bash
# 1. Set your server and workspace
export FLEXIUM_SERVER="localhost:50051/myworkspace"

# 2. Run an example
python examples/simple/mnist_train_auto.py

# 3. Open your workspace dashboard and click "Migrate" to move training between GPUs
```

## Directory Structure

### `simple/` - Getting Started
Basic examples to learn flexium:
- **mnist_train_auto.py** - MNIST training with minimal code changes (start here!)
- **migration_demo.py** - Simple migration demonstration
- **test_migration.py** - Verify migration works correctly
- **test_epoch_tracking.py** - Test epoch/batch tracking

### `gan/` - Generative Adversarial Networks
- **dcgan_train.py** - DCGAN on CIFAR-10 (handles 2 models + 2 optimizers)

### `diffusion/` - Diffusion Models
- **ddpm_train.py** - Denoising Diffusion Probabilistic Model

### `llm/` - Language Models
- **gpt_train.py** - GPT-style character-level transformer

### `vision/` - Vision Models
- **vit_train.py** - Vision Transformer on CIFAR-10

### `lightning/` - PyTorch Lightning Integration
- Lightning-compatible training examples

## Requirements

Most examples require:
- PyTorch with CUDA support
- 1+ NVIDIA GPUs (2+ for migration demos)
- Connection to a Flexium server (`FLEXIUM_SERVER` environment variable)

## Environment Variables

```bash
# Required: Server address with workspace
export FLEXIUM_SERVER="localhost:50051/myworkspace"

# Optional: Default device
export GPU_DEVICE=cuda:0
```

## Common Options

Most training examples support:
```bash
--epochs N                # Number of training epochs
--disabled                # Run without flexium (baseline comparison)
--verbose                 # Extra logging
```
