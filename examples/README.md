# Flexium Examples

Example scripts demonstrating flexium's transparent GPU migration capabilities.

## Quick Start

```bash
# 1. Start the orchestrator with dashboard
flexium-ctl server --dashboard

# 2. In another terminal, run an example
python examples/simple/mnist_train_auto.py

# 3. Open http://localhost:8080 and click "Migrate" to move training between GPUs
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

### `utils/` - Helper Scripts
- **run_demo.sh** - One-command demo (starts orchestrator + dashboard + MNIST training)
- **run_demo_2gpu.sh** - Same demo but limited to 2 GPUs (useful for shared machines)
- **run_orchestrator.py** - Start orchestrator programmatically from Python

## Requirements

Most examples require:
- PyTorch with CUDA support
- 1+ NVIDIA GPUs (2+ for migration demos)
- Running orchestrator (`flexium-ctl server --dashboard`)

## Common Options

Most training examples support:
```bash
--orchestrator HOST:PORT  # Orchestrator address (default: localhost:50051)
--epochs N                # Number of training epochs
--disabled                # Run without flexium (baseline comparison)
--verbose                 # Extra logging
```
