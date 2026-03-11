<p align="center">
  <img src="assets/logo_with_text.png" alt="Flexium Logo" width="400">
</p>

<h1 align="center">Flexium</h1>

<p align="center">
  <strong>Zero-downtime GPU migration for PyTorch training</strong><br>
  Move running training jobs between GPUs without stopping them.
</p>

<p align="center">
  <a href="https://pypi.org/project/flexium/"><img src="https://img.shields.io/pypi/v/flexium.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/flexium/"><img src="https://img.shields.io/pypi/pyversions/flexium.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="https://flexium.ai">Sign Up Free</a>
</p>

---

## What is Flexium?

Flexium enables **live GPU migration** for PyTorch training. Move your training between GPUs without stopping, checkpointing, or losing progress.

```python
import flexium.auto

with flexium.auto.run():
    # Your existing training code - unchanged
    for epoch in range(100):
        for batch in dataloader:
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()
    # Training transparently migrates between GPUs when needed
```

**Zero code changes. Zero downtime. Zero memory residue.**

## Features

- **Live Migration** - Move training between GPUs without stopping
- **Zero Code Changes** - Just wrap your training loop
- **Zero Memory Residue** - Source GPU is completely freed
- **Works Everywhere** - PyTorch, Lightning, HuggingFace, and more
- **Web Dashboard** - Visual GPU management and monitoring
- **Spot Instance Support** - Survive preemptions automatically

## Quick Start

### 1. Install

```bash
pip install flexium
```

### 2. Get Your API Key

Sign up for free at [https://flexium.ai](https://flexium.ai)

### 3. Add Two Lines to Your Training

```python
import flexium.auto

with flexium.auto.run():
    # Your existing training code - no changes needed
    model = MyModel().cuda()
    for epoch in range(100):
        train_one_epoch(model)
```

That's it! Your training is now migratable.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR TRAINING CODE                        │
│                                                              │
│   with flexium.auto.run():                                  │
│       model.train()   ◄─── Training runs normally           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Heartbeat
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLEXIUM CLOUD                              │
│                                                              │
│   Monitors your GPUs and triggers migrations when needed    │
│   - Spot preemption? Auto-migrate to another GPU            │
│   - GPU failure? Auto-recover to healthy GPU                │
│   - Rebalancing? Move to optimize utilization               │
└─────────────────────────────────────────────────────────────┘
```

When migration is needed, Flexium:
1. **Captures** complete GPU state at driver level
2. **Frees** source GPU completely (0 MB residue)
3. **Restores** on target GPU seamlessly

Your training code never knows it moved.

## Use Cases

### Spot Instance Training
```python
# Survive spot preemptions without losing progress
with flexium.auto.run():
    train_large_model()  # Auto-migrates when spot is reclaimed
```

### GPU Rebalancing
```python
# Free up GPUs for teammates without killing your job
with flexium.auto.run():
    train_model()  # Can be migrated via dashboard anytime
```

### Hardware Failure Recovery
```python
# Auto-recover from GPU errors
with flexium.auto.run():
    train_model()  # Auto-migrates on ECC errors, driver crashes
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with driver 580+
- CUDA 12.0+
- Linux x86_64

## Pricing

| Tier | Price | GPUs | Support |
|------|-------|------|---------|
| **Free** | $0/mo | Up to 4 | Community |
| **Pro** | $99/mo | Unlimited | Email (48h) |
| **Team** | $499/mo | Unlimited | Priority (24h) |
| **Enterprise** | Custom | Unlimited | Dedicated |

[**Sign up free →**](https://flexium.ai)

## Documentation

- [Getting Started](https://docs.flexium.ai/getting-started)
- [API Reference](https://docs.flexium.ai/api)
- [Examples](https://docs.flexium.ai/examples)
- [Troubleshooting](https://docs.flexium.ai/troubleshooting)

## Support

- **Docs**: [docs.flexium.ai](https://docs.flexium.ai)
- **Discord**: [Join our community](https://discord.gg/flexium)
- **Issues**: [GitHub Issues](https://github.com/flexiumai/flexium/issues)
- **Email**: support@flexium.ai

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ by the Flexium team</strong><br>
  <a href="https://flexium.ai">flexium.ai</a>
</p>
