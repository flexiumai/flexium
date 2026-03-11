# GPU Error Recovery

Flexium provides automatic recovery from common GPU errors, keeping your training jobs running.

## Supported Error Types

- **CUDA Out of Memory (OOM)** - Automatic batch size adjustment or migration
- **GPU Hardware Errors** - ECC errors, thermal throttling detection
- **Driver Issues** - Graceful handling of driver timeouts
- **Communication Failures** - Recovery from multi-GPU sync issues

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Training   │────▶│   Flexium    │────▶│  Recovery   │
│   Error     │     │   Handler    │     │   Action    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Orchestrator │
                    │  Migration   │
                    └──────────────┘
```

## Recovery Strategies

### Automatic Migration

When a GPU error is detected, Flexium can automatically migrate your training to a healthy GPU:

```python
import flexium

flexium.auto.patch()

# Training will automatically migrate on GPU errors
for epoch in range(100):
    train_epoch()
```

### Graceful Degradation

For multi-GPU training, Flexium can continue with fewer GPUs:

```python
from flexium.config import FlexiumConfig

config = FlexiumConfig(
    allow_degradation=True,
    min_gpus=1,  # Minimum GPUs to continue training
)
```

## Monitoring

GPU health is continuously monitored:

- Memory utilization and errors
- Temperature and power consumption
- PCIe bandwidth and errors
- Driver and CUDA status
