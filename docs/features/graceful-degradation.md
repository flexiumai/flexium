# Graceful Degradation

Flexium supports graceful degradation, allowing training to continue even when some GPUs become unavailable.

## Overview

In distributed training scenarios, GPU failures don't have to mean job failure. Flexium can automatically adjust your training to use fewer GPUs while maintaining training integrity.

## How It Works

1. **Detection** - Flexium detects GPU unavailability (failure, preemption, etc.)
2. **Assessment** - Determines if training can continue with remaining GPUs
3. **Adjustment** - Reconfigures distributed training for the new GPU count
4. **Continuation** - Training resumes with adjusted parallelism

## Configuration

```python
from flexium.config import FlexiumConfig

config = FlexiumConfig(
    allow_degradation=True,
    min_gpus=1,          # Minimum GPUs required
    rebalance_data=True, # Redistribute data across remaining GPUs
)
```

## Behavior

| Original GPUs | Available GPUs | Action |
|--------------|----------------|--------|
| 8 | 7 | Continue with 7, adjust batch distribution |
| 8 | 4 | Continue with 4, rebalance workload |
| 8 | 0 | Pause and wait for GPU availability |

## Use Cases

- **Spot Instance Training** - Continue when some spot instances are reclaimed
- **Shared Clusters** - Adapt when GPUs are reallocated
- **Hardware Issues** - Maintain training despite partial failures

## Limitations

- Learning rate and batch size may need manual adjustment for optimal convergence
- Some distributed strategies may not support arbitrary GPU counts
- Checkpoint compatibility requires same model architecture
