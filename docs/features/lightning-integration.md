# PyTorch Lightning Integration

Flexium integrates seamlessly with PyTorch Lightning, bringing zero-downtime migration to your Lightning training workflows.

## Installation

```bash
pip install flexium[lightning]
```

## Quick Start

```python
import flexium
flexium.auto.patch()

import pytorch_lightning as pl
from pytorch_lightning import Trainer

class MyModel(pl.LightningModule):
    # Your model definition
    pass

# Training works normally - Flexium handles migration automatically
trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

## How It Works

Flexium hooks into Lightning's training loop to:

1. Monitor for migration signals from the orchestrator
2. Capture Lightning-specific state (callbacks, loggers, etc.)
3. Handle checkpoint saving/loading transparently
4. Restore training at the exact step and epoch

## Callback Integration

Flexium provides a Lightning callback for additional control:

```python
from flexium.integrations.lightning import FlexiumCallback

trainer = Trainer(
    max_epochs=100,
    callbacks=[FlexiumCallback()]
)
```

## Supported Features

| Feature | Support |
|---------|---------|
| Single GPU Training | Yes |
| Multi-GPU DDP | Yes |
| Model Checkpointing | Yes |
| Learning Rate Schedulers | Yes |
| Custom Callbacks | Yes |
| Mixed Precision | Yes |

## Example

See the complete example at [examples/lightning/mnist_lightning.py](https://github.com/flexiumai/flexium/blob/master/examples/lightning/mnist_lightning.py).
