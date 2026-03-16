# PyTorch Lightning Integration

Flexium integrates seamlessly with PyTorch Lightning, bringing zero-downtime migration to your Lightning training workflows.

## Installation

```bash
pip install flexium[lightning]
```

## Quick Start

```python
import flexium.auto
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class MyModel(pl.LightningModule):
    # Your model definition
    pass

# Wrap your training with flexium.auto.run()
with flexium.auto.run():
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, datamodule)
```

## How It Works

Flexium works transparently with Lightning - just wrap your training with `flexium.auto.run()`. When migration is triggered via the dashboard, Flexium captures the complete GPU state at driver level and restores it on the target GPU. Your Lightning training continues seamlessly.

## Supported Features

| Feature | Support |
|---------|---------|
| Single GPU Training | Yes |
| Multi-GPU DDP | Not yet supported |
| Mixed Precision | Yes |

## Example

See the complete example at [examples/lightning/mnist_lightning.py](https://github.com/flexiumai/flexium/blob/master/examples/lightning/mnist_lightning.py).
