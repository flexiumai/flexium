# Framework Compatibility

Flexium works seamlessly with any PyTorch-based framework. No special integration code, callbacks, or wrappers needed - just call `flexium.init()` and your existing code works unchanged.

## Why It Just Works

Flexium operates at a level below your Python code, so any framework built on PyTorch works automatically. No special integration, callbacks, or wrappers needed.

## Supported Frameworks

| Framework | Status | Notes |
|-----------|--------|-------|
| **PyTorch** | ✅ Full support | Native support |
| **PyTorch Lightning** | ✅ Full support | No callback needed |
| **Hugging Face Transformers** | ✅ Full support | Models and Trainer |
| **Hugging Face Accelerate** | ✅ Full support | Single GPU |
| **timm** | ✅ Full support | All models |
| **torchvision** | ✅ Full support | Models and transforms |
| **FastAI** | ✅ Full support | Learner and DataLoaders |
| **PyTorch Geometric** | ✅ Full support | GNN models |
| **Detectron2** | ✅ Full support | Object detection |
| **MMDetection** | ✅ Full support | OpenMMLab ecosystem |

**Note:** Multi-GPU training (DDP, FSDP, DeepSpeed) is not yet supported. Flexium currently supports single-GPU training per process.

## Examples

### PyTorch Lightning

```python
import flexium
flexium.init()

import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    # Your standard Lightning module
    ...

trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(model, datamodule)
```

### Hugging Face Transformers

```python
import flexium
flexium.init()

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### Hugging Face Accelerate

```python
import flexium
flexium.init()

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(batch)
    loss = compute_loss(outputs)
    accelerator.backward(loss)
    optimizer.step()
```

### timm (PyTorch Image Models)

```python
import flexium
flexium.init()

import timm

model = timm.create_model('resnet50', pretrained=True).cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    images = batch['image'].cuda()
    outputs = model(images)
    loss = criterion(outputs, batch['label'].cuda())
    loss.backward()
    optimizer.step()
```

### FastAI

```python
import flexium
flexium.init()

from fastai.vision.all import *

dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

## How Migration Works

When you trigger a migration via the [dashboard](https://app.flexium.ai):

1. **Pause** - Training pauses at a safe point
2. **Capture** - GPU state is captured at driver level
3. **Move** - State is restored on target GPU
4. **Resume** - Training continues seamlessly

Your framework code never knows anything happened - the driver handles all the device remapping transparently.

## Requirements

- NVIDIA Driver 550+ (pause/resume) or 580+ (GPU migration)
- PyTorch 2.0+
- Linux x86_64
- Single GPU per process (multi-GPU support coming soon)

## Testing Framework Compatibility

We maintain integration tests for all supported frameworks. Run them with:

```bash
# Install test dependencies
pip install -e ".[test-frameworks]"

# Run framework tests
pytest tests/integration/test_framework_compatibility.py -v
```
