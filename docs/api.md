# API Reference

Complete API documentation for flexium.

**Requirements:** NVIDIA Driver 580+, PyTorch 2.0+, Linux x86_64

## Table of Contents

1. [Quick Start](#quick-start)
2. [flexium.auto.run()](#flexiumautorun)
3. [flexium.auto.recoverable()](#flexiumautorecoverable)
4. [flexium.auto.get_device()](#flexiumautoget_device)
5. [flexium.auto.is_migration_enabled()](#flexiumautois_migration_enabled)
6. [Additional Auto APIs](#additional-auto-apis)
7. [PyTorch Lightning](#pytorch-lightning)
8. [Configuration](#configuration)
9. [Dashboard Controls](#dashboard-controls)

---

## Quick Start

```python
import flexium.auto

with flexium.auto.run():
    # Standard PyTorch code - no changes needed!
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            data, target = batch[0].cuda(), batch[1].cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

That's it! Your training is now migratable via the Flexium dashboard.

---

## flexium.auto.run()

Context manager for transparent GPU management with live migration support.

```python
@contextmanager
def run(
    orchestrator: Optional[str] = None,
    device: Optional[str] = None,
    disabled: bool = False,
) -> Iterator[None]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orchestrator` | `Optional[str]` | `None` | Flexium server address (host:port/workspace). If None, uses config/env |
| `device` | `Optional[str]` | `None` | Initial device. If None, uses config/env or "cuda:0" |
| `disabled` | `bool` | `False` | Bypass flexium entirely (for benchmarking) |

### Configuration Resolution

Priority (highest to lowest):
1. Parameters passed to `run()`
2. Environment variables (`FLEXIUM_SERVER`, `GPU_DEVICE`)
3. Config file (`./.flexiumrc` or `~/.flexiumrc`)
4. Default (local mode with warning)

### Examples

**Basic usage:**
```python
import flexium.auto

with flexium.auto.run():
    model = Net().cuda()
    # Training code...
```

**With Flexium server:**
```python
with flexium.auto.run(orchestrator="app.flexium.ai/myworkspace"):
    model = Net().cuda()
```

**Specific device:**
```python
with flexium.auto.run(device="cuda:2"):
    model = Net().cuda()  # Goes to cuda:2
```

**Benchmarking (disabled):**
```python
with flexium.auto.run(disabled=True):
    # Pure PyTorch, no flexium overhead
    model = Net().cuda()
```

**Environment variables:**
```bash
export FLEXIUM_SERVER=app.flexium.ai/myworkspace
export GPU_DEVICE=cuda:1
python train.py
```

### What Gets Patched

When inside `run()`, the following PyTorch functions are patched for device routing:

| Function | Behavior |
|----------|----------|
| `tensor.cuda()` | Routes to managed device |
| `module.cuda()` | Routes to managed device |

These patches ensure that after Flexium swaps GPU identities during migration, user code calling `.cuda()` still goes to the correct physical GPU.

### Warning Message

If no server is configured:

```
============================================================
[flexium] WARNING: No server configured!
[flexium] Running in local mode (no migration support)
[flexium]
[flexium] To enable Flexium, set FLEXIUM_SERVER:
[flexium]   export FLEXIUM_SERVER=app.flexium.ai/myworkspace
============================================================
```

---

## flexium.auto.recoverable()

Automatic GPU error recovery - migrate and optionally retry on CUDA errors.

```python
class recoverable:
    def __init__(self, retries: int = 3):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retries` | `int` | `3` | Maximum retry attempts (decorator/iterator only) |

### Supported Errors

| Error Type | Detection |
|------------|-----------|
| OOM | `torch.cuda.OutOfMemoryError` or message contains "out of memory" |
| ECC | Message contains "uncorrectable ECC error" |
| Device Assert | Message contains "device-side assert" |
| Illegal Access | Message contains "illegal memory access" |
| Launch Failure | Message contains "launch failure" |

### Three Usage Patterns

**Option 1: Simple Context Manager (operation is LOST)**

```python
with flexium.auto.recoverable():
    output = model(batch.cuda())  # Lost if OOM, training continues
```

**Option 2: Decorator (operation is RETRIED)**

```python
@flexium.auto.recoverable(retries=3)
def train_step(model, batch):
    return model(batch.cuda())

train_step(model, batch)  # Retried on OOM
```

**Option 3: Iterator (advanced, operation is RETRIED)**

```python
for attempt in flexium.auto.recoverable(retries=3):
    with attempt:
        output = model(batch.cuda())  # Retried on OOM
```

### Raises

- `RuntimeError`: If recovery fails after `retries` attempts (decorator/iterator only)
- Original exception: If the error is not a recoverable CUDA error

### Notes

- **Simple context manager**: Operation is lost, training continues
- **Decorator/Iterator**: Operation is retried on new GPU
- Requires migration (driver 580+) and orchestrator connection
- Non-CUDA errors are always re-raised immediately

For more details, see [GPU Error Recovery](features/gpu-error-recovery.md).

---

## flexium.auto.get_device()

Get the current managed device.

```python
def get_device() -> str:
```

### Returns

Current device string (e.g., `"cuda:0"`, `"cuda:1"`, `"cpu"`).

### Example

```python
import flexium.auto

with flexium.auto.run():
    device = flexium.auto.get_device()
    print(f"Training on: {device}")

    # Can be used for manual tensor placement
    custom_tensor = torch.zeros(100).to(device)
```

---

## flexium.auto.is_migration_enabled()

Check if migration and pause functionality is available.

```python
def is_migration_enabled() -> bool:
```

### Returns

`True` if environment requirements are met (CUDA available, driver requirements met).
`False` if requirements are not met - training continues but migration/pause are disabled.

### Example

```python
import flexium.auto

with flexium.auto.run():
    if flexium.auto.is_migration_enabled():
        print("Migration available - can migrate via dashboard")
    else:
        print("Migration disabled - requirements not met")

    # Training works either way
    model = Net().cuda()
    ...
```

### Notes

At startup, Flexium verifies:
- CUDA is available via PyTorch
- NVIDIA driver 580+ is installed

If requirements are not met, specific warnings are logged and training continues in degraded mode.

---

## Additional Auto APIs

These APIs are useful for advanced use cases and framework integrations (like PyTorch Lightning).

### flexium.auto.get_physical_device()

Get the physical device after migration.

```python
def get_physical_device() -> str:
```

After migration, this reflects the actual GPU the process is running on.

### flexium.auto.is_active()

Check if inside a `flexium.auto.run()` context.

```python
def is_active() -> bool:
```

### flexium.auto.is_migration_in_progress()

Check if a migration is currently happening.

```python
def is_migration_in_progress() -> bool:
```

### flexium.auto.get_process_id()

Get the Flexium process ID.

```python
def get_process_id() -> str:
```

Returns a string like `"gpu-abc12345"`.

---

## PyTorch Lightning

Flexium integrates with PyTorch Lightning via `FlexiumCallback`.

### Quick Start

```python
from pytorch_lightning import Trainer
from flexium.lightning import FlexiumCallback

trainer = Trainer(
    callbacks=[FlexiumCallback()],
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, dataloader)
```

### FlexiumCallback

```python
from flexium.lightning import FlexiumCallback

class FlexiumCallback(Callback):
    def __init__(
        self,
        orchestrator: Optional[str] = None,
        device: Optional[str] = None,
        disabled: bool = False,
    ) -> None:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orchestrator` | `str` | `None` | Flexium server address |
| `device` | `str` | `None` | Initial device |
| `disabled` | `bool` | `False` | Disable Flexium |

### Example

```python
from pytorch_lightning import Trainer, LightningModule
from flexium.lightning import FlexiumCallback

class MyModel(LightningModule):
    # Your standard Lightning module - no changes needed
    ...

trainer = Trainer(
    callbacks=[FlexiumCallback(orchestrator="app.flexium.ai/myworkspace")],
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, dataloader)
```

For more details, see [Lightning Integration](features/lightning-integration.md).

---

## Configuration

### Config File Format

```yaml
# ~/.flexiumrc or ./.flexiumrc
server: app.flexium.ai/myworkspace
device: cuda:0
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FLEXIUM_SERVER` | Flexium server address | `app.flexium.ai/workspace` |
| `GPU_DEVICE` | Initial device | `cuda:0` |
| `FLEXIUM_LOG_LEVEL` | Logging level | `DEBUG` |

### Programmatic Configuration

```python
from flexium.config import load_config

# Load with defaults
config = load_config()

# Override specific values
config = load_config(orchestrator="app.flexium.ai/myworkspace", device="cuda:2")

# Access values
print(config.orchestrator)  # "app.flexium.ai/myworkspace"
print(config.device)        # "cuda:2"
```

---

## Dashboard Controls

All process management is done through the web dashboard at [app.flexium.ai](https://app.flexium.ai).

### Process Management

View all your training processes in the dashboard:
- See process status, GPU assignment, memory usage
- Filter by device or status

### Migration

To migrate a process:
1. Find the process in the dashboard
2. Click the "Migrate" button
3. Select the target GPU
4. Migration happens seamlessly

### Pause/Resume

To pause a process (frees GPU completely):
1. Find the process in the dashboard
2. Click "Pause"
3. GPU memory is freed immediately

To resume:
1. Find the paused process
2. Click "Resume"
3. Optionally select a specific GPU, or let Flexium choose

### Device Status

The dashboard shows real-time status of all GPUs:
- Memory usage per GPU
- Running processes on each GPU
- GPU health status

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)
