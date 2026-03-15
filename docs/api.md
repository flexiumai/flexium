# API Reference

Complete API documentation for flexium.

**Requirements:** NVIDIA Driver 580+, PyTorch 2.0+, Linux x86_64

## Table of Contents

1. [Quick Start](#quick-start)
2. [flexium.auto.run()](#flexiumautorun)
3. [flexium.auto.get_device()](#flexiumautoget_device)
4. [flexium.auto.is_migration_enabled()](#flexiumautois_migration_enabled)
5. [Additional Auto APIs](#additional-auto-apis)
6. [PyTorch Lightning](#pytorch-lightning)
7. [Configuration](#configuration)
8. [CLI Reference](#cli-reference)
9. [Orchestrator Client](#orchestrator-client)

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

That's it! Your training is now migratable via the orchestrator.

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
| `orchestrator` | `Optional[str]` | `None` | Orchestrator address (host:port). If None, uses config/env |
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

**With orchestrator:**
```python
with flexium.auto.run(orchestrator="localhost:80"):
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
export FLEXIUM_SERVER=localhost:80
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

If no orchestrator is configured:

```
============================================================
[flexium] WARNING: No orchestrator configured!
[flexium] Running in local mode (no migration support)
[flexium]
[flexium] To enable orchestrator, either:
[flexium]   - Set FLEXIUM_SERVER=host:port environment variable
[flexium]   - Create ~/.flexiumrc with: orchestrator: host:port
[flexium]   - Pass orchestrator='host:port' to run()
============================================================
```

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
| `orchestrator` | `str` | `None` | Orchestrator address |
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
    callbacks=[FlexiumCallback(orchestrator="localhost:80")],
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
orchestrator: localhost:80
device: cuda:0
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FLEXIUM_SERVER` | Orchestrator address | `localhost:80` |
| `GPU_DEVICE` | Initial device | `cuda:0` |
| `FLEXIUM_LOG_LEVEL` | Logging level | `DEBUG` |

### Programmatic Configuration

```python
from flexium.config import load_config

# Load with defaults
config = load_config()

# Override specific values
config = load_config(orchestrator="myserver:80", device="cuda:2")

# Access values
print(config.orchestrator)  # "myserver:80"
print(config.device)        # "cuda:2"
```

---

## CLI Reference

### Server Commands

```bash
# Start orchestrator
flexium-ctl server [PORT] [OPTIONS]

Options:
  --dashboard           Enable web dashboard
  --dashboard-port PORT Dashboard port (default: 8080)
```

**Examples:**
```bash
flexium-ctl server                    # Port 80, no dashboard
flexium-ctl server 50052              # Custom port
flexium-ctl server --dashboard        # With dashboard at :8080
flexium-ctl server --dashboard --dashboard-port 9000
```

### Process Management

```bash
# List processes
flexium-ctl list [OPTIONS]

Options:
  --device DEVICE       Filter by device
  --json                Output as JSON
```

**Examples:**
```bash
flexium-ctl list
flexium-ctl list --device cuda:0
flexium-ctl list --json
```

### Migration

```bash
# Migrate a process
flexium-ctl migrate PROCESS_ID TARGET_DEVICE
```

**Examples:**
```bash
flexium-ctl migrate gpu-abc123 cuda:1
flexium-ctl migrate gpu-abc123 cuda:2
```

### Pause/Resume

```bash
# Pause a process (frees GPU completely)
flexium-ctl pause PROCESS_ID

# Resume a paused process
flexium-ctl resume PROCESS_ID [DEVICE]
```

**Examples:**
```bash
flexium-ctl pause gpu-abc123
flexium-ctl resume gpu-abc123           # Resume on any available GPU
flexium-ctl resume gpu-abc123 cuda:2    # Resume on specific GPU
```

### Device Status

```bash
# Show device status
flexium-ctl devices
```

---

## Orchestrator Client

For programmatic control of the orchestrator.

### `OrchestratorClient`

```python
from flexium.orchestrator.client import OrchestratorClient

client = OrchestratorClient("localhost:80")
```

### Methods

**`list_processes()`**
```python
def list_processes(device_filter: str = "") -> List[Dict[str, Any]]:
    """List all registered processes."""
```

**`request_migration()`**
```python
def request_migration(process_id: str, target_device: str) -> bool:
    """Request migration for a process."""
```

**`pause()`**
```python
def pause(process_id: str) -> bool:
    """Pause a process (free GPU)."""
```

**`resume()`**
```python
def resume(process_id: str, target_device: Optional[str] = None) -> bool:
    """Resume a paused process."""
```

### Example

```python
from flexium.orchestrator.client import OrchestratorClient

client = OrchestratorClient("localhost:80")

# List all processes
processes = client.list_processes()
for p in processes:
    print(f"{p['process_id']}: {p['device']} - {p['status']}")

# Request migration
client.request_migration("gpu-abc123", "cuda:1")

# Pause/resume
client.pause("gpu-abc123")
client.resume("gpu-abc123", "cuda:2")
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [Troubleshooting](troubleshooting.md)
