# GPU Error Recovery

Flexium provides automatic GPU error recovery, allowing training to survive and recover from common CUDA errors like OOM, ECC errors, and device asserts.

## Overview

GPU errors are a common cause of failed training runs:

- **OOM (Out of Memory)**: Batch size too large or memory fragmentation
- **ECC Errors**: Hardware memory corruption
- **Device Assert**: Kernel-level assertion failures
- **Illegal Memory Access**: Memory access violations
- **Launch Failures**: CUDA kernel launch issues

With Flexium's `recoverable()`, your training can automatically detect these errors, migrate to a healthy GPU, and optionally retry the failed operation.

## Three Ways to Use `recoverable()`

Flexium provides three patterns, from simplest to most control:

---

### Option 1: Simple Context Manager (Recommended)

**The current operation is LOST, but training continues on the new GPU.**

This is the simplest approach. If a GPU error occurs, Flexium migrates to a new GPU and suppresses the exception. The code inside the `with` block that failed is **not retried** - that batch/operation is lost, but training continues with the next iteration.

```python
import flexium.auto

with flexium.auto.run():
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for batch in dataloader:
        with flexium.auto.recoverable():
            # If OOM happens here, this batch is LOST
            # but we migrate to new GPU and continue
            output = model(batch.cuda())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Training continues here on next batch
```

**Output when OOM occurs:**
```
[flexium] GPU error: OOM
[flexium] WARNING: The current operation is LOST. Migrating to new GPU...
[flexium] Migrating to cuda:2...
[flexium] Migration complete. Training continues (current batch was lost).
```

!!! warning "Operation is Lost"
    With this pattern, the failing operation (batch) is **not retried**. For most deep learning training, losing one batch is acceptable. If you need to retry the exact same operation, use the decorator or iterator pattern below.

---

### Option 2: Decorator (Replays the Operation)

**The operation is RETRIED on the new GPU.**

Wrap your training step in a function with the `@recoverable` decorator. If a GPU error occurs, Flexium migrates to a new GPU and **calls the function again** with the same arguments.

```python
import flexium.auto

@flexium.auto.recoverable(retries=3)
def train_step(model, batch, optimizer, criterion):
    output = model(batch.cuda())
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

with flexium.auto.run():
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for batch in dataloader:
        loss = train_step(model, batch, optimizer, criterion)
        # If OOM happened, train_step was retried on new GPU
```

You can also use `@recoverable` without parentheses (uses default 3 retries):

```python
@flexium.auto.recoverable
def train_step(model, batch):
    ...
```

**Output when OOM occurs:**
```
[flexium] GPU error: OOM (attempt 1/3)
[flexium] Recovering: migrating to cuda:2...
[flexium] Recovery successful - now on cuda:2, retrying operation...
```

---

### Option 3: Iterator Pattern (Advanced)

**Most control over retry logic.**

You write the retry loop structure. This is useful when you need custom logic between retries.

```python
import flexium.auto

with flexium.auto.run():
    model = Net().cuda()

    for batch in dataloader:
        for attempt in flexium.auto.recoverable(retries=3):
            with attempt:
                output = model(batch.cuda())
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

---

## Supported Errors

| Error Type | Description | Recovery Action |
|------------|-------------|-----------------|
| OOM | Out of memory | Migrate to GPU with more VRAM |
| ECC | Uncorrectable ECC error | Mark GPU unhealthy, migrate away |
| Device Assert | Device-side assertion | Migrate to healthy GPU |
| Illegal Access | Illegal memory access | Migrate to healthy GPU |
| Launch Failure | CUDA launch failure | Migrate to healthy GPU |

## Configuration

### Retries (Decorator/Iterator only)

Control how many retry attempts are made:

```python
# Decorator with custom retries
@flexium.auto.recoverable(retries=5)
def train_step():
    ...

# Iterator with custom retries
for attempt in flexium.auto.recoverable(retries=5):
    with attempt:
        ...
```

### Error Propagation

Non-CUDA errors and unrecognized RuntimeErrors are **always re-raised immediately**:

```python
with flexium.auto.recoverable():
    raise ValueError("Not a CUDA error")  # Re-raised immediately
```

## Requirements

- **Migration must be enabled** (NVIDIA driver 580+ required)
- **Orchestrator connection** for finding alternative GPUs
- **Multiple GPUs available** for recovery to work

If migration is disabled or no alternative GPU is available:
- Simple context manager: the original error is re-raised
- Decorator/Iterator: retries are attempted, then the error is re-raised

## How It Works

1. **Error Detection**: CUDA errors are caught and classified by type

2. **Error State Clearing**: CUDA error state is cleared via:
   - `torch.cuda.synchronize()`
   - `torch.cuda.empty_cache()`
   - `torch.cuda.reset_peak_memory_stats()`

3. **Recovery Target**: For OOM errors, Flexium parses the error message to estimate memory needed and requests a GPU with sufficient free VRAM

4. **Migration**: Training is migrated using zero-residue migration

5. **Continuation/Retry**:
   - Simple context manager: Exception is suppressed, training continues
   - Decorator: Function is called again with same arguments
   - Iterator: Next iteration of the for loop runs

## Limitations

- Recovery requires an orchestrator connection (doesn't work in standalone mode)
- Only works with supported CUDA error types
- If all GPUs are exhausted or unsuitable, the error is eventually re-raised
- Some errors (like ECC) may indicate hardware problems that affect all GPUs
