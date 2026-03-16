# GPU Error Recovery

Flexium provides automatic GPU error recovery, allowing training to survive and recover from common CUDA errors like OOM, ECC errors, and device asserts.

## Overview

GPU errors are a common cause of failed training runs:

- **OOM (Out of Memory)**: Batch size too large or memory fragmentation
- **ECC Errors**: Hardware memory corruption
- **Device Assert**: Kernel-level assertion failures
- **Illegal Memory Access**: Memory access violations
- **Launch Failures**: CUDA kernel launch issues

With Flexium's `recoverable()` context manager, your training can automatically:

1. Detect these errors
2. Clear the CUDA error state
3. Request a suitable GPU from the orchestrator
4. Migrate to the new GPU
5. Retry the failed operation

## Usage

Wrap your training step with the `recoverable()` iterator:

```python
import flexium.auto
import torch

with flexium.auto.run():
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            # Wrap potentially failing operations
            for attempt in flexium.auto.recoverable():
                with attempt:
                    data, target = batch[0].cuda(), batch[1].cuda()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
```

If an OOM error occurs, Flexium will:

1. Clear CUDA error state
2. Find a GPU with more available VRAM
3. Migrate your training to that GPU
4. Retry the batch

## Supported Errors

| Error Type | Description | Recovery Action |
|------------|-------------|-----------------|
| OOM | Out of memory | Migrate to GPU with more VRAM |
| ECC | Uncorrectable ECC error | Mark GPU unhealthy, migrate away |
| Device Assert | Device-side assertion | Migrate to healthy GPU |
| Illegal Access | Illegal memory access | Migrate to healthy GPU |
| Launch Failure | CUDA launch failure | Migrate to healthy GPU |

## Configuration

### Max Retries

Control how many recovery attempts are made:

```python
# Allow up to 5 recovery attempts (default is 3)
for attempt in flexium.auto.recoverable(max_retries=5):
    with attempt:
        # your training step
```

### Error Propagation

Non-CUDA errors and unrecognized errors are re-raised immediately:

```python
for attempt in flexium.auto.recoverable():
    with attempt:
        # This will be raised immediately (not a CUDA error)
        raise ValueError("Invalid input")
```

## Simple Context Manager

For simpler cases where you don't need automatic retry, use as a direct context manager:

```python
# Single attempt - errors are classified but not automatically retried
with flexium.auto.recoverable():
    output = model(data.cuda())
```

This is useful for error classification and logging, but won't automatically retry failed operations.

## Requirements

- **Migration must be enabled** (driver 580+ required)
- **Orchestrator connection** for finding alternative GPUs
- **Multiple GPUs available** for OOM recovery to work

If migration is disabled or no alternative GPU is available, the original error is re-raised after exhausting retries.

## How It Works

1. **Error Detection**: When a CUDA error occurs inside the `recoverable()` block, it's caught and classified

2. **Error State Clearing**: CUDA error state is cleared via:
   - `torch.cuda.synchronize()`
   - `torch.cuda.empty_cache()`
   - `torch.cuda.reset_peak_memory_stats()`

3. **Recovery Target**: For OOM errors, Flexium parses the error message to estimate memory needed and requests a GPU with sufficient free VRAM

4. **Migration**: If a suitable GPU is found, your training is migrated there using zero-residue migration

5. **Retry**: The failed code block is re-executed on the new GPU

## Example: OOM Recovery

```python
import flexium.auto
import torch

with flexium.auto.run():
    model = LargeModel().cuda()

    for batch in dataloader:
        for attempt in flexium.auto.recoverable(max_retries=3):
            with attempt:
                # This might OOM on smaller GPUs
                output = model(batch.cuda())
                loss = output.sum()
                loss.backward()

        # Continues here after successful attempt
        optimizer.step()
        optimizer.zero_grad()
```

Output when OOM occurs:
```
[flexium] GPU error: OOM (attempt 1/3)
[flexium] Recovering: migrating to cuda:2...
[flexium] Recovery successful - now on cuda:2, retrying operation...
```

## Limitations

- Recovery requires an orchestrator connection (doesn't work in standalone mode)
- Only works with supported CUDA error types
- If all GPUs are exhausted or unsuitable, the error is eventually re-raised
- Some errors (like ECC) may indicate hardware problems that persist
