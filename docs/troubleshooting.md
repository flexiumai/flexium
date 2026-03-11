# Troubleshooting

Common issues and their solutions when using flexium.

## Table of Contents

1. [Connection Issues](#connection-issues)
2. [Training Issues](#training-issues)
3. [Migration Issues](#migration-issues)
4. [Memory Issues](#memory-issues)
5. [GPU Error Recovery](#gpu-error-recovery)
6. [Debugging](#debugging)
7. [FAQ](#faq)

---

## Connection Issues

### Cannot connect to orchestrator

**Symptoms:**
- "Failed to connect to orchestrator"
- Process shows as "stale" immediately
- Heartbeat errors in logs

**Solutions:**

1. **Check orchestrator is running:**
   ```bash
   ps aux | grep "flexium-ctl server"
   # Should show the server process
   ```

2. **Verify port is correct:**
   ```bash
   # Check what port orchestrator is on
   netstat -tlnp | grep 50051
   ```

3. **Check network connectivity:**
   ```bash
   # Test connection
   nc -zv localhost 50051
   ```

4. **Check firewall:**
   ```bash
   # Allow port if needed
   sudo ufw allow 50051
   ```

5. **Try explicit address:**
   ```python
   # Instead of
   with flexium.auto.run():

   # Try
   with flexium.auto.run(orchestrator="127.0.0.1:50051"):
   ```

### Training process issues

**Solutions:**

1. **Enable debug logging:**
   ```bash
   # Enable debug logging
   FLEXIUM_LOG_LEVEL=DEBUG python train.py
   ```

2. **Check CUDA availability:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Process starts before orchestrator is ready

**Symptoms:**
- "Orchestrator not ready after 30s" warning
- Process runs in local mode even though orchestrator is running

**Solutions:**

Flexium automatically waits up to 30 seconds for the orchestrator to be ready. If your orchestrator takes longer to start:

1. **Wait for orchestrator to fully start:**
   ```bash
   # Start orchestrator
   flexium-ctl server --dashboard

   # Wait for "Orchestrator started" message before starting training
   ```

2. **The training will still work:**
   If the orchestrator isn't ready in time, training continues in local mode (graceful degradation). It will reconnect when the orchestrator becomes available.

---

## Training Issues

### Training doesn't start

**Symptoms:**
- No output from training
- Process hangs after "Process: gpu-XXXXXX"

**Solutions:**

1. **Check for import errors:**
   ```bash
   # Run without flexium first
   python train.py --disabled
   ```

2. **Check CUDA is available:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

3. **Check process output:**
   The process output should appear in your terminal. If not, check stderr.

### Model not on correct device

**Symptoms:**
- CUDA errors about tensor device mismatch
- Model on cuda:0 but data on cuda:1

**Solutions:**

1. **Use `.cuda()` consistently:**
   ```python
   with flexium.auto.run():
       model.cuda()       # Correct
       model.to("cuda")   # Correct
       model.to("cuda:0") # Correct - will be redirected
   ```

---

## Migration Issues

### Migration not happening

**Symptoms:**
- Click "Migrate" but nothing happens
- `flexium-ctl migrate` returns success but process stays on same GPU

**Solutions:**

1. **Check process status:**
   ```bash
   flexium-ctl list
   # Process should show "running", not "stale" or "migrating"
   ```

2. **Verify heartbeats are working:**
   The orchestrator only sends migration commands via heartbeat response.
   If heartbeats aren't working, migration won't happen.

3. **Check for pending migration:**
   If a migration is already pending, new requests are ignored.
   Wait for the current migration to complete.

4. **Same device check:**
   Migration to the same device is a no-op.

### Migration fails mid-way

**Symptoms:**
- "Checkpoint failed" error
- Old process killed but new one doesn't start

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h /tmp
   # Checkpoints need space
   ```

2. **Check checkpoint directory:**
   ```bash
   ls -la /tmp/flexium/checkpoints/
   ```

3. **Check for CUDA errors:**
   ```bash
   nvidia-smi
   # Ensure target GPU is available
   ```

### Training doesn't resume correctly after migration

**Symptoms:**
- Loss spikes after migration
- Different results than without migration

**Solutions:**

1. **RNG state issue:** Ensure RNG state is being saved/restored.
   This is automatic with flexium but custom RNG usage might not be captured.

2. **Learning rate scheduler:** If using a scheduler, ensure it's being saved:
   ```python
   # The scheduler state should be part of optimizer state
   # Or register it explicitly
   ```

3. **Custom state:** Register any custom state:
   ```python
   from flexium.tracking import get_tracker
   tracker = get_tracker()
   tracker.register_custom_state("my_state", my_state_dict)
   ```

---

## Memory Issues

### CUDA out of memory

**Symptoms:**
- "CUDA out of memory" error
- Process killed by OOM

**Solutions:**

1. **Reduce batch size:**
   ```python
   dataloader = DataLoader(dataset, batch_size=32)  # Try smaller
   ```

2. **Migrate to GPU with more memory:**
   ```bash
   # Find GPU with more free memory
   nvidia-smi

   # Migrate
   flexium-ctl migrate <process_id> cuda:X
   ```

3. **Clear cache before migration:**
   ```python
   torch.cuda.empty_cache()
   ```

### Memory not freed after migration

**Symptoms:**
- Old GPU still shows memory usage after migration
- nvidia-smi shows process on old GPU

**Solutions:**

This shouldn't happen with Flexium.AI's architecture. If it does:

1. **Check process is actually dead:**
   ```bash
   ps aux | grep train.py
   # Should only show one process (the new one)
   ```

2. **Check nvidia-smi:**
   ```bash
   nvidia-smi
   # Old GPU should show no process from your training
   ```

3. **Force cleanup:**
   ```bash
   # Kill any orphaned processes
   pkill -f "train.py"
   ```

---

## GPU Error Recovery

### OOM recovery not working

**Symptoms:**
- Process crashes on OOM instead of recovering
- No migration to another GPU

**Solutions:**

1. **Check orchestrator is connected:**
   ```python
   with flexium.auto.run(orchestrator="localhost:50051"):
       ...
   ```
   When connected to orchestrator, you can migrate to a GPU with more VRAM.

2. **Check other GPUs have enough VRAM:**
   ```bash
   nvidia-smi
   # Recovery GPU needs: memory_in_use + tried_to_allocate + 1GB buffer
   ```

3. **Manually migrate to larger GPU:**
   ```bash
   flexium-ctl migrate <process-id> cuda:1
   ```

### Recovery to CPU is slow

**Symptoms:**
- Training continues but much slower on CPU

**Solutions:**

CPU fallback is intentionally slow - it's for fault tolerance, not performance. Options:

1. **Wait for a GPU to become available:**
   The orchestrator will migrate back to GPU when one is free.

2. **Reduce batch size for CPU:**
   ```python
   device = flexium.auto.get_device()
   batch_size = 64 if device.startswith("cuda") else 16
   ```

3. **Stop and restart on GPU:**
   ```bash
   # Kill the process and start fresh on GPU
   pkill -f train.py
   python train.py
   ```

### GPU marked as unhealthy

**Symptoms:**
- Dashboard shows GPU with red "Unhealthy" badge
- Processes won't schedule on that GPU

**Solutions:**

1. **Check GPU hardware:**
   ```bash
   nvidia-smi -q -d ECC
   # Look for ECC errors
   ```

2. **Mark GPU as healthy (if issue resolved):**
   - In dashboard: Click "Mark Healthy" button on the GPU card
   - Via CLI: Not yet available

3. **Wait for auto-recovery:**
   Unhealthy GPUs automatically recover after 1 hour (default timeout).

### ECC errors

**Symptoms:**
- GPU marked unhealthy after training
- "ECC error" in logs

**Solutions:**

ECC errors indicate hardware memory problems. Options:

1. **Reboot the machine:**
   Sometimes clears transient ECC errors.

2. **Run GPU memory test:**
   ```bash
   nvidia-smi -q -d ECC
   ```

3. **Contact system admin:**
   Persistent ECC errors may indicate failing GPU.

---

## Debugging

### Enable Debug Logging

```bash
FLEXIUM_LOG_LEVEL=DEBUG python train.py
```

### Debug Mode

```python
# Disable flexium entirely for debugging
with flexium.auto.run(disabled=True):
    ...
```

### Automatic Debugger Detection

If you attach a debugger (PyCharm, VS Code, pdb), flexium automatically switches to debug mode.

### Check Process Status

```bash
# See what processes are running
flexium-ctl list

# Get detailed status
flexium-ctl list --json | jq '.[] | select(.process_id == "gpu-abc123")'
```

### Check Orchestrator Logs

```bash
# Run orchestrator with debug logging
FLEXIUM_LOG_LEVEL=DEBUG flexium-ctl server --dashboard
```

### Inspect Checkpoint

```python
import torch

checkpoint = torch.load("/tmp/flexium/checkpoints/gpu-abc123_1234567890.pt")
print(f"Device: {checkpoint.get('device', 'unknown')}")
print(f"GPU UUID: {checkpoint.get('gpu_uuid', 'unknown')}")
print(f"Models: {list(checkpoint.get('models', {}).keys())}")
```

---

## FAQ

### Q: What happens if the orchestrator crashes?

A: Training continues normally. The process will show as "stale" in the orchestrator when it restarts, but training is unaffected. Migration won't work until the orchestrator is back up.

### Q: Can I run multiple training jobs on the same GPU?

A: Yes, but they'll compete for memory. flexium doesn't prevent this - it just tracks and migrates processes.

### Q: Does flexium work with DataParallel/DistributedDataParallel?

A: Not currently. flexium is designed for single-GPU training per process. Multi-GPU training within a single process is a future enhancement.

### Q: What's the overhead of flexium?

A: Minimal, typically < 2%. The main overhead is:
- Device string comparison on .cuda()/.to() calls
- Iterator wrapping (negligible)
- Background heartbeat thread (minimal CPU)

### Q: Can I use flexium with mixed precision training?

A: Yes! flexium is transparent to mixed precision:
```python
with flexium.auto.run():
    model = Model().cuda()
    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Q: How do I exclude flexium in production?

A: Use the disabled flag:
```python
import os

with flexium.auto.run(disabled=os.environ.get("PRODUCTION") == "1"):
    ...
```

Or use environment variables:
```bash
# Development
python train.py

# Production
PRODUCTION=1 python train.py  # Bypasses flexium
```

### Q: Can the orchestrator run on a different machine?

A: Yes! Just specify the address:
```python
with flexium.auto.run(orchestrator="192.168.1.100:50051"):
    ...
```

Or via environment:
```bash
export GPU_ORCHESTRATOR=192.168.1.100:50051
```

---

## Getting Help

If you're still stuck:

1. Check the GitHub Issues on your repository
2. Enable debug logging and include the output
3. Include your environment details:
   ```bash
   python --version
   pip show torch
   nvidia-smi
   ```

---

## See Also

- [API Reference](api.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Examples](examples.md)
