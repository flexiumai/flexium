# Troubleshooting

Common issues and their solutions when using flexium.

## Table of Contents

1. [Connection Issues](#connection-issues)
2. [Training Issues](#training-issues)
3. [Migration Issues](#migration-issues)
4. [Memory Issues](#memory-issues)
5. [Debugging](#debugging)
6. [FAQ](#faq)

---

## Connection Issues

### Cannot connect to Flexium

**Symptoms:**
- "Failed to connect" error
- Process shows as "stale" immediately
- Heartbeat errors in logs

**Solutions:**

1. **Check FLEXIUM_SERVER is set correctly:**
   ```bash
   echo $FLEXIUM_SERVER
   # Should show: app.flexium.ai/yourworkspace
   ```

2. **Check network connectivity:**
   ```bash
   # Test connection to Flexium cloud
   nc -zv app.flexium.ai 443
   ```

3. **Check firewall allows outbound connections:**
   ```bash
   # Ensure outbound HTTPS is allowed
   curl -v https://app.flexium.ai/health
   ```

4. **Verify workspace exists:**
   Log in to [app.flexium.ai](https://app.flexium.ai) and confirm your workspace name is correct.

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

### Process runs in local mode

**Symptoms:**
- "Running in local mode" message
- Process not visible in dashboard

**Solutions:**

1. **Check FLEXIUM_SERVER is set:**
   ```bash
   export FLEXIUM_SERVER="app.flexium.ai/yourworkspace"
   ```

2. **The training will still work:**
   If connection fails, training continues in local mode (graceful degradation). It will reconnect when the connection is restored.

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
- Click "Migrate" in dashboard but nothing happens
- Process stays on same GPU

**Solutions:**

1. **Check process status in dashboard:**
   Process should show "running", not "stale" or "migrating"

2. **Verify heartbeats are working:**
   Flexium sends migration commands via heartbeat response.
   If heartbeats aren't working, migration won't happen.

3. **Check for pending migration:**
   If a migration is already pending, new requests are ignored.
   Wait for the current migration to complete.

4. **Same device check:**
   Migration to the same device is a no-op.

### Migration fails mid-way

**Symptoms:**
- "Migration failed" error
- Process stays on original GPU

**Solutions:**

1. **Check /dev/shm space:**
   ```bash
   df -h /dev/shm
   # Migration uses shared memory
   ```

2. **Check for CUDA errors:**
   ```bash
   nvidia-smi
   # Ensure target GPU is available and has enough memory
   ```

3. **Check driver version:**
   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   # 550+ for pause/resume, 580+ for GPU migration
   ```

---

## Memory Issues

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

View all your processes in the dashboard at [app.flexium.ai](https://app.flexium.ai). Click on any process to see detailed status including device, memory usage, and runtime.

---

## FAQ

### Q: What happens if connection to Flexium is lost?

A: Training continues normally in "local mode". Key behaviors:
- **Running processes**: Continue training without interruption. They automatically reconnect when connection is restored, preserving runtime tracking.
- **Paused processes**: If paused for more than 5 minutes without connection, they auto-resume on the last-used device to prevent indefinite hangs.
- Migration won't work until connection is restored, but training continues unaffected.

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

### Q: Where does the Flexium server run?

A: Flexium is a cloud-hosted service at app.flexium.ai. You don't need to run any servers - just set your workspace:
```bash
export FLEXIUM_SERVER="app.flexium.ai/yourworkspace"
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
