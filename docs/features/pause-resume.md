# Pause/Resume

Flexium provides robust pause and resume capabilities for long-running training jobs.

## Overview

Training jobs can be paused at any point and resumed later, whether on the same GPU, a different GPU, or even a different machine.

## How It Works

1. **Pause Signal** - The orchestrator sends a pause signal to the training process
2. **Checkpoint Creation** - Flexium captures the complete training state
3. **Resource Release** - GPU memory and resources are freed
4. **Resume Signal** - When ready, the orchestrator signals the process to resume
5. **State Restoration** - Training continues from the exact point it was paused

## Dashboard Display

When a process is paused:
- **Status** shows "Paused" with a distinctive badge
- **Memory** displays the last known memory usage before pause (not 0)
- **Runtime** continues to track total training time (not reset)

This allows you to see at a glance how much GPU memory will be needed when resuming.

## Auto-Resume on Orchestrator Disconnect

If the orchestrator becomes unavailable while a process is paused (e.g., server restart), Flexium will:

1. Attempt to reconnect for up to 5 minutes
2. If reconnection fails, **automatically resume** training on the last-used device
3. Continue running in "local mode" without orchestrator control
4. Automatically reconnect to the orchestrator when it becomes available again

This ensures your paused training jobs don't hang indefinitely if the orchestrator restarts.

## Use Cases

- **Resource Sharing** - Temporarily yield GPU for higher-priority jobs
- **Cost Optimization** - Pause during expensive compute windows
- **Maintenance** - Safely pause for system updates
- **Debugging** - Pause to inspect training state

## API

```python
import flexium

# Automatic pause/resume handling
flexium.auto.patch()

# Manual control (advanced)
from flexium import FlexiumDriver

driver = FlexiumDriver()
driver.pause()   # Pause training
driver.resume()  # Resume training
```

## Configuration

Configure pause behavior in your Flexium config:

```python
from flexium.config import FlexiumConfig

config = FlexiumConfig(
    pause_timeout=300,  # Max seconds to wait for clean pause
    checkpoint_on_pause=True,  # Save checkpoint when pausing
)
```
