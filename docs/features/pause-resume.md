# Pause/Resume

Flexium provides pause and resume capabilities for long-running training jobs.

**Driver Requirement:** NVIDIA 550+ (pause/resume on same GPU), 580+ (resume on different GPU)

## Overview

Training jobs can be paused at any point and resumed later. With driver 550+, you can resume on the same GPU. With driver 580+, you can resume on any GPU on the same machine.

## How It Works

1. **Pause** - You trigger pause via the dashboard
2. **GPU State Captured** - Flexium captures the complete GPU state at driver level
3. **GPU Freed** - GPU memory is completely released (0 MB residue)
4. **Resume** - You trigger resume via the dashboard
5. **Training Continues** - Training continues from the exact point it was paused

## Dashboard Display

When a process is paused:
- **Status** shows "Paused" with a distinctive badge
- **Memory** displays the last known memory usage before pause (not 0)
- **Runtime** continues to track total training time (not reset)

This allows you to see at a glance how much GPU memory will be needed when resuming.

## Auto-Resume on Disconnect

If connection to Flexium is lost while a process is paused, Flexium will:

1. Attempt to reconnect for up to 5 minutes
2. If reconnection fails, **automatically resume** training on the last-used device
3. Continue running in "local mode"
4. Automatically reconnect when connection is restored

This ensures your paused training jobs don't hang indefinitely.

## Use Cases

- **Resource Sharing** - Temporarily yield GPU for higher-priority jobs
- **Maintenance** - Safely pause for system updates

## Usage

```python
import flexium.auto

with flexium.auto.run():
    # Your training code
    train_model()
    # Pause/resume is triggered via the dashboard
```

Pause and resume are triggered through the web dashboard - there is no manual API for pausing within your code.
