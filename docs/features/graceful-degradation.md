# Graceful Degradation

Flexium supports graceful degradation - your training continues even if connection to Flexium is lost.

## Overview

Flexium is not a single point of failure. If the connection to Flexium cloud is lost, your training continues normally in "local mode". When connection is restored, Flexium automatically reconnects.

## How It Works

1. **Connection Lost** - Network issue or Flexium maintenance
2. **Local Mode** - Training continues on current GPU without interruption
3. **Reconnection** - Flexium automatically attempts to reconnect
4. **Restored** - When connection is back, full functionality resumes

## Behavior

| Scenario | Training | Migration | Dashboard |
|----------|----------|-----------|-----------|
| Connected | ✅ Runs | ✅ Available | ✅ Visible |
| Disconnected | ✅ Runs | ❌ Unavailable | ❌ Not visible |
| Reconnected | ✅ Runs | ✅ Available | ✅ Visible |

## What You'll See

When connection is lost:
```
[flexium] Lost connection, attempting reconnect...
[flexium] Running in local mode
```

When connection is restored:
```
[flexium] Reconnected!
```

## Key Points

- **Training never stops** due to Flexium connection issues
- **No data loss** - your training progress is unaffected
- **Automatic reconnection** - no manual intervention needed
- **Paused processes auto-resume** after 5 minutes if connection isn't restored
