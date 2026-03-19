# Works Offline

Your training keeps running even if the connection to Flexium is lost.

## Overview

If the connection to Flexium is lost (network issues, server maintenance, etc.), your training continues normally. When connection is restored, Flexium automatically reconnects - no action needed from you.

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
