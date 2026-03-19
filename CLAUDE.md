# Flexium Client - Claude Code Context

This is the **client library** repository. For the main project context, principles, and session tracking, see:

**[flexium-server/CLAUDE.md](https://github.com/flexiumai/flexium-server/blob/main/CLAUDE.md)**

## Quick Reference

### Project Principles
1. **Test Every Bug Fix** - Every bug fix must have a corresponding test
2. **SOLID Principles** - All code must comply with SOLID principles

### Repository Structure
- `flexium/` - Main library code (auto-patches PyTorch)
- `docs/` - Public documentation (mkdocs)
- `examples/` - Usage examples
- `tests/` - Test suite

### Common Commands
```bash
# Run tests
pytest tests/ -v

# Build docs
mkdocs serve

# Install from GitHub (latest)
pip install git+https://github.com/flexiumai/flexium.git
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `FLEXIUM_SERVER` | Server URL: `app.flexium.ai/workspace` |
| `GPU_DEVICE` | Initial GPU device (default: `cuda:0`) |
| `FLEXIUM_DEBUG` | Enable debug logging |

### Connection Example
```python
import os
os.environ["FLEXIUM_SERVER"] = "app.flexium.ai/myworkspace"

import flexium.auto
with flexium.auto.run():
    # your training code
```

## Session Logs

Session logs are maintained in the flexium-server repository:
- `flexium-server/sessions/flexium/` - Client-related sessions
- `flexium-server/sessions/flexium-server/` - Server-related sessions

To save a session, use `/save-session` - the skill is defined at:
`flexium-server/.claude/skills/save-session.md`
