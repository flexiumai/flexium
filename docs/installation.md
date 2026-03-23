# Installation Guide

This guide covers all installation methods and requirements for Flexium.AI.

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with CUDA support | NVIDIA A100/H100 or consumer RTX 30xx/40xx |
| RAM | 8 GB | 16+ GB |
| Storage | 1 GB for flexium | SSD recommended |

### Software

| Requirement | Version | Notes |
|-------------|---------|-------|
| Operating System | Linux x86_64 | Ubuntu 20.04+, RHEL 8+, Debian 10+ |
| Python | 3.8 - 3.12 | 3.10+ recommended |
| NVIDIA Driver | **550+** / **580+** | 550+ for pause/resume, 580+ for GPU migration |
| CUDA | 12.4+ | Required for driver 550+ |
| PyTorch | 2.0+ | With CUDA support |

!!! info "Driver Requirements"
    - **Driver 550+**: Pause/resume training on the same GPU
    - **Driver 580+**: Migrate training to a different GPU with zero VRAM residue

### Verify Driver Version

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# 550+ for pause/resume, 580+ for GPU migration
```

If your driver is older than 550, you'll need to update:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-580

# Or download from NVIDIA website:
# https://www.nvidia.com/Download/index.aspx
```

---

## Installation Methods

### Method 1: From PyPI (Recommended)

```bash
pip install flexium
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/flexiumai/flexium.git
cd flexium

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[all]"
```

### Method 3: From GitHub Release

```bash
pip install https://github.com/flexiumai/flexium/releases/download/v0.1.1/flexium-0.1.1-py3-none-any.whl
```

---

## PyTorch Installation

Flexium requires PyTorch with CUDA 12.4+ support. Install PyTorch **before** installing flexium.

### For CUDA 12.4+

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### For Latest PyTorch

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the install command for your system. Make sure to select CUDA 12.4 or higher.

### Verify PyTorch CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.x.x+cu124
CUDA available: True
CUDA version: 12.4
```

---

## Dependencies

### Core Dependencies (Auto-installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `python-socketio[client]` | >=5.0.0 | WebSocket communication |
| `pynvml` | >=11.0.0 | GPU monitoring |
| `flask` | >=2.0.0 | Web dashboard |

### Development Dependencies

```bash
pip install flexium[dev]
```

| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `pytest-cov` | Coverage |
| `mypy` | Type checking |
| `ruff` | Linting |

---

## Environment Setup

### Option 1: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv flexium-env
source flexium-env/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install flexium
pip install flexium
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n flexium python=3.10
conda activate flexium

# Install PyTorch with CUDA 12.4
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Install flexium
pip install flexium
```

### Option 3: System-wide Installation

```bash
# Not recommended, but possible
sudo pip install flexium
```

---

## Configuration

### Config File (Recommended)

Create `~/.flexiumrc`:

```yaml
# Server address with workspace
server: app.flexium.ai/myworkspace

# Default device
device: cuda:0

# Heartbeat interval (seconds)
heartbeat_interval: 3.0
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLEXIUM_SERVER` | Server address with workspace (host:port/workspace) | None (local mode) |
| `GPU_DEVICE` | Default GPU device | `cuda:0` |
| `FLEXIUM_LOG_LEVEL` | Log level | `INFO` |
| `FLEXIUM_DEBUG` | Enable debug mode | `false` |

Example:

```bash
# Format: host:port/workspace
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"
export GPU_DEVICE=cuda:0
export FLEXIUM_LOG_LEVEL=DEBUG
```

!!! note "URL Format"
    The `FLEXIUM_SERVER` variable uses a token-in-path format: `host:port/workspace`. This routes your training jobs to the correct workspace orchestrator.

### Project-Local Config

Create `.flexiumrc` in your project directory (takes precedence over `~/.flexiumrc`):

```yaml
server: app.flexium.ai/myworkspace
device: cuda:0
```

---

## Verification

### Step 1: Check Installation

```bash
# Verify flexium is installed
python -c "import flexium; print(f'Flexium version: {flexium.__version__}')"

# Verify module loads
python -c "import flexium.auto; print('OK')"
```

### Step 2: Check GPU Access

```bash
python -c "
import torch
import pynvml
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
print(f'GPUs detected: {device_count}')
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f'  GPU {i}: {name}')
pynvml.nvmlShutdown()
"
```

### Step 3: Test Server Connection

```bash
# Set your server and workspace
export FLEXIUM_SERVER="app.flexium.ai/myworkspace"
```

### Step 4: Test Training Integration

```bash
# Create test script
cat > test_flexium.py << 'EOF'
import flexium
flexium.init()

import torch
x = torch.zeros(100, 100).cuda()
print(f"Tensor on: {x.device}")
print("Flexium integration working!")
EOF

# Run test
FLEXIUM_SERVER="app.flexium.ai/myworkspace" python test_flexium.py
```

---

## Troubleshooting Installation

### "CUDA not available"

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**
1. Install NVIDIA driver: `sudo apt install nvidia-driver-550` (or `nvidia-driver-580` for GPU migration)
2. Reinstall PyTorch with CUDA 12.4: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### "Module 'flexium' not found"

```bash
# Check installation
pip show flexium

# Reinstall
pip install --force-reinstall flexium
```

### "python-socketio installation fails"

```bash
# Install build dependencies
sudo apt install build-essential python3-dev

# Then install flexium
pip install flexium
```

### "pynvml fails to initialize"

This usually means the NVIDIA driver is not loaded:

```bash
# Check if driver is loaded
lsmod | grep nvidia

# Load driver if needed
sudo modprobe nvidia
```

### "Permission denied" errors

```bash
# Add user to video group
sudo usermod -aG video $USER

# Log out and back in, or use newgrp
newgrp video
```

---

## Next Steps

- [Getting Started](getting-started.md) - Quick start guide
- [Architecture](ARCHITECTURE.md) - How flexium works
- [API Reference](api.md) - Complete API docs
- [Troubleshooting](troubleshooting.md) - Common issues
