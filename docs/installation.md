# Installation Guide

This guide covers all installation methods and requirements for Flexium.AI.

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with CUDA support | NVIDIA A100/H100 or consumer RTX 30xx/40xx |
| RAM | 8 GB | 16+ GB |
| Storage | 1 GB for flexium | SSD with 10+ GB free for checkpoints |

### Software

| Requirement | Version | Notes |
|-------------|---------|-------|
| Operating System | Linux x86_64 | Ubuntu 20.04+, RHEL 8+, Debian 10+ |
| Python | 3.8 - 3.12 | 3.10+ recommended |
| NVIDIA Driver | **580+** | **Required** for zero-residue migration |
| CUDA | 11.0+ | Must match PyTorch CUDA version |
| PyTorch | 2.0+ | With CUDA support |

!!! warning "Driver 580+ Required"
    Zero-residue migration requires NVIDIA driver version 580 or higher. Earlier drivers do not support the necessary migration features.

### Verify Driver Version

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Expected output: 580.xx or higher
```

If your driver is older than 580, you'll need to update:

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

Flexium requires PyTorch with CUDA support. Install PyTorch **before** installing flexium.

### For CUDA 11.8

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### For Latest PyTorch

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the install command for your system.

### Verify PyTorch CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
```

---

## Dependencies

### Core Dependencies (Auto-installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `grpcio` | >=1.50.0 | gRPC communication |
| `protobuf` | >=4.0.0 | Protocol buffers |
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
| `grpcio-tools` | Protobuf compilation |

---

## Environment Setup

### Option 1: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv flexium-env
source flexium-env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install flexium
pip install flexium
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n flexium python=3.10
conda activate flexium

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

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
# Orchestrator address
orchestrator: localhost:50051

# Default device
device: cuda:0

# Checkpoint directory
checkpoint_dir: /tmp/flexium/checkpoints

# Heartbeat interval (seconds)
heartbeat_interval: 3.0

# Resource requirements
min_gpus: 1
max_gpus: 1
priority: 50
preemptible: true
migratable: true
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
export FLEXIUM_SERVER="localhost:50051/myworkspace"
export GPU_DEVICE=cuda:0
export FLEXIUM_LOG_LEVEL=DEBUG
```

!!! note "URL Format"
    The `FLEXIUM_SERVER` variable uses a token-in-path format: `host:port/workspace`. This routes your training jobs to the correct workspace orchestrator.

### Project-Local Config

Create `.flexiumrc` in your project directory (takes precedence over `~/.flexiumrc`):

```yaml
orchestrator: localhost:50051
device: cuda:0
```

---

## Verification

### Step 1: Check Installation

```bash
# Verify flexium is installed
python -c "import flexium; print(f'Flexium version: {flexium.__version__}')"

# Verify CLI is available
flexium-ctl --help
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
export FLEXIUM_SERVER="localhost:50051/myworkspace"
```

### Step 4: Test Training Integration

```bash
# Create test script
cat > test_flexium.py << 'EOF'
import flexium.auto
import torch

with flexium.auto.run():
    x = torch.zeros(100, 100).cuda()
    print(f"Tensor on: {x.device}")
    print("Flexium integration working!")
EOF

# Run test
FLEXIUM_SERVER="localhost:50051/myworkspace" python test_flexium.py
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
1. Install NVIDIA driver: `sudo apt install nvidia-driver-580`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "Module 'flexium' not found"

```bash
# Check installation
pip show flexium

# Reinstall
pip install --force-reinstall flexium
```

### "grpcio installation fails"

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

## Docker Installation

### Using Pre-built Image

```bash
# Pull image (when available)
docker pull flexium/orchestrator:latest

# Run orchestrator
docker run -d -p 50051:50051 -p 8080:8080 flexium/orchestrator
```

### Building from Source

```bash
# From flexium directory
docker build -t flexium:local .

# Run
docker run --gpus all -p 50051:50051 -p 8080:8080 flexium:local
```

---

## Next Steps

- [Getting Started](getting-started.md) - Quick start guide
- [Architecture](ARCHITECTURE.md) - How flexium works
- [API Reference](api.md) - Complete API docs
- [Troubleshooting](troubleshooting.md) - Common issues
