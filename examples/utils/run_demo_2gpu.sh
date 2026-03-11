#!/bin/bash
#
# Flexium Quick Demo (2 GPU Mode)
#
# Same as run_demo.sh but restricts to only 2 GPUs using CUDA_VISIBLE_DEVICES.
# This is useful for:
#   - Testing migration with limited GPUs
#   - Not interfering with other users on shared machines
#   - Simulating a 2-GPU setup
#
# Usage: ./examples/utils/run_demo_2gpu.sh
#        ./examples/utils/run_demo_2gpu.sh 0,1    # Use GPUs 0 and 1
#        ./examples/utils/run_demo_2gpu.sh 2,3    # Use GPUs 2 and 3
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Default to GPUs 0 and 1, or use provided argument
GPUS="${1:-0,1}"

echo "=============================================="
echo "  Flexium.AI - Quick Demo (2 GPU Mode)"
echo "=============================================="
echo ""
echo "  Using GPUs: $GPUS"
echo "  (These will appear as cuda:0 and cuda:1)"
echo ""

# Set CUDA_VISIBLE_DEVICES for this script and all child processes
export CUDA_VISIBLE_DEVICES="$GPUS"

# Check for required commands
if ! command -v flexium-ctl &> /dev/null; then
    echo "Error: flexium-ctl not found. Install flexium first:"
    echo "  pip install -e ."
    exit 1
fi

# Check if orchestrator is already running
if pgrep -f "flexium-ctl server" > /dev/null 2>&1; then
    echo "Orchestrator already running. Using existing server."
    ORCHESTRATOR_RUNNING=true
else
    ORCHESTRATOR_RUNNING=false
    echo "Starting orchestrator with dashboard..."
    flexium-ctl server 50051 --dashboard &
    SERVER_PID=$!
    sleep 2
    echo "  Server PID: $SERVER_PID"
    echo "  gRPC: localhost:50051"
    echo "  Dashboard: http://localhost:8080"
fi

echo ""
echo "=============================================="
echo "  Starting MNIST Training"
echo "=============================================="
echo ""
echo "Open the dashboard at http://localhost:8080"
echo "You will see 2 GPUs: cuda:0 and cuda:1"
echo "Click 'Migrate' to move training between them"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "----------------------------------------------"
echo ""

# Run the training example
python examples/simple/mnist_train_auto.py --epochs 100

# Cleanup
if [ "$ORCHESTRATOR_RUNNING" = false ] && [ -n "$SERVER_PID" ]; then
    echo ""
    echo "Stopping orchestrator..."
    kill $SERVER_PID 2>/dev/null || true
fi

echo "Demo complete!"
