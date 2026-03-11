#!/bin/bash
#
# Flexium Quick Demo
#
# This script starts the orchestrator with dashboard and runs
# MNIST training - a complete demo in one command.
#
# Usage: ./examples/utils/run_demo.sh
#
# What happens:
#   1. Starts orchestrator server on port 50051
#   2. Starts web dashboard on http://localhost:8080
#   3. Runs MNIST training example
#   4. You can open the dashboard and click "Migrate" to move training between GPUs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  Flexium.AI - Quick Demo"
echo "=============================================="
echo ""

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
echo "Click 'Migrate' to move training between GPUs"
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
