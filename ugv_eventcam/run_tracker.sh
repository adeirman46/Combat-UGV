#!/bin/bash
# ============================================================
# SNN Event Camera Tracker v2 — Launcher
# Framework: spikingjelly (github.com/fangwei123456/spikingjelly)
# Pre-trained: CIFAR10 ResNet18 SNN backbone
# ============================================================

set -e

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# Install spikingjelly if not already installed
python3 -c "import spikingjelly" 2>/dev/null || {
    echo "[Setup] Installing spikingjelly..."
    pip install spikingjelly
}

echo "=============================================="
echo "  SNN Event Camera Tracker v2"
echo "  Framework: spikingjelly (PLIF neurons)"
echo "  Backbone : Pre-trained CIFAR10 ResNet18 SNN"
echo "  Press 'q' to quit"
echo "=============================================="

# Run the SNN event tracker
python3 /home/irman/Documents/ugv_eventcam/event_tracker.py
