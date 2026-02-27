#!/bin/bash
# ============================================================
# SNN Event Tracker — Single View (Orange Heatmap)
# Shows only the SNN spike activation with tracking overlay
# ============================================================

set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# Install spikingjelly if needed
python3 -c "import spikingjelly" 2>/dev/null || {
    echo "[Setup] Installing spikingjelly..."
    pip install spikingjelly
}

echo "=============================================="
echo "  SNN Event Tracker — Single View"
echo "  Orange SNN Spike Activation + Tracking"
echo "  Press 'q' to quit"
echo "=============================================="

python3 /home/irman/Documents/ugv_eventcam/snn_tracker.py
