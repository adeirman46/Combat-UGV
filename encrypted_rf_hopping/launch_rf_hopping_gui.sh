#!/bin/bash
# ============================================================
# launch_rf_hopping_gui.sh
# Launches the encrypted RF frequency hopping pub+sub WITH GUI
# Full GNU Radio Qt waterfall, frequency sink, jammer controls
# Uses micromamba ros2_env
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${SCRIPT_DIR}/encrypted_rf_hopping"

echo "============================================"
echo " Encrypted RF Frequency Hopping (GUI Mode)"
echo "============================================"

# Activate micromamba ros2_env
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# Build the package
echo "[INFO] Building encrypted_rf_hopping package..."
cd "${SCRIPT_DIR}"
colcon build --packages-select encrypted_rf_hopping
source install/setup.bash

# Launch both pub_gui and sub_gui
echo "[INFO] Launching GUI publisher + subscriber..."
ros2 launch encrypted_rf_hopping rf_hopping_gui.launch.py
