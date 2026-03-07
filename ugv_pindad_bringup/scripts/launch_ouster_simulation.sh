#!/bin/bash
set -e

# ===================================================================================
# UGV PINDAD SIMULATION LAUNCH SCRIPT — OUSTER OS1-128 LiDAR
# ===================================================================================
# Replaces the Velodyne VLP-16 (16ch, 28.8K pts) with the Ouster OS1-128
# (128ch, 262K pts) for dramatically improved point cloud density.
# ===================================================================================

# 1. Setup Workspace
WS_DIR="/home/irman/ugv_pindad_real"
cd $WS_DIR

# 2. Activate Micromamba Environment
echo "[INFO] Activating ROS 2 Environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# 3. Sync source files from Combat-UGV to workspace before building
echo "[INFO] Syncing files from Combat-UGV source to workspace..."
cp "$HOME/Combat-UGV/ugv_pindad_bringup/urdf/ugv_pindad_ouster.urdf.xacro" "$HOME/ugv_pindad_real/ugv_pindad_bringup/urdf/"
cp "$HOME/Combat-UGV/ugv_pindad_bringup/launch/ouster_lidar_simulation.launch.py" "$HOME/ugv_pindad_real/ugv_pindad_bringup/launch/"
cp "$HOME/Combat-UGV/ugv_pindad_bringup/config/ouster_lidar_view.rviz" "$HOME/ugv_pindad_real/ugv_pindad_bringup/config/"
cp "$HOME/Combat-UGV/ugv_pindad_bringup/scripts/"*.py "$HOME/ugv_pindad_real/ugv_pindad_bringup/scripts/" 2>/dev/null || true

# 4. Build Package (Ensures latest changes are included)
echo "[INFO] Cleaning previous build..."
rm -rf build/ugv_pindad_bringup install/ugv_pindad_bringup
echo "[INFO] Building ugv_pindad_bringup..."
colcon build --packages-select ugv_pindad_bringup

# 4. Source Environment
echo "[INFO] Sourcing setup.bash..."
source $WS_DIR/install/setup.bash

# 5. Export Gazebo Resource Path (CRITICAL for meshes)
# This allows Gz to find "file://$(find ugv_pindad_bringup)/meshes/..."
export IGN_GAZEBO_RESOURCE_PATH=$(ros2 pkg prefix ugv_pindad_bringup)/share/ugv_pindad_bringup:${IGN_GAZEBO_RESOURCE_PATH}
export QML_IMPORT_PATH=$CONDA_PREFIX/lib/ign-gazebo-6/plugins/gui:$QML_IMPORT_PATH
export QML2_IMPORT_PATH=$CONDA_PREFIX/lib/ign-gazebo-6/plugins/gui:$QML2_IMPORT_PATH
export IGN_GUI_PLUGIN_PATH=$CONDA_PREFIX/lib/ign-gazebo-6/plugins/gui:$IGN_GUI_PLUGIN_PATH

echo "[DEBUG] CONDA_PREFIX: $CONDA_PREFIX"
echo "[DEBUG] QML_IMPORT_PATH: $QML_IMPORT_PATH"
echo "[INFO] IGN_GAZEBO_RESOURCE_PATH set."

# 6. Ensure Scripts are Executable
chmod +x $WS_DIR/ugv_pindad_bringup/scripts/camera_viz.py

# 7. Launch Simulation
echo "=========================================================="
echo " LAUNCHING UGV SIMULATION — OUSTER OS1-128 LiDAR"
echo "=========================================================="
echo " LiDAR:     Ouster OS1-128 (128ch, 262K pts/scan)"
echo " Channels:  128 vertical × 2048 horizontal"
echo " VFOV:      ±22.5° (45° total)"
echo " Range:     0.8m — 120m"
echo " Noise:     ±3mm (gaussian)"
echo " World:     obstacle.world (50×50m)"
echo "=========================================================="

# Execute the python launch file directly from source to avoid needing to rebuild/install again
LAUNCH_FILE="$HOME/ugv_pindad_real/ugv_pindad_bringup/launch/ouster_lidar_simulation.launch.py"

# The launch file should now definitely exist since we synced it above.

ros2 launch "$LAUNCH_FILE"
