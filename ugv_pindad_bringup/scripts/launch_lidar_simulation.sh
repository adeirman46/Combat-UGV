#!/bin/bash
set -e

# ===================================================================================
# UGV PINDAD SIMULATION LAUNCH SCRIPT
# ===================================================================================

# 1. Setup Workspace
WS_DIR="/home/irman/ugv_pindad_real"
cd $WS_DIR

# 2. Activate Micromamba Environment
echo "[INFO] Activating ROS 2 Environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# 3. Build Package (Optional - ensures latest changes)
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
echo "[INFO] Launching (UGV + LiDAR + 4 ZED Cameras + Qt Viz)..."
ros2 launch ugv_pindad_bringup lidar_simulation.launch.py
