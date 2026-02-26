#!/bin/bash

# ===================================================================================
# LAUNCH SCRIPT: UGV Pindad Simulation (Forest Environment)
# ===================================================================================

# 1. Source the ROS 2 Environment
# Use micromamba shell hook for bash (works even if user's login shell is zsh)
eval "$(micromamba shell hook --shell bash)"
micromamba activate ros2_env

# 2. Source the Local Workspace
if [ -f "$HOME/ugv_pindad_real/install/setup.bash" ]; then
  source "$HOME/ugv_pindad_real/install/setup.bash"
fi

# 3. Set Gazebo Resource Paths
# Add the 'maps' and 'models' directories to the path so Gazebo can find assets
export IGN_GAZEBO_RESOURCE_PATH="$HOME/ugv_pindad_real/ugv_pindad_bringup/models:$HOME/ugv_pindad_real/ugv_pindad_bringup:$HOME/ugv_pindad_real/ugv_pindad_bringup/maps:$HOME/ugv_pindad_real/ugv_pindad_bringup/meshes:$IGN_GAZEBO_RESOURCE_PATH"

echo "========================================================"
echo "LAUNCHING UGV SIMULATION IN FOREST ENVIRONMENT"
echo "========================================================"
echo "SDF World: forest.sdf"
echo "Heightmap: heightmap.png"
echo "========================================================"

# 4. Launch Simulation
# Execute the python launch file directly from source to avoid needing to rebuild/install again
LAUNCH_FILE="$HOME/ugv_pindad_real/ugv_pindad_bringup/launch/forest_simulation.launch.py"
ros2 launch "$LAUNCH_FILE"
