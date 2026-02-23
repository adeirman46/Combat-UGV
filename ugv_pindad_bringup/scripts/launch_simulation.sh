#!/bin/bash
# ============================================================================
# UGV Pindad — Simulation Launch Script
# ============================================================================
# This script activates the micromamba ros2_env, builds the workspace,
# and launches the full Gazebo simulation with the UGV.
#
# Usage:
#   ./launch_simulation.sh           # Launch Gazebo only
#   ./launch_simulation.sh --teleop  # Launch Gazebo + Qt control panel
#   ./launch_simulation.sh --wasd    # Launch Gazebo + terminal WASD
#
# Requirements:
#   - micromamba with ros2_env environment
#   - UGV Pindad workspace at ~/ugv_pindad_real
# ============================================================================

set -e  # Exit on any error

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
WORKSPACE_DIR="$HOME/ugv_pindad_real"
ENV_NAME="ros2_env"
PACKAGE_NAME="ugv_pindad_bringup"

# --------------------------------------------------------------------------
# Color output helpers
# --------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             UGV PINDAD — SIMULATION LAUNCHER                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# --------------------------------------------------------------------------
# Print header
# --------------------------------------------------------------------------
print_header

# --------------------------------------------------------------------------
# Step 1: Activate micromamba environment
# --------------------------------------------------------------------------
print_step "Activating micromamba environment: ${ENV_NAME}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

# Verify ROS2 is available
if ! command -v ros2 &> /dev/null; then
    print_error "ros2 command not found. Is ros2_env properly configured?"
    exit 1
fi
print_step "ROS2 environment active ($(ros2 --version 2>/dev/null || echo 'version unknown'))"

# --------------------------------------------------------------------------
# Step 2: Build the workspace (if needed)
# --------------------------------------------------------------------------
print_step "Building workspace..."
cd "${WORKSPACE_DIR}"

# Only rebuild if source files are newer than the install directory
if [ ! -d "install" ] || [ "$(find ${PACKAGE_NAME} -newer install -name '*.py' -o -name '*.xml' -o -name '*.xacro' -o -name '*.yaml' 2>/dev/null | head -1)" ]; then
    colcon build --packages-select "${PACKAGE_NAME}" 2>&1 | tail -5
    print_step "Build complete."
else
    print_step "Build is up to date, skipping."
fi

# --------------------------------------------------------------------------
# Step 3: Source Gazebo environment (CRITICAL — fixes "Preparing world" hang)
# --------------------------------------------------------------------------
# Gazebo Classic from robostack/conda-forge does NOT auto-set its paths.
# Without these, Gazebo cannot find ground_plane, sun, or any built-in models
# and hangs indefinitely at "Preparing your world...".
# --------------------------------------------------------------------------
GAZEBO_SETUP="$CONDA_PREFIX/share/gazebo/setup.sh"
if [ -f "${GAZEBO_SETUP}" ]; then
    print_step "Sourcing Gazebo environment: ${GAZEBO_SETUP}"
    source "${GAZEBO_SETUP}"
else
    print_warn "Gazebo setup.sh not found at ${GAZEBO_SETUP}"
fi

# Disable online model database to prevent network timeout hangs
export GAZEBO_MODEL_DATABASE_URI=""
print_step "GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"

# --------------------------------------------------------------------------
# Step 4: Source the workspace overlay
# --------------------------------------------------------------------------
print_step "Sourcing workspace overlay..."
source "${WORKSPACE_DIR}/install/setup.bash"

# --------------------------------------------------------------------------
# Step 4: Launch based on mode
# --------------------------------------------------------------------------
MODE="${1:-}"

case "${MODE}" in
    --teleop)
        # Launch Gazebo + Qt Control Panel (in background)
        print_step "Launching Gazebo simulation + Qt Control Panel..."
        echo -e "${YELLOW}The Qt Control Panel will open in a separate window.${NC}"
        echo -e "${YELLOW}Use WASD keys or buttons to control the UGV.${NC}"

        # Launch Gazebo in background
        ros2 launch "${PACKAGE_NAME}" gazebo.launch.py &
        GAZEBO_PID=$!

        # Wait for Gazebo to initialize
        sleep 8

        # Launch teleop
        ros2 launch "${PACKAGE_NAME}" teleop.launch.py &
        TELEOP_PID=$!

        # Wait for both processes
        echo -e "${GREEN}[READY]${NC} Simulation running. Press Ctrl+C to stop."
        trap "kill ${GAZEBO_PID} ${TELEOP_PID} 2>/dev/null; exit 0" INT TERM
        wait
        ;;

    --wasd)
        # Launch Gazebo + WASD terminal teleop
        print_step "Launching Gazebo simulation + WASD Teleop..."
        echo -e "${YELLOW}Use WASD keys in this terminal to control the UGV.${NC}"

        # Launch Gazebo in background
        ros2 launch "${PACKAGE_NAME}" gazebo.launch.py &
        GAZEBO_PID=$!

        # Wait for Gazebo to fully initialize and controllers to activate
        sleep 10

        # Launch WASD teleop in foreground
        ros2 run "${PACKAGE_NAME}" wasd_teleop_node

        # Cleanup
        kill ${GAZEBO_PID} 2>/dev/null
        ;;

    *)
        # Launch Gazebo only
        print_step "Launching Gazebo simulation..."
        echo -e "${YELLOW}To control the UGV, open another terminal and run:${NC}"
        echo -e "  ${CYAN}micromamba activate ${ENV_NAME}${NC}"
        echo -e "  ${CYAN}source ${WORKSPACE_DIR}/install/setup.bash${NC}"
        echo -e "  ${CYAN}ros2 run ${PACKAGE_NAME} qt_control_panel   ${NC}  # Qt GUI"
        echo -e "  ${CYAN}ros2 run ${PACKAGE_NAME} wasd_teleop_node   ${NC}  # Terminal WASD"
        echo ""

        ros2 launch "${PACKAGE_NAME}" gazebo.launch.py
        ;;
esac
