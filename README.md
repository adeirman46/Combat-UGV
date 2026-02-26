# Combat-UGV

This repository contains the simulation environment and control software for the Combat UGV (Unmanned Ground Vehicle). The simulation is built on ROS 2 and Gazebo, utilizing a designated `micromamba` environment (`ros2_env`) for dependency management.

## Installation

### 1. Install Micromamba
If you don't have micromamba installed, you can install it using the official curl script:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
Follow the on-screen prompts to initialize it for your shell and restart your terminal.

### 2. Setup the ROS 2 Environment (`ros2_env`)
We use a dedicated micromamba environment named `ros2_env` containing all required ROS 2 packages and dependencies.

You can create and activate the environment using the provided `environment.yml`:
```bash
micromamba env create -n ros2_env -f environment.yml
```

Once created, activate the environment:
```bash
micromamba activate ros2_env
```

### 3. Build the Workspace
Before running the simulations, build the ROS 2 packages using `colcon`:
```bash
cd ~/Combat-UGV  # or your respective workspace directory
colcon build --packages-select ugv_pindad_bringup
source install/setup.bash
```

## Usage

Convenience launch scripts are located in the `ugv_pindad_bringup/scripts/` directory. Ensure your `ros2_env` is active (`micromamba activate ros2_env`) and workspace is sourced before running them.

### Forest Simulation
To launch the UGV in a rich forest environment:
```bash
cd ~/Combat-UGV/ugv_pindad_bringup/scripts
./launch_forest_simulation.sh
```

### Basic Simulation with WASD Teleoperation
To launch the standard simulation and control the UGV using your keyboard (WASD keys) in the terminal:
```bash
cd ~/Combat-UGV/ugv_pindad_bringup/scripts
./launch_simulation.sh --wasd
```
> **Note:** You can also run `./launch_simulation.sh --teleop` to launch a Qt-based control GUI, or just `./launch_simulation.sh` to launch only the Gazebo world.

### LiDAR Simulation
To launch the simulation dedicated to testing LiDAR sensors, perception, and mapping:
```bash
cd ~/Combat-UGV/ugv_pindad_bringup/scripts
./launch_lidar_simulation.sh
```
