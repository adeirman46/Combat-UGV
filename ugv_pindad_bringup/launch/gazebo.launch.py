# ============================================================================
# UGV Pindad — Gazebo Simulation Launch File
# ============================================================================
# This launch file brings up the complete simulation:
#   1. Starts Gazebo Classic with an empty world
#   2. Processes the xacro URDF into a robot description
#   3. Spawns the UGV entity in Gazebo
#   4. Starts the robot_state_publisher for TF broadcasting
#   5. Activates the joint_state_broadcaster and diff_drive_controller
#
# Usage:
#   ros2 launch ugv_pindad_bringup gazebo.launch.py
# ============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# ==========================================================================
# Helper: detect Gazebo paths from conda/micromamba environment
# ==========================================================================
def _get_gazebo_env_vars():
    """
    Detect Gazebo Classic paths from the conda/micromamba environment.

    Returns a dict of environment variable names → values that must be set
    for Gazebo to find its models, plugins, and resources.
    Without these, Gazebo hangs at 'Preparing your world...' forever.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    gazebo_share = os.path.join(conda_prefix, "share", "gazebo-11")

    env_vars = {}

    if os.path.isdir(gazebo_share):
        # Model path — includes built-in models (ground_plane, sun, etc.)
        models_dir = os.path.join(gazebo_share, "models")
        existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
        if models_dir not in existing_model_path:
            env_vars["GAZEBO_MODEL_PATH"] = (
                f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
            )

        # Resource path — media, materials, worlds
        existing_resource = os.environ.get("GAZEBO_RESOURCE_PATH", "")
        if gazebo_share not in existing_resource:
            env_vars["GAZEBO_RESOURCE_PATH"] = (
                f"{gazebo_share}:{existing_resource}" if existing_resource else gazebo_share
            )

        # Plugin path — Gazebo physics/sensor/rendering plugins
        plugins_dir = os.path.join(conda_prefix, "lib", "gazebo-11", "plugins")
        existing_plugins = os.environ.get("GAZEBO_PLUGIN_PATH", "")
        if plugins_dir not in existing_plugins:
            env_vars["GAZEBO_PLUGIN_PATH"] = (
                f"{plugins_dir}:{existing_plugins}" if existing_plugins else plugins_dir
            )

        # OGRE resource path — rendering backend
        ogre_dir = os.path.join(conda_prefix, "lib", "OGRE")
        if os.path.isdir(ogre_dir):
            env_vars["OGRE_RESOURCE_PATH"] = ogre_dir

    # Disable online model database — prevents network timeout hangs
    env_vars["GAZEBO_MODEL_DATABASE_URI"] = ""

    return env_vars


def generate_launch_description():
    """Generate the complete Gazebo simulation launch description."""

    # ------------------------------------------------------------------
    # Package paths
    # ------------------------------------------------------------------
    pkg_share = get_package_share_directory("ugv_pindad_bringup")
    gazebo_ros_share = get_package_share_directory("gazebo_ros")

    # Path to the URDF xacro file
    xacro_file = os.path.join(pkg_share, "urdf", "ugv_pindad.urdf.xacro")

    # Path to controller configuration
    controllers_config = os.path.join(pkg_share, "config", "controllers.yaml")

    # ------------------------------------------------------------------
    # Gazebo environment variables (CRITICAL for robostack/conda installs)
    # ------------------------------------------------------------------
    gazebo_env_actions = []
    for var_name, var_value in _get_gazebo_env_vars().items():
        gazebo_env_actions.append(SetEnvironmentVariable(name=var_name, value=var_value))

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock instead of wall clock",
    )

    # ------------------------------------------------------------------
    # Process xacro → URDF string
    # ------------------------------------------------------------------
    robot_description_content = Command(["xacro ", xacro_file])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ------------------------------------------------------------------
    # Gazebo — start the simulator with an empty world
    # ------------------------------------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "verbose": "true",
            "pause": "false",
        }.items(),
    )

    # ------------------------------------------------------------------
    # Robot State Publisher — broadcasts TF from joint states + URDF
    # ------------------------------------------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # ------------------------------------------------------------------
    # Spawn Entity — insert UGV model into Gazebo
    # ------------------------------------------------------------------
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_ugv_pindad",
        output="screen",
        arguments=[
            "-topic", "robot_description",   # read URDF from this topic
            "-entity", "ugv_pindad",          # entity name in Gazebo
            "-x", "0.0",                      # spawn position X
            "-y", "0.0",                      # spawn position Y
            "-z", "0.5",                      # spawn position Z (above ground)
        ],
    )

    # ------------------------------------------------------------------
    # Assemble the launch description
    # ------------------------------------------------------------------
    # NOTE: No controller spawners needed — the diff_drive and
    # joint_state_publisher plugins are loaded directly by Gazebo
    # from the URDF <gazebo> tags. They activate automatically.
    # ------------------------------------------------------------------
    return LaunchDescription(
        gazebo_env_actions + [
        use_sim_time_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
