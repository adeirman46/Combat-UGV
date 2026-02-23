# ============================================================================
# UGV Pindad — Teleop Launch File
# ============================================================================
# Launches the teleoperation nodes for controlling the UGV:
#   1. WASD keyboard teleop node (terminal-based)
#   2. Qt GUI control panel (graphical)
#
# Both nodes publish to the same cmd_vel topic, so they can be used
# interchangeably or simultaneously (last command wins).
#
# Usage:
#   ros2 launch ugv_pindad_bringup teleop.launch.py
# ============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate teleop launch description with WASD + Qt GUI nodes."""

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock instead of wall clock",
    )

    # ------------------------------------------------------------------
    # Qt Control Panel — graphical control interface
    # ------------------------------------------------------------------
    # Launches the PyQt5 GUI with directional buttons, speed sliders,
    # and keyboard capture (WASD keys work when the window has focus).
    # ------------------------------------------------------------------
    qt_control_panel = Node(
        package="ugv_pindad_bringup",
        executable="qt_control_panel",
        name="qt_control_panel",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # ------------------------------------------------------------------
    # Assemble launch description
    # ------------------------------------------------------------------
    return LaunchDescription([
        use_sim_time_arg,
        qt_control_panel,
    ])
