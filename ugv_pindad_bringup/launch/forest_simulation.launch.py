#!/usr/bin/env python3

# ===================================================================================
# LAUNCH FILE: LiDAR Simulation (Gz Fortress)
# ===================================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # -----------------------------------------------------------------------
    # 1. PATHS
    # -----------------------------------------------------------------------
    pkg_ugv_bringup = get_package_share_directory('ugv_pindad_bringup')
    pkg_ros_gz_sim  = get_package_share_directory('ros_gz_sim')

    xacro_file       = os.path.join(pkg_ugv_bringup, 'urdf', 'ugv_pindad_lidar.urdf.xacro')
    # Force use of absolute source path for baylands to prevent install/share sync issues
    world_file       = '/home/irman/ugv_pindad_real/ugv_pindad_bringup/worlds/baylands.sdf'
    rviz_config_file = os.path.join(pkg_ugv_bringup, 'config', 'lidar_view.rviz')

    # -----------------------------------------------------------------------
    # 2. PROCESS URDF
    # -----------------------------------------------------------------------
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    robot_description_content = doc.toxml()

    # -----------------------------------------------------------------------
    # 3. ENVIRONMENT
    # -----------------------------------------------------------------------
    gz_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=os.path.join(pkg_ugv_bringup, 'meshes') +
              os.pathsep + pkg_ugv_bringup +
              os.pathsep + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    )

    # -----------------------------------------------------------------------
    # 4. GZ SIM
    # -----------------------------------------------------------------------
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_file}',
        }.items()
    )

    # -----------------------------------------------------------------------
    # 5. ROBOT STATE PUBLISHER
    # -----------------------------------------------------------------------
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True,
        }]
    )

    # -----------------------------------------------------------------------
    # 6. SPAWN ENTITY (Delayed to allow massive Baylands world to load first)
    # -----------------------------------------------------------------------
    spawn_entity = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-topic', 'robot_description',
                    '-name', 'ugv_pindad',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '1.5', # Higher Z to drop onto terrain
                ],
                output='screen'
            )
        ]
    )

    # -----------------------------------------------------------------------
    # 7. ROS ↔ GZ BRIDGE
    # -----------------------------------------------------------------------
    # IMPORTANT: gpu_lidar in Gz publishes PointCloud on <topic>/points
    # (i.e., /lidar/points/points), NOT on <topic> directly.
    # -----------------------------------------------------------------------
    # 7. ROS ↔ GZ BRIDGE
    # -----------------------------------------------------------------------
    # Dynamic Bridge Arguments
    bridge_args = [
        # cmd_vel: bidirectional
        '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
        # odom: Gz -> ROS
        '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        # tf: Gz -> ROS
        '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        # joint_states: Gz -> ROS
        '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
        # lidar: Gz -> ROS (Note: Gz suffix /points/points is managed via remapping or here)
        '/lidar/points/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
        # clock
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
    ]

    # Add Camera Topics (Front, Rear, Left, Right)
    cameras = ['front_camera', 'rear_camera', 'left_camera', 'right_camera']
    for cam in cameras:
        # Image (RGB) -> /<cam>/rgb/image_raw
        bridge_args.append(f'/{cam}/rgb/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image')
        # Depth -> /<cam>/depth/image_raw
        bridge_args.append(f'/{cam}/depth/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image')
        # Camera Info
        bridge_args.append(f'/{cam}/rgb/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        remappings=[
            ('/lidar/points/points', '/lidar/points'),
        ],
        output='screen'
    )

    # -----------------------------------------------------------------------
    # 8. STATIC TF
    # -----------------------------------------------------------------------
    lidar_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'lidar_link',
            '--child-frame-id', 'ugv_pindad/base_footprint/velodyne_lidar',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # -----------------------------------------------------------------------
    # 9. RVIZ2
    # -----------------------------------------------------------------------
    rviz_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ]
    )

    # -----------------------------------------------------------------------
    # 10. QT VISUALIZATION & CONTROL
    # -----------------------------------------------------------------------
    from launch.actions import ExecuteProcess
    
    # Run camera_viz.py directly from source script
    camera_viz_script = os.path.join(
        os.environ['HOME'], 'ugv_pindad_real/ugv_pindad_bringup/scripts/camera_viz.py'
    )
    
    camera_viz_node = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', camera_viz_script],
                output='screen'
            )
        ]
    )

    qt_control_script = os.path.join(
        os.environ['HOME'], 'ugv_pindad_real/ugv_pindad_bringup/ugv_pindad_bringup/qt_control_panel.py'
    )

    qt_control_node = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', qt_control_script],
                output='screen'
            )
        ]
    )

    # -----------------------------------------------------------------------
    # RETURN
    # -----------------------------------------------------------------------
    return LaunchDescription([
        gz_resource_path,
        gz_sim,
        node_robot_state_publisher,
        spawn_entity,
        bridge,
        lidar_frame_bridge,
        rviz_node,
        qt_control_node,
        camera_viz_node,
    ])
