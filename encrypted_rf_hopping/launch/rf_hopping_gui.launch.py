#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='encrypted_rf_hopping',
            executable='pub_gui',
            name='encrypted_rf_hopping_pub_gui',
            output='screen',
        ),
        Node(
            package='encrypted_rf_hopping',
            executable='sub_gui',
            name='encrypted_rf_hopping_sub_gui',
            output='screen',
        ),
    ])
