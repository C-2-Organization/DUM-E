#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dum_e_bringup',
            executable='robot_state_care_node',
            name='robot_state_care_node',
            output='screen',
        )
    ])
