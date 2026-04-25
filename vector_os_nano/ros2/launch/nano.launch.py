# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Vector OS Nano full-system launch file.

Launches the hardware bridge, perception, skill server, world model service,
and agent node in staggered order so each layer is ready before the next starts.

Usage:
    ros2 launch vector_os nano.launch.py
    ros2 launch vector_os nano.launch.py serial_port:=/dev/ttyACM1
    ros2 launch vector_os nano.launch.py use_rviz:=true

Startup order:
    t=0s  hardware_bridge    — connects to servos immediately
    t=3s  perception_bridge  — waits for camera driver to initialise
    t=5s  world_model        — waits for joint_states to be available
    t=6s  skill_server       — waits for hardware + world model
    t=7s  agent_node         — waits for all services to be live
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the Vector OS Nano full-system launch description."""

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyACM0",
        description="Serial port for the SO-101 hardware bridge",
    )
    baudrate_arg = DeclareLaunchArgument(
        "baudrate",
        default_value="1000000",
        description="Baud rate for the SO-101 serial bus",
    )
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="30.0",
        description="Joint state publish rate in Hz",
    )
    color_topic_arg = DeclareLaunchArgument(
        "color_topic",
        default_value="/camera/color/image_raw",
        description="RGB image topic for the perception bridge",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/aligned_depth_to_color/image_raw",
        description="Aligned depth image topic",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz visualiser",
    )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    # t=0s: hardware bridge (immediate — must be first)
    hardware_bridge = Node(
        package="vector_os",
        executable="hardware_bridge",
        name="hardware_bridge",
        output="screen",
        parameters=[
            {
                "serial_port": LaunchConfiguration("serial_port"),
                "baudrate": LaunchConfiguration("baudrate"),
                "publish_rate": LaunchConfiguration("publish_rate"),
            }
        ],
    )

    # t=3s: perception bridge (waits for camera driver)
    perception_bridge = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="vector_os",
                executable="perception_bridge",
                name="perception_bridge",
                output="screen",
                parameters=[
                    {
                        "color_topic": LaunchConfiguration("color_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                    }
                ],
            ),
        ],
    )

    # t=5s: world model service (waits for joint_states to be published)
    world_model = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="vector_os",
                executable="world_model_service",
                name="world_model_service",
                output="screen",
            ),
        ],
    )

    # t=6s: skill server (waits for hardware + world model)
    skill_server = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="vector_os",
                executable="skill_server",
                name="skill_server",
                output="screen",
            ),
        ],
    )

    # t=7s: agent node (waits for all services to be live)
    agent_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package="vector_os",
                executable="agent",
                name="agent",
                output="screen",
            ),
        ],
    )

    return LaunchDescription(
        [
            serial_port_arg,
            baudrate_arg,
            publish_rate_arg,
            color_topic_arg,
            depth_topic_arg,
            use_rviz_arg,
            hardware_bridge,
            perception_bridge,
            world_model,
            skill_server,
            agent_node,
        ]
    )
