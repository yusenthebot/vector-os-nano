# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ROS2 integration layer for Vector OS Nano. Optional.

Usage:
    from vector_os_nano.ros2 import ROS2_AVAILABLE

    if ROS2_AVAILABLE:
        from vector_os_nano.ros2 import HardwareBridgeNode, PerceptionBridgeNode

This module intentionally does NOT import any ROS2 modules at the top level.
All ROS2 imports are guarded so the SDK works without ROS2 installed.
"""
from __future__ import annotations

ROS2_AVAILABLE: bool = False
try:
    import rclpy  # noqa: F401
    ROS2_AVAILABLE = True
except ImportError:
    pass

if ROS2_AVAILABLE:
    from vector_os_nano.ros2.nodes.hardware_bridge import HardwareBridgeNode
    from vector_os_nano.ros2.nodes.perception_node import PerceptionBridgeNode
    from vector_os_nano.ros2.nodes.skill_server import SkillServerNode
    from vector_os_nano.ros2.nodes.world_model_node import WorldModelServiceNode
    from vector_os_nano.ros2.nodes.agent_node import AgentNode

    __all__ = [
        "ROS2_AVAILABLE",
        "HardwareBridgeNode",
        "PerceptionBridgeNode",
        "SkillServerNode",
        "WorldModelServiceNode",
        "AgentNode",
    ]
else:
    __all__ = ["ROS2_AVAILABLE"]
