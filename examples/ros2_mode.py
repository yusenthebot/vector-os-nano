"""ROS2 mode -- programmatic node launch.

For most users, the launch file is simpler:
    ros2 launch vector_os nano.launch.py serial_port:=/dev/ttyACM0

This example shows how to start a single ROS2 node from Python directly,
which is useful when integrating Vector OS into an existing ROS2 system.

Requirements:
    Ubuntu 22.04 + ROS2 Humble
    pip install vector-os-nano[all]
    source /opt/ros/humble/setup.bash

Usage:
    source /opt/ros/humble/setup.bash
    python examples/ros2_mode.py
"""

from vector_os.ros2 import ROS2_AVAILABLE

if not ROS2_AVAILABLE:
    print(
        "ROS2 is not available.\n"
        "Install ROS2 Humble: https://docs.ros.org/en/humble/Installation.html\n"
        "Then source the setup: source /opt/ros/humble/setup.bash"
    )
    raise SystemExit(1)

import rclpy
from vector_os.ros2.nodes.hardware_bridge import HardwareBridgeNode

rclpy.init()

# HardwareBridgeNode is a LifecycleNode -- it manages connect/disconnect
# transitions automatically and publishes /joint_states at 30 Hz.
node = HardwareBridgeNode()

print("Hardware bridge running.")
print("Published topics: /joint_states")
print("Services: /gripper/open, /gripper/close")
print("Actions: /follow_joint_trajectory, /gripper_command")
print("Press Ctrl-C to stop.")

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
