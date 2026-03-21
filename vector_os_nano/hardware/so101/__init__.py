"""SO-101 robot arm implementation.

Exports (when Task 2 is complete):
    SO101Arm     — 5-DOF arm driver (ArmProtocol)
    SO101Gripper — gripper driver (GripperProtocol)
    SerialBus    — low-level SCS serial communication
    joint_config — encoder/radian mapping constants
"""
from __future__ import annotations

# joint_config is available in Task 1 (already ported)
from vector_os_nano.hardware.so101.joint_config import (  # noqa: F401
    JOINT_CONFIG,
    ARM_JOINT_NAMES,
    ALL_JOINT_NAMES,
)

# Hardware driver modules are implemented in Task 2
try:
    from vector_os_nano.hardware.so101.arm import SO101Arm
    from vector_os_nano.hardware.so101.gripper import SO101Gripper
    from vector_os_nano.hardware.so101.serial_bus import SerialBus
    __all__ = ["SO101Arm", "SO101Gripper", "SerialBus", "JOINT_CONFIG", "ARM_JOINT_NAMES", "ALL_JOINT_NAMES"]
except ImportError:
    __all__ = ["JOINT_CONFIG", "ARM_JOINT_NAMES", "ALL_JOINT_NAMES"]
