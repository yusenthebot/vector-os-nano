"""vector_os_nano.hardware.sim — PyBullet simulation backend.

Provides drop-in replacements for hardware arm and gripper,
allowing skills and the Agent to run in headless simulation
without physical hardware.

Exports:
    SimulatedArm    -- ArmProtocol-compatible PyBullet arm
    SimulatedGripper -- GripperProtocol-compatible PyBullet gripper

PyBullet is imported lazily inside each class; importing this module
does NOT require pybullet to be installed.
"""
from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm
from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

__all__ = ["SimulatedArm", "SimulatedGripper"]
