"""vector_os_nano.hardware.sim — simulation backends.

Provides drop-in replacements for hardware arm and gripper,
allowing skills and the Agent to run in headless simulation
without physical hardware.

Two backends available:
    MuJoCo (recommended) — higher fidelity contact physics, real grasping
    PyBullet (legacy)    — lighter weight, basic simulation

All physics engines are imported lazily; importing this module does NOT
require mujoco or pybullet to be installed.
"""
from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm
from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

__all__ = [
    "MuJoCoArm",
    "MuJoCoGripper",
    "MuJoCoPerception",
    "SimulatedArm",
    "SimulatedGripper",
]
