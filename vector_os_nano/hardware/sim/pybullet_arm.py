# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PyBullet-based simulated SO-101 arm.

Implements ArmProtocol — drop-in replacement for SO101Arm.
PyBullet is imported lazily so the module is safe to import on systems
without pybullet installed (import error deferred until connect()).

Usage:
    arm = SimulatedArm(gui=False)   # headless (default)
    arm.connect()
    arm.move_joints([0.0, -1.2, 0.5, 0.8, 0.3])
    arm.disconnect()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy PyBullet import
# ---------------------------------------------------------------------------

_pybullet: Any = None


def _get_pybullet() -> Any:
    global _pybullet
    if _pybullet is None:
        import pybullet as p  # noqa: PLC0415

        _pybullet = p
    return _pybullet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Simulation step rate (Hz) — PyBullet default
_SIM_HZ = 240

# Default URDF for simulation (primitive-geometry version, no mesh deps)
_DEFAULT_SIM_URDF = (
    Path(__file__).parent.parent / "urdf" / "so101_sim.urdf"
)


# ---------------------------------------------------------------------------
# SimulatedArm
# ---------------------------------------------------------------------------


class SimulatedArm:
    """SO-101 arm in PyBullet simulation.

    ArmProtocol-compatible drop-in replacement for SO101Arm.  Runs a full
    rigid-body dynamics simulation in DIRECT mode (headless) by default.

    Attributes:
        _gui: Whether to open a GUI window on connect().
        _urdf_path: Absolute path to the URDF to load.
        _physics_client: PyBullet physics server ID (set on connect).
        _robot_id: Body ID of the loaded robot (set on connect).
        _connected: True after a successful connect().
        _arm_joint_indices: Ordered list of joint indices for the 5 arm DOFs.
        _gripper_joint_index: Joint index for the gripper joint (or None).
        _ik_solver: Optional external IK solver (injected via set_ik_solver).
    """

    def __init__(
        self,
        gui: bool = False,
        urdf_path: str | None = None,
    ) -> None:
        self._gui: bool = gui
        self._urdf_path: str = (
            urdf_path if urdf_path is not None else str(_DEFAULT_SIM_URDF)
        )
        self._physics_client: int | None = None
        self._robot_id: int | None = None
        self._connected: bool = False
        self._arm_joint_indices: list[int] = []
        self._gripper_joint_index: int | None = None
        self._ik_solver: Any = None

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable identifier."""
        return "sim_so101"

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of arm joint names."""
        return list(_ARM_JOINT_NAMES)

    @property
    def dof(self) -> int:
        """Degrees of freedom (arm joints, excluding gripper)."""
        return 5

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start PyBullet, load URDF, set up table scene.

        Raises:
            RuntimeError: If PyBullet cannot be imported or URDF fails to load.
        """
        p = _get_pybullet()

        mode = p.GUI if self._gui else p.DIRECT
        self._physics_client = p.connect(mode)

        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)

        # Load ground plane from pybullet_data
        import pybullet_data  # noqa: PLC0415

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._physics_client,
        )
        p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        # Simple table surface (static box)
        table_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.01],
            physicsClientId=self._physics_client,
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.01],
            rgbaColor=[0.8, 0.7, 0.5, 1.0],
            physicsClientId=self._physics_client,
        )
        p.createMultiBody(
            0,
            table_col,
            table_vis,
            basePosition=[0.3, 0.0, 0.01],
            physicsClientId=self._physics_client,
        )

        # Load robot
        self._robot_id = p.loadURDF(
            self._urdf_path,
            basePosition=[0.0, 0.0, 0.02],
            useFixedBase=True,
            physicsClientId=self._physics_client,
        )

        # Discover joint indices by name
        self._arm_joint_indices = []
        self._gripper_joint_index = None
        num_joints = p.getNumJoints(
            self._robot_id, physicsClientId=self._physics_client
        )
        for i in range(num_joints):
            info = p.getJointInfo(
                self._robot_id, i, physicsClientId=self._physics_client
            )
            joint_name = info[1].decode("utf-8")
            if joint_name in _ARM_JOINT_NAMES:
                self._arm_joint_indices.append(i)
            elif joint_name == "gripper":
                self._gripper_joint_index = i

        # Sort arm joints to match _ARM_JOINT_NAMES order
        name_to_idx = {}
        for i in range(num_joints):
            info = p.getJointInfo(
                self._robot_id, i, physicsClientId=self._physics_client
            )
            name_to_idx[info[1].decode("utf-8")] = i
        self._arm_joint_indices = [
            name_to_idx[n] for n in _ARM_JOINT_NAMES if n in name_to_idx
        ]

        self._connected = True
        logger.info(
            "SimulatedArm connected (PyBullet DIRECT=%s), arm_joints=%s",
            not self._gui,
            self._arm_joint_indices,
        )

    def disconnect(self) -> None:
        """Disconnect from PyBullet. Idempotent — safe to call twice."""
        if self._physics_client is not None:
            p = _get_pybullet()
            try:
                p.disconnect(self._physics_client)
            except Exception:  # noqa: BLE001
                pass
            self._physics_client = None
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "SimulatedArm: not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Read current joint positions in radians from PyBullet.

        Returns:
            List of 5 floats in _ARM_JOINT_NAMES order.

        Raises:
            RuntimeError: if not connected.
        """
        self._require_connection()
        p = _get_pybullet()
        return [
            float(
                p.getJointState(
                    self._robot_id, idx, physicsClientId=self._physics_client
                )[0]
            )
            for idx in self._arm_joint_indices
        ]

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def move_joints(
        self,
        positions: list[float],
        duration: float = 3.0,
    ) -> bool:
        """Move joints using PyBullet position control.

        Steps the simulation for ``duration * _SIM_HZ`` steps at 240 Hz.

        Args:
            positions: Target joint positions in radians, length must equal dof.
            duration: Simulated motion duration in seconds.

        Returns:
            True on completion.

        Raises:
            ValueError: if len(positions) != dof.
            RuntimeError: if not connected.
        """
        self._require_connection()
        if len(positions) != self.dof:
            raise ValueError(
                f"SimulatedArm.move_joints: expected {self.dof} positions, "
                f"got {len(positions)}"
            )

        p = _get_pybullet()
        for idx, pos in zip(self._arm_joint_indices, positions):
            p.setJointMotorControl2(
                self._robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=pos,
                physicsClientId=self._physics_client,
            )

        steps = max(1, int(duration * _SIM_HZ))
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._physics_client)

        return True

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Move end-effector to target Cartesian position via IK.

        Uses PyBullet's built-in IK if no external solver is set.

        Args:
            target_xyz: Target (x, y, z) in metres, base frame.
            duration: Simulated motion duration in seconds.

        Returns:
            True if IK solution found and motion completed, False otherwise.

        Raises:
            RuntimeError: if not connected.
        """
        self._require_connection()
        solution = self.ik(target_xyz)
        if solution is None:
            return False
        return self.move_joints(solution, duration=duration)

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics via PyBullet link state.

        Temporarily moves to joint_positions, reads the end-effector link
        world position, then restores the original configuration.

        Args:
            joint_positions: Joint positions in radians, length == dof.

        Returns:
            (position_xyz, rotation_3x3) tuple.
        """
        self._require_connection()
        p = _get_pybullet()

        # Save current positions
        original = self.get_joint_positions()

        # Set requested joints (no simulation steps — just reset state)
        for idx, pos in zip(self._arm_joint_indices, joint_positions):
            p.resetJointState(
                self._robot_id,
                idx,
                pos,
                physicsClientId=self._physics_client,
            )

        # Read end-effector link state (last arm joint link)
        ee_link = self._arm_joint_indices[-1]
        link_state = p.getLinkState(
            self._robot_id,
            ee_link,
            physicsClientId=self._physics_client,
        )
        pos_world = list(link_state[0])  # world position of link frame

        # Restore original joint positions
        for idx, orig in zip(self._arm_joint_indices, original):
            p.resetJointState(
                self._robot_id,
                idx,
                orig,
                physicsClientId=self._physics_client,
            )

        # Build identity rotation (PyBullet FK gives quaternion; return 3x3 identity
        # as a reasonable default — full rotation matrix needs quaternion conversion)
        rotation_3x3: list[list[float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        return pos_world, rotation_3x3

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Inverse kinematics using PyBullet's built-in numerical IK.

        Args:
            target_xyz: Target (x, y, z) in metres, robot base frame.
            current_joints: Seed configuration (ignored — PyBullet uses
                            current robot state as seed automatically).

        Returns:
            Joint positions in radians (5 values) if solution found.
        """
        self._require_connection()
        p = _get_pybullet()

        ee_link = self._arm_joint_indices[-1]
        solution = p.calculateInverseKinematics(
            self._robot_id,
            ee_link,
            target_xyz,
            physicsClientId=self._physics_client,
        )
        # calculateInverseKinematics returns a tuple for ALL movable joints —
        # take only the first len(arm_joint_indices) values corresponding to arm
        arm_solution = [float(v) for v in solution[: self.dof]]
        return arm_solution

    def stop(self) -> None:
        """Stop all motion by commanding current joint positions as targets.

        Raises:
            RuntimeError: if not connected.
        """
        self._require_connection()
        positions = self.get_joint_positions()
        self.move_joints(positions, duration=0.05)

    def set_ik_solver(self, solver: Any) -> None:
        """Inject an external IK solver (e.g. Pinocchio-based IKSolver).

        The solver must implement:
            solver.ik(target_xyz, current_joints) -> list[float] | None
            solver.fk(joint_positions) -> tuple[list[float], list[list[float]]]
        """
        self._ik_solver = solver

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_object(
        self,
        name: str,
        position: list[float],
        color: list[float] | None = None,
        size: float = 0.03,
    ) -> int:
        """Add a cube object to the PyBullet scene.

        Args:
            name: Descriptive label (not used by PyBullet, for readability).
            position: [x, y, z] world position in metres.
            color: RGBA colour list (default red [1, 0, 0, 1]).
            size: Half-extent of the cube in metres.

        Returns:
            PyBullet body ID of the created object.
        """
        self._require_connection()
        p = _get_pybullet()

        if color is None:
            color = [1.0, 0.0, 0.0, 1.0]

        half = size / 2.0
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half, half, half],
            physicsClientId=self._physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half, half, half],
            rgbaColor=color,
            physicsClientId=self._physics_client,
        )
        obj_id = p.createMultiBody(
            0.05,
            col,
            vis,
            basePosition=position,
            physicsClientId=self._physics_client,
        )
        logger.debug("SimulatedArm: added object '%s' id=%d at %s", name, obj_id, position)
        return obj_id
