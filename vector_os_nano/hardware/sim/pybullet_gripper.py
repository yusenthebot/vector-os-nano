"""PyBullet-backed simulated gripper.

Implements GripperProtocol — drop-in replacement for SO101Gripper.
Wraps a SimulatedArm instance for physics access.

When the arm is not connected (no physics client), open/close still track
state so the gripper is safe to use in tests without a running simulation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

logger = logging.getLogger(__name__)

# Gripper joint angle for open / closed states
_GRIPPER_OPEN_RAD: float = 0.0
_GRIPPER_CLOSED_RAD: float = 1.2   # approximately closed position


class SimulatedGripper:
    """Simulated gripper backed by PyBullet.

    Implements GripperProtocol using the gripper joint on the SimulatedArm's
    physics body. Falls back to pure state tracking when the arm is not
    connected (e.g. in unit tests).

    Args:
        sim_arm: A SimulatedArm instance (may or may not be connected).
    """

    def __init__(self, sim_arm: "SimulatedArm") -> None:
        self._arm = sim_arm
        self._is_open: bool = True

    # ------------------------------------------------------------------
    # GripperProtocol implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the gripper fully.

        Returns:
            True — always succeeds (physics or state-only).
        """
        self._is_open = True
        self._drive_gripper_joint(_GRIPPER_OPEN_RAD)
        logger.debug("SimulatedGripper: open")
        return True

    def close(self) -> bool:
        """Close the gripper.

        Returns:
            True — always succeeds.
        """
        self._is_open = False
        self._drive_gripper_joint(_GRIPPER_CLOSED_RAD)
        logger.debug("SimulatedGripper: close")
        return True

    def is_holding(self) -> bool:
        """Return True if the gripper is closed on an object.

        Without force sensing this uses contact detection:
        - If arm is connected and pybullet is available, checks contact
          points between the gripper body and any other object.
        - Otherwise returns False (no physics to check against).

        Returns:
            True if an object contact is detected.
        """
        if not self._arm._connected or self._arm._physics_client is None:
            return False
        if self._is_open:
            return False
        if self._arm._robot_id is None:
            return False

        # Check contact points on the gripper body
        try:
            from vector_os_nano.hardware.sim.pybullet_arm import _get_pybullet  # noqa: PLC0415

            p = _get_pybullet()
            contacts = p.getContactPoints(
                self._arm._robot_id,
                physicsClientId=self._arm._physics_client,
            )
            # Filter out self-contacts (same body)
            external = [
                c for c in (contacts or [])
                if c[2] != self._arm._robot_id
            ]
            return len(external) > 0
        except Exception:  # noqa: BLE001
            return False

    def get_position(self) -> float:
        """Return normalised gripper position.

        Returns:
            1.0 = fully open, 0.0 = fully closed.
        """
        return 1.0 if self._is_open else 0.0

    def get_force(self) -> float | None:
        """Grip force — not available in simulation v0.1.

        Returns:
            None always.
        """
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drive_gripper_joint(self, target_rad: float) -> None:
        """Command the gripper joint in PyBullet if arm is connected."""
        if not self._arm._connected:
            return
        if self._arm._physics_client is None:
            return
        if self._arm._gripper_joint_index is None:
            return

        try:
            from vector_os_nano.hardware.sim.pybullet_arm import _get_pybullet  # noqa: PLC0415

            p = _get_pybullet()
            p.setJointMotorControl2(
                self._arm._robot_id,
                self._arm._gripper_joint_index,
                p.POSITION_CONTROL,
                targetPosition=target_rad,
                physicsClientId=self._arm._physics_client,
            )
            # Step a few cycles so the gripper actually moves
            for _ in range(24):
                p.stepSimulation(physicsClientId=self._arm._physics_client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SimulatedGripper: could not drive joint: %s", exc)
