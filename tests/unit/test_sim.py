"""Unit tests for SimulatedArm and SimulatedGripper.

These tests do NOT require pybullet to be installed.
They validate construction, properties, and state machine behaviour
that can be verified without running the physics engine.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# SimulatedArm — no-pybullet unit tests
# ---------------------------------------------------------------------------


class TestSimulatedArmCreation:
    """SimulatedArm can be instantiated without connecting."""

    def test_default_construction(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm is not None

    def test_gui_false_by_default(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm._gui is False

    def test_gui_true_accepted(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=True)
        assert arm._gui is True

    def test_custom_urdf_path(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(urdf_path="/tmp/fake.urdf")
        assert arm._urdf_path == "/tmp/fake.urdf"

    def test_not_connected_initially(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm._connected is False

    def test_physics_client_none_before_connect(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm._physics_client is None

    def test_robot_id_none_before_connect(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm._robot_id is None


class TestSimulatedArmProperties:
    """ArmProtocol property contract."""

    def test_name(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm.name == "sim_so101"

    def test_dof(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert arm.dof == 5

    def test_joint_names_length(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert len(arm.joint_names) == 5

    def test_joint_names_content(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        expected = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        assert arm.joint_names == expected

    def test_joint_names_are_strings(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        for name in arm.joint_names:
            assert isinstance(name, str)

    def test_name_is_string(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert isinstance(arm.name, str)

    def test_dof_is_int(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert isinstance(arm.dof, int)


class TestSimulatedArmProtocolCompliance:
    """Verify SimulatedArm satisfies ArmProtocol at runtime."""

    def test_isinstance_arm_protocol(self) -> None:
        from vector_os_nano.hardware.arm import ArmProtocol
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert isinstance(arm, ArmProtocol)

    def test_has_connect_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "connect", None))

    def test_has_disconnect_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "disconnect", None))

    def test_has_get_joint_positions_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "get_joint_positions", None))

    def test_has_move_joints_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "move_joints", None))

    def test_has_move_cartesian_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "move_cartesian", None))

    def test_has_fk_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "fk", None))

    def test_has_ik_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "ik", None))

    def test_has_stop_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "stop", None))

    def test_has_set_ik_solver_method(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm()
        assert callable(getattr(arm, "set_ik_solver", None))


# ---------------------------------------------------------------------------
# SimulatedGripper — no-pybullet unit tests
# ---------------------------------------------------------------------------


class _MockArm:
    """Minimal mock satisfying SimulatedGripper's constructor requirement."""

    def __init__(self) -> None:
        self._physics_client: int | None = None
        self._robot_id: int | None = None
        self._gripper_joint_index: int | None = None
        self._connected: bool = False


class TestSimulatedGripperCreation:
    """SimulatedGripper can be created without a physics session."""

    def test_creation_with_mock_arm(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        assert gripper is not None

    def test_initial_state_open(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        assert gripper._is_open is True

    def test_arm_reference_stored(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        mock = _MockArm()
        gripper = SimulatedGripper(mock)
        assert gripper._arm is mock


class TestSimulatedGripperOpenClose:
    """State toggles correctly without PyBullet (arm not connected)."""

    def test_open_returns_true(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        result = gripper.open()
        assert result is True

    def test_close_returns_true(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        result = gripper.close()
        assert result is True

    def test_open_sets_state_open(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.close()
        gripper.open()
        assert gripper._is_open is True

    def test_close_sets_state_closed(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.close()
        assert gripper._is_open is False

    def test_toggle_open_close_open(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.close()
        assert gripper._is_open is False
        gripper.open()
        assert gripper._is_open is True

    def test_get_position_open(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.open()
        assert gripper.get_position() == pytest.approx(1.0)

    def test_get_position_closed(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.close()
        assert gripper.get_position() == pytest.approx(0.0)

    def test_is_holding_when_open(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.open()
        assert gripper.is_holding() is False

    def test_is_holding_when_closed_no_object(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        gripper.close()
        # Without physics engine, no object contact — not holding
        assert gripper.is_holding() is False

    def test_get_force_returns_none(self) -> None:
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        assert gripper.get_force() is None


class TestSimulatedGripperProtocolCompliance:
    """GripperProtocol satisfied."""

    def test_isinstance_gripper_protocol(self) -> None:
        from vector_os_nano.hardware.gripper import GripperProtocol
        from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

        gripper = SimulatedGripper(_MockArm())
        assert isinstance(gripper, GripperProtocol)


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------


class TestSimPackageExports:
    """sim/__init__.py exports the public API."""

    def test_import_simulated_arm(self) -> None:
        from vector_os_nano.hardware.sim import SimulatedArm  # noqa: F401

    def test_import_simulated_gripper(self) -> None:
        from vector_os_nano.hardware.sim import SimulatedGripper  # noqa: F401

    def test_simulated_arm_is_class(self) -> None:
        from vector_os_nano.hardware.sim import SimulatedArm

        assert isinstance(SimulatedArm, type)

    def test_simulated_gripper_is_class(self) -> None:
        from vector_os_nano.hardware.sim import SimulatedGripper

        assert isinstance(SimulatedGripper, type)
