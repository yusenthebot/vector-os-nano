"""Integration tests for PyBullet-backed SimulatedArm and SimulatedGripper.

All tests skip if pybullet is not installed.
Physics runs in DIRECT mode (no GUI window).
"""
from __future__ import annotations

import math

import pytest

pybullet = pytest.importorskip("pybullet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_arm():
    """Connected SimulatedArm in DIRECT mode. Disconnects after test."""
    from vector_os.hardware.sim.pybullet_arm import SimulatedArm

    arm = SimulatedArm(gui=False)
    arm.connect()
    yield arm
    arm.disconnect()


@pytest.fixture
def sim_gripper(sim_arm):
    """SimulatedGripper backed by a connected sim_arm."""
    from vector_os.hardware.sim.pybullet_gripper import SimulatedGripper

    return SimulatedGripper(sim_arm)


# ---------------------------------------------------------------------------
# Connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    def test_connect_sets_connected_flag(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        try:
            assert arm._connected is True
        finally:
            arm.disconnect()

    def test_connect_assigns_physics_client(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        try:
            assert arm._physics_client is not None
        finally:
            arm.disconnect()

    def test_connect_assigns_robot_id(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        try:
            assert arm._robot_id is not None
        finally:
            arm.disconnect()

    def test_connect_populates_arm_joint_indices(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        try:
            assert len(arm._arm_joint_indices) == arm.dof
        finally:
            arm.disconnect()

    def test_disconnect_clears_connected_flag(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        arm.disconnect()
        assert arm._connected is False

    def test_disconnect_clears_physics_client(self) -> None:
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        arm.disconnect()
        assert arm._physics_client is None

    def test_disconnect_idempotent(self) -> None:
        """Second disconnect() must not raise."""
        from vector_os.hardware.sim.pybullet_arm import SimulatedArm

        arm = SimulatedArm(gui=False)
        arm.connect()
        arm.disconnect()
        arm.disconnect()  # should not raise


# ---------------------------------------------------------------------------
# get_joint_positions
# ---------------------------------------------------------------------------


class TestGetJointPositions:
    def test_returns_list_of_correct_length(self, sim_arm) -> None:
        positions = sim_arm.get_joint_positions()
        assert isinstance(positions, list)
        assert len(positions) == sim_arm.dof

    def test_all_values_are_float(self, sim_arm) -> None:
        for pos in sim_arm.get_joint_positions():
            assert isinstance(pos, float)

    def test_initial_positions_near_zero(self, sim_arm) -> None:
        """Robot starts at zero config."""
        for pos in sim_arm.get_joint_positions():
            assert abs(pos) < 0.5  # within 0.5 rad of zero at startup


# ---------------------------------------------------------------------------
# move_joints
# ---------------------------------------------------------------------------


class TestMoveJoints:
    HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
    POSE_A = [0.3, -0.5, 0.4, 0.2, -0.1]

    def test_move_to_home_returns_true(self, sim_arm) -> None:
        result = sim_arm.move_joints(self.HOME, duration=0.1)
        assert result is True

    def test_move_to_pose_a_returns_true(self, sim_arm) -> None:
        result = sim_arm.move_joints(self.POSE_A, duration=0.1)
        assert result is True

    def test_joint_positions_track_target(self, sim_arm) -> None:
        """After moving, read-back should be close to commanded target."""
        sim_arm.move_joints(self.POSE_A, duration=0.5)
        actual = sim_arm.get_joint_positions()
        for cmd, got in zip(self.POSE_A, actual):
            assert abs(cmd - got) < 0.3, f"joint error: cmd={cmd:.3f} got={got:.3f}"

    def test_wrong_dof_raises_value_error(self, sim_arm) -> None:
        with pytest.raises(ValueError):
            sim_arm.move_joints([0.0, 0.0, 0.0])  # only 3 joints

    def test_empty_positions_raises_value_error(self, sim_arm) -> None:
        with pytest.raises(ValueError):
            sim_arm.move_joints([])


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_does_not_raise(self, sim_arm) -> None:
        sim_arm.move_joints([0.2, -0.3, 0.1, 0.0, 0.0], duration=0.1)
        sim_arm.stop()  # must not raise

    def test_stop_returns_none(self, sim_arm) -> None:
        result = sim_arm.stop()
        assert result is None


# ---------------------------------------------------------------------------
# add_object
# ---------------------------------------------------------------------------


class TestAddObject:
    def test_add_object_returns_int(self, sim_arm) -> None:
        obj_id = sim_arm.add_object("cube", [0.3, 0.0, 0.05])
        assert isinstance(obj_id, int)

    def test_add_multiple_objects(self, sim_arm) -> None:
        id1 = sim_arm.add_object("red", [0.3, 0.0, 0.05], color=[1, 0, 0, 1])
        id2 = sim_arm.add_object("blue", [0.4, 0.0, 0.05], color=[0, 0, 1, 1])
        assert id1 != id2

    def test_add_object_custom_size(self, sim_arm) -> None:
        obj_id = sim_arm.add_object("small", [0.3, 0.0, 0.05], size=0.02)
        assert obj_id >= 0


# ---------------------------------------------------------------------------
# IK solution
# ---------------------------------------------------------------------------


class TestIKSolution:
    def test_ik_returns_list_of_floats(self, sim_arm) -> None:
        solution = sim_arm.ik((0.2, 0.0, 0.1))
        assert isinstance(solution, list)
        for v in solution:
            assert isinstance(v, float)

    def test_ik_returns_correct_dof_count(self, sim_arm) -> None:
        solution = sim_arm.ik((0.2, 0.0, 0.1))
        assert len(solution) == sim_arm.dof

    def test_ik_values_are_finite(self, sim_arm) -> None:
        solution = sim_arm.ik((0.2, 0.0, 0.1))
        for v in solution:
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# fk
# ---------------------------------------------------------------------------


class TestFK:
    def test_fk_returns_tuple(self, sim_arm) -> None:
        result = sim_arm.fk([0.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fk_position_is_list_of_3(self, sim_arm) -> None:
        pos, rot = sim_arm.fk([0.0, 0.0, 0.0, 0.0, 0.0])
        assert len(pos) == 3

    def test_fk_position_values_finite(self, sim_arm) -> None:
        pos, _ = sim_arm.fk([0.0, 0.0, 0.0, 0.0, 0.0])
        for v in pos:
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# move_cartesian (IK-backed)
# ---------------------------------------------------------------------------


class TestMoveCartesian:
    def test_move_cartesian_returns_bool(self, sim_arm) -> None:
        result = sim_arm.move_cartesian((0.2, 0.0, 0.1), duration=0.1)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Gripper integration
# ---------------------------------------------------------------------------


class TestSimulatedGripperIntegration:
    def test_open_returns_true(self, sim_gripper) -> None:
        assert sim_gripper.open() is True

    def test_close_returns_true(self, sim_gripper) -> None:
        assert sim_gripper.close() is True

    def test_get_position_open(self, sim_gripper) -> None:
        sim_gripper.open()
        pos = sim_gripper.get_position()
        assert isinstance(pos, float)
        assert pos == pytest.approx(1.0)

    def test_get_position_closed(self, sim_gripper) -> None:
        sim_gripper.close()
        pos = sim_gripper.get_position()
        assert isinstance(pos, float)
        assert pos == pytest.approx(0.0)

    def test_gripper_physics_open(self, sim_gripper, sim_arm) -> None:
        """When arm is connected, open() drives the gripper joint."""
        sim_gripper.open()
        # Joint should have been commanded — no exception
        assert sim_gripper._is_open is True

    def test_gripper_physics_close(self, sim_gripper, sim_arm) -> None:
        """When arm is connected, close() drives the gripper joint."""
        sim_gripper.close()
        assert sim_gripper._is_open is False
