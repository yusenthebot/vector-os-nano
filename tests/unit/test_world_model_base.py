"""Unit tests for Go2 MuJoCo base integration — WorldModel + Agent.

TDD RED phase: tests written before implementation.
Covers:
    - RobotState.position_xy and .heading fields
    - WorldModel.update_robot_state() with position_xy / heading
    - RobotState.to_dict() / from_dict() round-trip
    - Agent(base=...) stores self._base
    - Agent._build_context() passes base= to SkillContext
    - Agent._sync_robot_state() pulls position/heading from base
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wm():
    from vector_os_nano.core.world_model import WorldModel
    return WorldModel()


@pytest.fixture
def mock_base():
    base = MagicMock()
    base.get_position.return_value = (3.0, 4.0, 0.0)
    base.get_heading.return_value = 1.57
    return base


# ---------------------------------------------------------------------------
# RobotState default fields
# ---------------------------------------------------------------------------

class TestRobotStateBaseFields:
    def test_robot_state_has_position_xy(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState()
        assert rs.position_xy == (0.0, 0.0)

    def test_robot_state_has_heading(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState()
        assert rs.heading == 0.0

    def test_robot_state_position_xy_custom(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState(position_xy=(1.5, 2.3))
        assert rs.position_xy == pytest.approx((1.5, 2.3))

    def test_robot_state_heading_custom(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState(heading=0.5)
        assert rs.heading == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# WorldModel.update_robot_state() with new fields
# ---------------------------------------------------------------------------

class TestUpdateRobotStateBaseFields:
    def test_update_robot_state_position_xy(self, wm):
        wm.update_robot_state(position_xy=(1.5, 2.3))
        assert wm.get_robot().position_xy == pytest.approx((1.5, 2.3))

    def test_update_robot_state_heading(self, wm):
        wm.update_robot_state(heading=0.5)
        assert wm.get_robot().heading == pytest.approx(0.5)

    def test_update_position_xy_preserves_other_fields(self, wm):
        wm.update_robot_state(gripper_state="closed")
        wm.update_robot_state(position_xy=(1.0, 2.0))
        robot = wm.get_robot()
        assert robot.gripper_state == "closed"
        assert robot.position_xy == pytest.approx((1.0, 2.0))

    def test_update_heading_preserves_other_fields(self, wm):
        wm.update_robot_state(position_xy=(1.0, 2.0))
        wm.update_robot_state(heading=0.8)
        robot = wm.get_robot()
        assert robot.position_xy == pytest.approx((1.0, 2.0))
        assert robot.heading == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestRobotStateSerializationBase:
    def test_to_dict_includes_position_xy(self, wm):
        wm.update_robot_state(position_xy=(1.5, 2.3))
        d = wm.get_robot().to_dict()
        assert "position_xy" in d
        assert d["position_xy"] == pytest.approx([1.5, 2.3])

    def test_to_dict_includes_heading(self, wm):
        wm.update_robot_state(heading=0.5)
        d = wm.get_robot().to_dict()
        assert "heading" in d
        assert d["heading"] == pytest.approx(0.5)

    def test_from_dict_round_trips_position_xy(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState(position_xy=(1.5, 2.3), heading=0.5)
        d = rs.to_dict()
        restored = RobotState.from_dict(d)
        assert restored.position_xy == pytest.approx((1.5, 2.3))

    def test_from_dict_round_trips_heading(self):
        from vector_os_nano.core.world_model import RobotState
        rs = RobotState(position_xy=(1.5, 2.3), heading=0.5)
        d = rs.to_dict()
        restored = RobotState.from_dict(d)
        assert restored.heading == pytest.approx(0.5)

    def test_from_dict_defaults_position_xy_when_absent(self):
        from vector_os_nano.core.world_model import RobotState
        d = {"gripper_state": "open"}
        rs = RobotState.from_dict(d)
        assert rs.position_xy == (0.0, 0.0)

    def test_from_dict_defaults_heading_when_absent(self):
        from vector_os_nano.core.world_model import RobotState
        d = {"gripper_state": "open"}
        rs = RobotState.from_dict(d)
        assert rs.heading == 0.0


# ---------------------------------------------------------------------------
# Agent base param
# ---------------------------------------------------------------------------

class TestAgentBaseParam:
    def test_agent_base_param_stores_base(self, mock_base):
        from vector_os_nano.core.agent import Agent
        agent = Agent(base=mock_base)
        assert agent._base is mock_base

    def test_agent_base_param_defaults_to_none(self):
        from vector_os_nano.core.agent import Agent
        agent = Agent()
        assert agent._base is None

    def test_agent_build_context_passes_base(self, mock_base):
        from vector_os_nano.core.agent import Agent
        agent = Agent(base=mock_base)
        ctx = agent._build_context()
        assert ctx.base is mock_base

    def test_agent_build_context_base_none_when_no_base(self):
        from vector_os_nano.core.agent import Agent
        agent = Agent()
        ctx = agent._build_context()
        assert ctx.base is None

    def test_agent_sync_robot_state_with_base(self, mock_base):
        from vector_os_nano.core.agent import Agent
        agent = Agent(base=mock_base)
        # _sync_robot_state is called in __init__, but call explicitly too
        mock_base.get_position.return_value = (3.0, 4.0, 0.0)
        mock_base.get_heading.return_value = 1.57
        agent._sync_robot_state()
        robot = agent._world_model.get_robot()
        assert robot.position_xy == pytest.approx((3.0, 4.0))
        assert robot.heading == pytest.approx(1.57)

    def test_agent_sync_robot_state_base_exception_is_safe(self):
        """If base.get_position() raises, _sync_robot_state must not propagate."""
        from vector_os_nano.core.agent import Agent
        bad_base = MagicMock()
        bad_base.get_position.side_effect = RuntimeError("sensor error")
        bad_base.get_heading.side_effect = RuntimeError("sensor error")
        agent = Agent(base=bad_base)
        # No exception should bubble up
        agent._sync_robot_state()
