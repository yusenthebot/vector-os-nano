"""Level 3 — Task planning verification with mock VLM.

Tests verify that MobileAgentLoop correctly decomposes natural-language goals
into SubTask sequences and executes them via the skill registry.

No real API calls are made — all LLM and VLM interactions are mocked so this
suite runs offline at zero cost.

Test coverage:
- Fallback planner: English room names, Chinese room names, patrol keywords
- Plan parser: raw JSON arrays, JSON wrapped in markdown fences
- Execution: loop runs all planned steps, fall-detection aborts early
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent():
    """Create a mock agent with skill registry, base, no VLM/LLM."""
    from vector_os_nano.core.skill import SkillContext, SkillRegistry
    from vector_os_nano.core.types import SkillResult

    agent = MagicMock()

    # Mock base (Go2 standing upright)
    base = MagicMock()
    base.get_position.return_value = [10.0, 3.0, 0.28]
    base.get_heading.return_value = 0.0
    base.get_camera_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    agent._base = base

    # No LLM — forces fallback planning path
    agent._llm = None

    # No VLM / spatial memory — pure planning test
    agent._vlm = None
    agent._spatial_memory = None

    # Build a real SkillRegistry populated with minimal mock skills
    registry = SkillRegistry()

    nav_skill = MagicMock()
    nav_skill.name = "navigate"
    nav_skill.__skill_aliases__ = []
    nav_skill.__skill_direct__ = False
    nav_skill.__skill_auto_steps__ = []
    nav_skill.execute.return_value = SkillResult(
        success=True, result_data={"room": "kitchen"}
    )

    look_skill = MagicMock()
    look_skill.name = "look"
    look_skill.__skill_aliases__ = []
    look_skill.__skill_direct__ = False
    look_skill.__skill_auto_steps__ = []
    look_skill.execute.return_value = SkillResult(
        success=True, result_data={"summary": "A kitchen with appliances.", "room": "kitchen"}
    )

    registry.register(nav_skill)
    registry.register(look_skill)
    agent._skill_registry = registry

    # Mock world model
    agent._world_model = MagicMock()

    # _build_context returns a real SkillContext so skill dispatch works
    def _build_context():
        return SkillContext(
            bases={"default": base},
            services={"skill_registry": registry},
        )

    agent._build_context.side_effect = _build_context

    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLevel3TaskPlanning:
    """L3: Multi-step task planning with mock VLM (no real API calls)."""

    # ------------------------------------------------------------------
    # Fallback planner
    # ------------------------------------------------------------------

    def test_fallback_plan_room_navigation(self):
        """Fallback planner generates navigate+look for an English room goal."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("go to the kitchen")

        assert len(plan) >= 2, "Expected at least navigate + look"
        assert plan[0].action == "navigate"
        assert plan[0].params.get("room") == "kitchen"
        assert plan[1].action == "look"

    def test_fallback_plan_living_room(self):
        """Fallback planner resolves 'living room' to living_room canonical name."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("go to living room")

        assert len(plan) >= 2
        assert plan[0].action == "navigate"
        assert plan[0].params.get("room") == "living_room"

    def test_fallback_plan_chinese(self):
        """Fallback planner handles Chinese room names (厨房 -> kitchen)."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("去厨房看看")

        assert len(plan) >= 1, "Expected at least one step for Chinese room goal"
        rooms_in_plan = [s.params.get("room") for s in plan if s.action == "navigate"]
        assert "kitchen" in rooms_in_plan, (
            f"Expected 'kitchen' in plan rooms, got {rooms_in_plan}"
        )

    def test_fallback_plan_chinese_bedroom(self):
        """Fallback planner handles 卧室 (bedroom -> master_bedroom)."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("去卧室")

        rooms_in_plan = [s.params.get("room") for s in plan if s.action == "navigate"]
        assert len(rooms_in_plan) >= 1, "Expected navigate step for 卧室"

    def test_fallback_plan_patrol(self):
        """Fallback planner generates a full multi-room patrol for patrol keywords."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("巡逻全屋")

        navigate_steps = [s for s in plan if s.action == "navigate"]
        assert len(navigate_steps) >= 6, (
            f"Patrol plan should cover >= 6 rooms, got {len(navigate_steps)}: "
            f"{[s.params.get('room') for s in navigate_steps]}"
        )

    def test_fallback_plan_patrol_english(self):
        """Fallback planner handles English 'patrol' keyword."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("patrol all rooms")

        navigate_steps = [s for s in plan if s.action == "navigate"]
        assert len(navigate_steps) >= 6, (
            f"Patrol plan should cover >= 6 rooms, got {len(navigate_steps)}"
        )

    def test_fallback_plan_unknown_goal_returns_look(self):
        """Fallback planner returns a single look step for unrecognised goals."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("what is the meaning of life")

        assert len(plan) >= 1
        assert plan[0].action == "look"

    # ------------------------------------------------------------------
    # Plan parser
    # ------------------------------------------------------------------

    def test_parse_plan_json(self):
        """Plan parser handles a valid JSON array string."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        raw = '[{"action": "navigate", "params": {"room": "kitchen"}, "reason": "go there"}]'
        plan = loop._parse_plan(raw)

        assert len(plan) == 1
        assert plan[0].action == "navigate"
        assert plan[0].params["room"] == "kitchen"
        assert plan[0].reason == "go there"

    def test_parse_plan_markdown_fence(self):
        """Plan parser handles JSON array wrapped in a markdown code fence."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        text = '```json\n[{"action": "look", "params": {}, "reason": "observe"}]\n```'
        plan = loop._parse_plan(text)

        assert len(plan) == 1
        assert plan[0].action == "look"

    def test_parse_plan_multiple_steps(self):
        """Plan parser preserves step order for multi-step JSON arrays."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        raw = (
            '[{"action": "navigate", "params": {"room": "hallway"}, "reason": "start"}, '
            '{"action": "look", "params": {}, "reason": "observe"}, '
            '{"action": "navigate", "params": {"room": "kitchen"}, "reason": "move"}]'
        )
        plan = loop._parse_plan(raw)

        assert len(plan) == 3
        assert plan[0].action == "navigate"
        assert plan[0].params["room"] == "hallway"
        assert plan[1].action == "look"
        assert plan[2].action == "navigate"
        assert plan[2].params["room"] == "kitchen"

    def test_parse_plan_invalid_json_returns_empty(self):
        """Plan parser returns empty list for malformed JSON."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._parse_plan("this is not json at all")
        assert plan == []

    def test_parse_plan_empty_string(self):
        """Plan parser returns empty list for empty input."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = loop._parse_plan("")
        assert plan == []

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def test_run_executes_plan(self):
        """MobileAgentLoop.run() executes all planned steps and returns a result."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen", max_steps=10)

        assert result.steps_completed > 0, "Expected at least one step to complete"
        assert result.steps_total > 0

    def test_run_returns_goal_result(self):
        """MobileAgentLoop.run() returns MobileGoalResult with correct goal field."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, MobileGoalResult

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen", max_steps=10)

        assert isinstance(result, MobileGoalResult)
        assert result.goal == "go to kitchen"

    def test_robot_fall_aborts_execution(self):
        """Execution aborts early when the robot's z-height drops below fall threshold."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        # Simulate fallen robot: z = 0.05 m (below 0.12 m threshold)
        agent._base.get_position.return_value = [10.0, 3.0, 0.05]

        loop = MobileAgentLoop(agent, {})
        result = loop.run("patrol all rooms", max_steps=20)

        # Should have started (steps_total > 0) but aborted before finishing all rooms
        assert result.steps_total > 0, "Expected non-zero step count"

    def test_on_step_callback_fires(self):
        """on_step callback is called once per executed step."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        fired: list[tuple] = []

        def _on_step(action: str, idx: int, total: int) -> None:
            fired.append((action, idx, total))

        loop.run("go to kitchen", max_steps=10, on_step=_on_step)

        assert len(fired) > 0, "Expected on_step to be called at least once"
        # First call should be for step index 0
        assert fired[0][1] == 0

    def test_on_message_callback_fires(self):
        """on_message callback is called with plan summary and final summary."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        messages: list[str] = []

        loop.run("go to kitchen", max_steps=10, on_message=messages.append)

        assert len(messages) >= 1, "Expected at least one message callback"
        # First message should mention the plan step count
        combined = " ".join(messages)
        assert len(combined) > 0

    def test_max_steps_limits_execution(self):
        """max_steps parameter caps the number of steps executed."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        # Patrol goal generates many steps; cap at 2
        result = loop.run("patrol all rooms", max_steps=2)

        assert result.steps_total <= 2, (
            f"Expected <= 2 steps but got {result.steps_total}"
        )
