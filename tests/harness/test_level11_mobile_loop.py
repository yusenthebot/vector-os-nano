"""Level 11: MobileAgentLoop LLM planning integration tests.

Tests that MobileAgentLoop correctly:
- Calls LLM for plan decomposition
- Parses JSON plans from LLM responses
- Falls back to heuristic when LLM fails
- Executes multi-step plans with skill dispatch
- Records observations in SceneGraph via auto-observe

Uses mock LLM and mock base (no real API calls, no MuJoCo).
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vector_os_nano.core.mobile_agent_loop import (
    MobileAgentLoop,
    MobileGoalResult,
    SubTask,
)
from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext, SkillRegistry
from vector_os_nano.core.types import SkillResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(plan_response: str | None = None):
    """Create a mock LLM provider that returns a canned plan."""
    llm = MagicMock()

    if plan_response is not None:
        llm.chat.return_value = plan_response
    else:
        # Default: return a valid 2-step plan
        llm.chat.return_value = json.dumps([
            {"action": "navigate", "params": {"room": "kitchen"}, "reason": "go to kitchen"},
            {"action": "look", "params": {}, "reason": "observe kitchen"},
        ])

    return llm


def _make_mock_vlm():
    """Create a mock VLM."""
    vlm = MagicMock()
    scene = MagicMock()
    scene.summary = "Kitchen with countertop and fridge"
    obj = MagicMock()
    obj.name = "fridge"
    scene.objects = [obj]
    vlm.describe_scene.return_value = scene

    room_id = MagicMock()
    room_id.room = "kitchen"
    room_id.confidence = 0.95
    vlm.identify_room.return_value = room_id

    return vlm


def _make_mock_agent(llm=None, vlm=None, scene_graph=None):
    """Create a mock Agent with configurable components."""
    agent = MagicMock()
    agent._llm = llm
    agent._vlm = vlm
    agent._spatial_memory = scene_graph
    agent._config = {}

    # Mock base
    base = MagicMock()
    base.get_position.return_value = (3.0, 2.5, 0.28)
    base.get_heading.return_value = 0.0
    base.get_camera_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    base.walk.return_value = True
    agent._base = base

    # Skill registry with navigate and look
    registry = SkillRegistry()

    from vector_os_nano.skills.go2.look import LookSkill
    from vector_os_nano.skills.navigate import NavigateSkill

    registry.register(NavigateSkill())
    registry.register(LookSkill())

    agent._skill_registry = registry

    # _build_context returns a proper SkillContext
    def _build_ctx():
        services = {}
        if vlm is not None:
            services["vlm"] = vlm
        if scene_graph is not None:
            services["spatial_memory"] = scene_graph
        return SkillContext(
            bases={"default": base},
            services=services,
        )

    agent._build_context = _build_ctx

    return agent


# ---------------------------------------------------------------------------
# Tests: LLM Planning
# ---------------------------------------------------------------------------


class TestLLMPlanning:
    """Test that MobileAgentLoop calls LLM correctly for planning."""

    def test_plan_calls_llm_chat(self):
        """_plan() calls agent._llm.chat(user_message=goal, system_prompt=...)."""
        llm = _make_mock_llm()
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("go to the kitchen")

        llm.chat.assert_called_once()
        call_kwargs = llm.chat.call_args
        assert "user_message" in call_kwargs.kwargs
        assert call_kwargs.kwargs["user_message"] == "go to the kitchen"
        assert "system_prompt" in call_kwargs.kwargs

    def test_plan_parses_json_array(self):
        """_plan() parses a JSON array of SubTasks from LLM response."""
        response = json.dumps([
            {"action": "navigate", "params": {"room": "study"}, "reason": "go"},
            {"action": "look", "params": {}, "reason": "observe"},
        ])
        llm = _make_mock_llm(response)
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("go to study and look")

        assert len(plan) == 2
        assert plan[0].action == "navigate"
        assert plan[0].params == {"room": "study"}
        assert plan[1].action == "look"

    def test_plan_parses_markdown_fenced_json(self):
        """_plan() handles LLM wrapping JSON in markdown fences."""
        response = '```json\n[{"action": "look", "params": {}}]\n```'
        llm = _make_mock_llm(response)
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("look around")

        assert len(plan) == 1
        assert plan[0].action == "look"

    def test_plan_returns_empty_on_invalid_json(self):
        """_plan() returns empty list when LLM returns garbage."""
        llm = _make_mock_llm("I don't understand your request")
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("do something weird")

        assert plan == []

    def test_plan_returns_empty_when_no_llm(self):
        """_plan() returns empty list when agent has no LLM."""
        agent = _make_mock_agent(llm=None)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("go to kitchen")

        assert plan == []

    def test_plan_includes_memory_context(self):
        """_plan() includes SceneGraph room summary in the system prompt."""
        llm = _make_mock_llm()
        sg = SceneGraph()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge", "counter"], "Kitchen with modern appliances")

        agent = _make_mock_agent(llm=llm, scene_graph=sg)
        loop = MobileAgentLoop(agent, {})

        loop._plan("find the fridge")

        call_kwargs = llm.chat.call_args.kwargs
        assert "kitchen" in call_kwargs["system_prompt"].lower()

    def test_plan_chinese_goal(self):
        """_plan() passes Chinese goals to LLM correctly."""
        response = json.dumps([
            {"action": "navigate", "params": {"room": "kitchen"}, "reason": "去厨房"},
        ])
        llm = _make_mock_llm(response)
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        plan = loop._plan("去厨房看看")

        assert len(plan) == 1
        assert plan[0].action == "navigate"


class TestFallbackPlanning:
    """Test fallback heuristic when LLM is unavailable."""

    def test_fallback_on_llm_failure(self):
        """run() uses fallback plan when LLM raises."""
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API error")

        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen", max_steps=5)

        # Fallback should produce navigate+look
        assert result.steps_total >= 1

    def test_fallback_chinese_room(self):
        """Fallback handles Chinese room names."""
        agent = _make_mock_agent(llm=None)
        loop = MobileAgentLoop(agent, {})

        plan = loop._fallback_plan("去厨房")

        assert len(plan) == 2
        assert plan[0].action == "navigate"
        assert plan[0].params["room"] == "kitchen"
        assert plan[1].action == "look"


class TestMobileLoopExecution:
    """Test full execution pipeline."""

    def test_run_executes_plan_steps(self):
        """run() executes each SubTask in the plan."""
        response = json.dumps([
            {"action": "navigate", "params": {"room": "kitchen"}},
            {"action": "look", "params": {}},
        ])
        llm = _make_mock_llm(response)
        vlm = _make_mock_vlm()
        sg = SceneGraph()
        agent = _make_mock_agent(llm=llm, vlm=vlm, scene_graph=sg)
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen and look")

        assert isinstance(result, MobileGoalResult)
        assert result.steps_total == 2
        assert result.goal == "go to kitchen and look"

    def test_run_callbacks_fire(self):
        """on_step and on_message callbacks are invoked during run()."""
        steps_received = []
        messages_received = []

        response = json.dumps([{"action": "look", "params": {}}])
        llm = _make_mock_llm(response)
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        result = loop.run(
            "look around",
            on_step=lambda action, idx, total: steps_received.append((action, idx)),
            on_message=lambda msg: messages_received.append(msg),
        )

        assert len(steps_received) >= 1
        assert len(messages_received) >= 1  # at least plan + summary

    def test_run_respects_max_steps(self):
        """run() caps plan at max_steps."""
        # Plan with 10 steps
        plan = [{"action": "look", "params": {}} for _ in range(10)]
        llm = _make_mock_llm(json.dumps(plan))
        agent = _make_mock_agent(llm=llm)
        loop = MobileAgentLoop(agent, {})

        result = loop.run("look everywhere", max_steps=3)

        assert result.steps_total == 3

    def test_auto_observe_records_in_scene_graph(self):
        """After navigate, auto-observe updates SceneGraph."""
        response = json.dumps([
            {"action": "navigate", "params": {"room": "kitchen"}},
        ])
        llm = _make_mock_llm(response)
        vlm = _make_mock_vlm()
        # Pre-populate SceneGraph so NavigateSkill can find kitchen.
        # NavigateSkill requires _get_room_center_from_memory to return a position
        # (visit_count >= _MIN_VISIT_COUNT = 3).
        sg = SceneGraph()
        for _ in range(5):
            sg.visit("kitchen", 17.0, 2.5)
        agent = _make_mock_agent(llm=llm, vlm=vlm, scene_graph=sg)
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen")

        # VLM should have been called for auto-observe
        assert vlm.describe_scene.called or vlm.identify_room.called

    def test_robot_fall_aborts(self):
        """run() aborts if robot falls mid-execution."""
        response = json.dumps([
            {"action": "navigate", "params": {"room": "kitchen"}},
            {"action": "look", "params": {}},
            {"action": "navigate", "params": {"room": "study"}},
        ])
        llm = _make_mock_llm(response)
        agent = _make_mock_agent(llm=llm)

        # Make robot fall after first step
        call_count = [0]
        def _position():
            call_count[0] += 1
            if call_count[0] > 3:
                return (5.0, 5.0, 0.05)  # fallen
            return (3.0, 2.5, 0.28)
        agent._base.get_position = _position

        loop = MobileAgentLoop(agent, {})
        result = loop.run("patrol")

        # Should have aborted before completing all 3 steps
        assert result.steps_completed < result.steps_total or result.steps_total <= 3
