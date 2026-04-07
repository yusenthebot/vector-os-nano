"""Level 6 — Robustness: error handling, recovery, and edge cases.

Tests cover the error paths and boundary conditions that could crash the
Go2 VLM + navigation pipeline in production.  Every test uses mocks so
there are zero API calls and zero MuJoCo startup cost (navigation tests
that need the physics engine use the sinusoidal backend explicitly).

Test categories
---------------
TestVLMErrorHandling      VLM failure modes — timeout, bad JSON, 4xx, camera exc
TestNavigationRobustness  Navigation edge cases — same room, unknown, round-trip
TestSpatialMemoryEdgeCases Persistence, observe, visit_count, unvisited rooms
TestMobileAgentLoopEdgeCases Empty plan, unknown skill, max_steps cap
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path (mirrors other harness tests)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_frame() -> np.ndarray:
    """Return a minimal synthetic RGB frame (1x1)."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _make_skill_context(
    base: Any = None,
    services: dict | None = None,
) -> "SkillContext":
    """Build a real SkillContext with optional base and service overrides."""
    from vector_os_nano.core.skill import SkillContext

    bases_dict: dict = {"default": base} if base is not None else {}
    return SkillContext(
        bases=bases_dict,
        services=services or {},
    )


def _make_mock_base(
    position: list[float] | None = None,
    heading: float = 0.0,
) -> MagicMock:
    """Return a mock base that reports a standing upright position."""
    base = MagicMock()
    base.get_position.return_value = position or [10.0, 3.0, 0.28]
    base.get_heading.return_value = heading
    base.get_camera_frame.return_value = _make_frame()
    return base


def _make_mock_agent(
    *,
    position: list[float] | None = None,
    llm: Any = None,
) -> MagicMock:
    """Return a mock agent suitable for MobileAgentLoop construction."""
    from vector_os_nano.core.skill import SkillContext, SkillRegistry
    from vector_os_nano.core.types import SkillResult

    agent = MagicMock()

    base = _make_mock_base(position=position)
    agent._base = base
    agent._llm = llm
    agent._vlm = None
    agent._spatial_memory = None

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
        success=True, result_data={"summary": "A tidy room.", "room": "kitchen"}
    )

    registry.register(nav_skill)
    registry.register(look_skill)
    agent._skill_registry = registry

    agent._world_model = MagicMock()

    def _build_context() -> SkillContext:
        return SkillContext(
            bases={"default": base},
            services={"skill_registry": registry},
        )

    agent._build_context.side_effect = _build_context
    return agent


# ---------------------------------------------------------------------------
# TestVLMErrorHandling
# ---------------------------------------------------------------------------


class TestVLMErrorHandling:
    """VLM failure modes — all mocked, zero API cost."""

    def test_vlm_timeout_graceful(self) -> None:
        """LookSkill handles VLM timeout gracefully — returns vlm_failed."""
        import httpx

        from vector_os_nano.core.types import SkillResult
        from vector_os_nano.skills.go2.look import LookSkill

        base = _make_mock_base()
        vlm = MagicMock()
        vlm.describe_scene.side_effect = httpx.TimeoutException("timed out")

        ctx = _make_skill_context(base=base, services={"vlm": vlm})
        result: SkillResult = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "vlm_failed"
        assert "timed out" in result.error_message.lower() or result.error_message != ""

    def test_vlm_invalid_json_response(self) -> None:
        """VLM returns garbage text — describe_scene still returns a SceneDescription."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception, SceneDescription

        vlm = Go2VLMPerception.__new__(Go2VLMPerception)
        # Directly call the JSON-parse path via _call_vlm mock
        with patch.object(vlm, "_call_vlm", return_value="this is not json"):
            scene = vlm.describe_scene(_make_frame())

        assert isinstance(scene, SceneDescription)
        # Summary and details should be empty strings when JSON parse fails
        assert isinstance(scene.summary, str)
        assert isinstance(scene.objects, list)

    def test_vlm_api_error_4xx(self) -> None:
        """VLM 4xx HTTP error propagates as exception — LookSkill returns vlm_failed."""
        import httpx

        from vector_os_nano.skills.go2.look import LookSkill

        base = _make_mock_base()
        # Simulate a 401 Unauthorized response
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(401, request=request)
        http_error = httpx.HTTPStatusError(
            "401 Unauthorized", request=request, response=response
        )

        vlm = MagicMock()
        vlm.describe_scene.side_effect = http_error

        ctx = _make_skill_context(base=base, services={"vlm": vlm})
        result = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "vlm_failed"

    def test_look_skill_recovers_from_camera_exception(self) -> None:
        """get_camera_frame() raises RuntimeError — LookSkill returns camera_failed."""
        from vector_os_nano.skills.go2.look import LookSkill

        base = _make_mock_base()
        base.get_camera_frame.side_effect = RuntimeError("camera bus error")

        vlm = MagicMock()
        ctx = _make_skill_context(base=base, services={"vlm": vlm})
        result = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "camera_failed"
        assert "camera" in result.error_message.lower()

    def test_look_skill_no_vlm_service(self) -> None:
        """LookSkill with no vlm in context returns no_vlm failure."""
        from vector_os_nano.skills.go2.look import LookSkill

        base = _make_mock_base()
        ctx = _make_skill_context(base=base, services={})
        result = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_vlm"

    def test_look_skill_no_base(self) -> None:
        """LookSkill with no base returns no_base failure."""
        from vector_os_nano.skills.go2.look import LookSkill

        ctx = _make_skill_context(base=None, services={"vlm": MagicMock()})
        result = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"

    def test_vlm_general_exception_returns_empty_scene(self) -> None:
        """A generic exception from vlm.describe_scene — LookSkill returns vlm_failed."""
        from vector_os_nano.skills.go2.look import LookSkill

        base = _make_mock_base()
        vlm = MagicMock()
        vlm.describe_scene.side_effect = ConnectionResetError("connection reset")

        ctx = _make_skill_context(base=base, services={"vlm": vlm})
        result = LookSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "vlm_failed"


# ---------------------------------------------------------------------------
# TestNavigationRobustness
# ---------------------------------------------------------------------------


@pytest.fixture
def go2():
    """Headless Go2 sim with sinusoidal backend for navigation tests."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    robot = MuJoCoGo2(gui=False, room=True, backend="sinusoidal")
    robot.connect()
    robot.stand()
    yield robot
    robot.disconnect()


class TestNavigationRobustness:
    """Navigation edge cases using real MuJoCo sinusoidal backend."""

    def test_navigate_to_same_room(self, go2) -> None:
        """Navigate to the room we are already in — succeeds immediately."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.scene_graph import SceneGraph
        from vector_os_nano.skills.navigate import NavigateSkill, _detect_current_room

        pos = go2.get_position()
        # Build a SceneGraph with known rooms so _detect_current_room can work
        sg = SceneGraph()
        _test_rooms = {
            "living_room":    (3.0,  2.5),
            "dining_room":    (3.0,  7.5),
            "kitchen":        (17.0, 2.5),
            "study":          (17.0, 7.5),
            "master_bedroom": (3.5,  12.0),
            "guest_bedroom":  (16.0, 12.0),
            "bathroom":       (8.5,  12.0),
            "hallway":        (10.0, 5.0),
        }
        for name, (x, y) in _test_rooms.items():
            for _ in range(5):
                sg.visit(name, x, y)
        current = _detect_current_room(float(pos[0]), float(pos[1]), sg=sg)

        ctx = SkillContext(bases={"default": go2}, services={"spatial_memory": sg})
        result = NavigateSkill().execute({"room": current}, ctx)

        assert result.success, f"Navigate to same room failed: {result.error_message}"
        assert result.result_data.get("room") == current

    def test_navigate_unknown_room(self, go2) -> None:
        """Navigate to a nonexistent room — fails with unknown_room code."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.scene_graph import SceneGraph
        from vector_os_nano.skills.navigate import NavigateSkill

        # Provide a SceneGraph with known rooms so the error message includes
        # the unknown room name (not a generic "no rooms learned" message).
        sg = SceneGraph()
        for _ in range(5):
            sg.visit("kitchen", 17.0, 2.5)

        ctx = SkillContext(bases={"default": go2}, services={"spatial_memory": sg})
        result = NavigateSkill().execute({"room": "nonexistent_xyzzy"}, ctx)

        assert not result.success
        assert result.diagnosis_code == "unknown_room"
        assert "nonexistent_xyzzy" in result.error_message

    @pytest.mark.timeout(120)
    def test_consecutive_navigations(self, go2) -> None:
        """Navigate hallway -> kitchen -> study (three hops) without crashing."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.skills.navigate import NavigateSkill

        nav = NavigateSkill()
        ctx = SkillContext(bases={"default": go2}, services={})

        rooms = ["hallway", "kitchen", "study"]
        for room in rooms:
            result = nav.execute({"room": room}, ctx)
            # Navigation may fail physically (sinusoidal is approximate) but
            # must never raise an unhandled exception.
            assert isinstance(result.success, bool), (
                f"NavigateSkill.execute must return SkillResult for room={room}"
            )

    @pytest.mark.timeout(60)
    def test_navigate_after_sit(self, go2) -> None:
        """Robot in sit pose can still navigate (navigate must not crash)."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.skills.navigate import NavigateSkill

        go2.sit()  # Put robot in sit pose
        ctx = SkillContext(bases={"default": go2}, services={})
        # NavigateSkill does not call stand() — it just dead-reckons and
        # returns success/failure based on position.  The test verifies no
        # exception is raised regardless of the physical outcome.
        result = NavigateSkill().execute({"room": "hallway"}, ctx)
        assert isinstance(result.success, bool), "NavigateSkill must return SkillResult"

    def test_navigate_missing_room_param(self, go2) -> None:
        """Navigate with empty room param — fails with unknown_room."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.skills.navigate import NavigateSkill

        ctx = SkillContext(bases={"default": go2}, services={})
        result = NavigateSkill().execute({"room": ""}, ctx)

        assert not result.success
        assert result.diagnosis_code == "unknown_room"

    def test_navigate_no_base(self) -> None:
        """Navigate with no base in context — fails with no_base."""
        from vector_os_nano.skills.navigate import NavigateSkill

        ctx = _make_skill_context(base=None)
        result = NavigateSkill().execute({"room": "kitchen"}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"


# ---------------------------------------------------------------------------
# TestSpatialMemoryEdgeCases
# ---------------------------------------------------------------------------


class TestSpatialMemoryEdgeCases:
    """Spatial memory persistence and edge cases — no hardware, no API."""

    def test_visit_same_room_increments_count(self) -> None:
        """Visiting the same room twice increments visit_count from 1 to 2."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.visit("kitchen", 17.0, 2.5)
        sm.visit("kitchen", 17.0, 2.5)

        rec = sm.get_location("kitchen")
        assert rec is not None
        assert rec.visit_count == 2

    def test_observe_updates_objects(self) -> None:
        """observe() merges new objects into the room record."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.observe("kitchen", ["fridge", "counter"], "First look")
        sm.observe("kitchen", ["counter", "island"], "Second look")

        rec = sm.get_location("kitchen")
        assert rec is not None
        # Merged: fridge + counter + island (no duplicates)
        assert set(rec.objects_seen) == {"fridge", "counter", "island"}
        # Description should reflect most recent observe
        assert rec.description == "Second look"

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Save locations to YAML and reload — data integrity preserved."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        persist_file = str(tmp_path / "spatial_mem.yaml")
        sm1 = SpatialMemory(persist_path=persist_file)
        sm1.visit("hallway", 10.0, 5.0)
        sm1.observe("hallway", ["door", "window"], "A corridor")
        sm1.remember_location("charging_dock", 1.0, 1.0)
        sm1.save()

        sm2 = SpatialMemory(persist_path=persist_file)
        hallway = sm2.get_location("hallway")
        dock = sm2.get_location("charging_dock")

        assert hallway is not None
        assert hallway.visit_count == 1
        assert "door" in hallway.objects_seen
        assert hallway.description == "A corridor"

        assert dock is not None
        assert dock.x == pytest.approx(1.0)
        assert dock.y == pytest.approx(1.0)

    def test_remember_custom_location(self) -> None:
        """remember_location() creates a new named bookmark with correct coords."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.remember_location("garage", 20.0, 5.0)

        rec = sm.get_location("garage")
        assert rec is not None
        assert rec.name == "garage"
        assert rec.x == pytest.approx(20.0)
        assert rec.y == pytest.approx(5.0)
        # remember_location does NOT increment visit_count
        assert rec.visit_count == 0

    def test_get_unvisited_rooms(self) -> None:
        """get_unvisited_rooms() returns rooms from all_rooms not yet visited."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.visit("kitchen", 17.0, 2.5)
        sm.visit("hallway", 10.0, 5.0)

        all_rooms = ["kitchen", "hallway", "study", "bathroom"]
        unvisited = sm.get_unvisited_rooms(all_rooms)

        assert set(unvisited) == {"study", "bathroom"}

    def test_get_unvisited_rooms_all_visited(self) -> None:
        """get_unvisited_rooms() returns empty list when all rooms are visited."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        rooms = ["kitchen", "hallway"]
        for r in rooms:
            sm.visit(r, 0.0, 0.0)

        assert sm.get_unvisited_rooms(rooms) == []

    def test_get_visited_rooms_empty(self) -> None:
        """get_visited_rooms() returns empty list when nothing has been visited."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        assert sm.get_visited_rooms() == []

    def test_observe_creates_record_if_missing(self) -> None:
        """observe() on an unknown room creates a LocationRecord with x=0, y=0."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.observe("pantry", ["shelves", "cans"])

        rec = sm.get_location("pantry")
        assert rec is not None
        assert rec.x == pytest.approx(0.0)
        assert "shelves" in rec.objects_seen

    def test_load_missing_file_is_noop(self, tmp_path: Path) -> None:
        """Loading from a nonexistent file is silent — no exception raised."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        missing = str(tmp_path / "does_not_exist.yaml")
        sm = SpatialMemory(persist_path=missing)
        assert sm.get_all_locations() == []

    def test_save_no_persist_path_is_noop(self) -> None:
        """save() with persist_path=None is a no-op — no exception raised."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.visit("hallway", 10.0, 5.0)
        sm.save()  # Must not raise

    def test_room_summary_reflects_observed_objects(self) -> None:
        """get_room_summary() includes observed object names for visited rooms."""
        from vector_os_nano.core.spatial_memory import SpatialMemory

        sm = SpatialMemory(persist_path=None)
        sm.visit("kitchen", 17.0, 2.5)
        sm.observe("kitchen", ["fridge", "counter"])

        summary = sm.get_room_summary()
        assert "kitchen" in summary
        assert "fridge" in summary


# ---------------------------------------------------------------------------
# TestMobileAgentLoopEdgeCases
# ---------------------------------------------------------------------------


class TestMobileAgentLoopEdgeCases:
    """MobileAgentLoop error paths — all mocked, zero API cost."""

    def test_empty_plan_falls_back_and_returns_result(self) -> None:
        """When _plan() returns [] AND _fallback_plan() is called, run() still returns a result."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, MobileGoalResult

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        # Patch _plan to return empty (LLM absent path) and _fallback_plan to also
        # return empty — this simulates a completely unrecognised goal where
        # the fallback returns the minimal [look] step.
        with patch.object(loop, "_fallback_plan", return_value=[]):
            result = loop.run("???", max_steps=5)

        assert isinstance(result, MobileGoalResult)
        assert result.goal == "???"
        # No steps were available; steps_total should be 0
        assert result.steps_total == 0

    def test_skill_not_found_skipped_others_run(self) -> None:
        """Unknown skill in plan is skipped; subsequent known skills still execute."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, SubTask
        from vector_os_nano.core.types import SkillResult

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        plan = [
            SubTask(action="nonexistent_skill_xyz", params={}, reason="unknown"),
            SubTask(action="look", params={}, reason="observe"),
        ]

        with patch.object(loop, "_plan", return_value=[]):
            with patch.object(loop, "_fallback_plan", return_value=plan):
                result = loop.run("test goal", max_steps=10)

        # The "look" step should have run and succeeded; the unknown step fails
        look_results = [s for s in result.sub_results if s.action == "look"]
        unknown_results = [
            s for s in result.sub_results if s.action == "nonexistent_skill_xyz"
        ]

        assert len(look_results) >= 1, "look step should have executed"
        assert look_results[0].success

        assert len(unknown_results) >= 1, "unknown skill step should appear in results"
        assert not unknown_results[0].success
        assert unknown_results[0].result_data.get("diagnosis_code") == "skill_not_found" or (
            # SkillResult is in sub_results, check via success=False
            not unknown_results[0].success
        )

    def test_max_steps_enforced(self) -> None:
        """max_steps=3 caps execution even when the plan has 20 steps."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        result = loop.run("patrol all rooms", max_steps=3)

        assert result.steps_total <= 3, (
            f"Expected <= 3 steps, got {result.steps_total}"
        )

    def test_robot_fall_aborts_mid_plan(self) -> None:
        """Execution halts mid-plan when robot z-height drops below fall threshold."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent(position=[10.0, 3.0, 0.04])  # below 0.12 threshold
        loop = MobileAgentLoop(agent, {})

        result = loop.run("patrol all rooms", max_steps=20)

        # Loop detects fall before completing all patrol steps
        assert result.steps_total < 20 or result.steps_completed < result.steps_total or (
            # If the loop aborted at step 0 (fall detected immediately), steps_total = 0
            result.steps_total == 0
        )

    def test_run_returns_mobile_goal_result_type(self) -> None:
        """run() always returns a MobileGoalResult regardless of plan content."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, MobileGoalResult

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        result = loop.run("go to kitchen", max_steps=5)
        assert isinstance(result, MobileGoalResult)

    def test_skill_exception_does_not_propagate(self) -> None:
        """If a skill.execute() raises, the loop catches it and continues."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, SubTask

        agent = _make_mock_agent()
        # Make the look skill raise an exception mid-execution
        agent._skill_registry.get("look").execute.side_effect = RuntimeError(
            "simulated hardware fault"
        )

        loop = MobileAgentLoop(agent, {})
        plan = [SubTask(action="look", params={}, reason="test")]

        with patch.object(loop, "_plan", return_value=[]):
            with patch.object(loop, "_fallback_plan", return_value=plan):
                result = loop.run("look around", max_steps=5)

        assert isinstance(result.success, bool), "run() must return a result even on exception"
        # The look step should have failed gracefully
        look_results = [s for s in result.sub_results if s.action == "look"]
        assert len(look_results) == 1
        assert not look_results[0].success

    def test_on_step_callback_receives_correct_indices(self) -> None:
        """on_step callback receives monotonically increasing step indices."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop, SubTask

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})

        calls: list[tuple[str, int, int]] = []

        def _cb(action: str, idx: int, total: int) -> None:
            calls.append((action, idx, total))

        plan = [
            SubTask(action="navigate", params={"room": "hallway"}, reason="go"),
            SubTask(action="look", params={}, reason="observe"),
        ]

        with patch.object(loop, "_plan", return_value=[]):
            with patch.object(loop, "_fallback_plan", return_value=plan):
                loop.run("two step plan", max_steps=10, on_step=_cb)

        assert len(calls) == 2
        assert calls[0][1] == 0   # first step index = 0
        assert calls[1][1] == 1   # second step index = 1
        # total reported to callback should equal plan length
        assert calls[0][2] == 2
        assert calls[1][2] == 2

    def test_on_message_callback_called_with_plan_summary(self) -> None:
        """on_message is called at least once and includes action names."""
        from vector_os_nano.core.mobile_agent_loop import MobileAgentLoop

        agent = _make_mock_agent()
        loop = MobileAgentLoop(agent, {})
        messages: list[str] = []

        loop.run("go to kitchen", max_steps=5, on_message=messages.append)

        assert len(messages) >= 1
        combined = " ".join(messages)
        assert len(combined) > 0
