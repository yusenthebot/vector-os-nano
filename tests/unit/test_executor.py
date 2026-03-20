"""Unit tests for vector_os.core.executor — TDD RED phase."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def world_model():
    from vector_os.core.world_model import WorldModel
    return WorldModel()


def make_skill(name: str, success: bool = True, preconditions: list[str] | None = None,
               postconditions: list[str] | None = None) -> Any:
    """Create a mock Skill that returns success/failure."""
    from vector_os.core.types import SkillResult
    skill = MagicMock()
    skill.name = name
    skill.description = f"Mock {name} skill"
    skill.parameters = {}
    skill.preconditions = preconditions or []
    skill.postconditions = postconditions or []
    skill.effects = {}
    skill.execute.return_value = SkillResult(success=success)
    return skill


def make_registry(*skills):
    from vector_os.core.skill import SkillRegistry
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_context(world_model):
    from vector_os.core.skill import SkillContext
    return SkillContext(
        arm=MagicMock(),
        gripper=MagicMock(),
        perception=None,
        world_model=world_model,
        calibration=None,
    )


def make_plan(steps=None, goal="test"):
    from vector_os.core.types import TaskPlan
    return TaskPlan(goal=goal, steps=steps or [])


def make_step(step_id, skill_name, params=None, depends_on=None,
              preconditions=None, postconditions=None):
    from vector_os.core.types import TaskStep
    return TaskStep(
        step_id=step_id,
        skill_name=skill_name,
        parameters=params or {},
        depends_on=depends_on or [],
        preconditions=preconditions or [],
        postconditions=postconditions or [],
    )


class TestEmptyPlan:
    def test_empty_plan_returns_success(self, world_model):
        from vector_os.core.executor import TaskExecutor
        executor = TaskExecutor()
        plan = make_plan()
        registry = make_registry()
        context = make_context(world_model)
        result = executor.execute(plan, registry, context)
        assert result.success is True
        assert result.steps_completed == 0
        assert result.steps_total == 0

    def test_empty_plan_has_empty_trace(self, world_model):
        from vector_os.core.executor import TaskExecutor
        executor = TaskExecutor()
        result = executor.execute(make_plan(), make_registry(), make_context(world_model))
        assert result.trace == []


class TestSingleStep:
    def test_single_step_success(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("home")
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "home")
        plan = make_plan([step])

        executor = TaskExecutor()
        result = executor.execute(plan, registry, context)

        assert result.success is True
        assert result.steps_completed == 1
        assert result.steps_total == 1

    def test_single_step_calls_skill_execute(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("home")
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "home", params={"speed": 1.0})
        plan = make_plan([step])

        TaskExecutor().execute(plan, registry, context)
        skill.execute.assert_called_once()

    def test_single_step_failure_propagates(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("pick", success=False)
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "pick")
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is False
        assert result.steps_completed == 0

    def test_unknown_skill_fails(self, world_model):
        from vector_os.core.executor import TaskExecutor
        registry = make_registry()  # Empty registry
        context = make_context(world_model)
        step = make_step("s0", "nonexistent_skill")
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is False
        assert result.failure_reason is not None


class TestPreconditions:
    def test_unsatisfied_precondition_fails(self, world_model):
        from vector_os.core.executor import TaskExecutor
        # gripper_holding_any is False by default
        skill = make_skill("place", preconditions=["gripper_holding_any"])
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "place", preconditions=["gripper_holding_any"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is False
        assert result.steps_completed == 0

    def test_satisfied_precondition_passes(self, world_model):
        from vector_os.core.executor import TaskExecutor
        world_model.update_robot_state(held_object="obj_001")
        skill = make_skill("place", preconditions=["gripper_holding_any"])
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "place", preconditions=["gripper_holding_any"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is True

    def test_gripper_empty_precondition(self, world_model):
        from vector_os.core.executor import TaskExecutor
        world_model.update_robot_state(gripper_state="open", held_object=None)
        skill = make_skill("pick", preconditions=["gripper_empty"])
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "pick", preconditions=["gripper_empty"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is True


class TestPostconditions:
    def test_postcondition_failure_on_success_skill(self, world_model):
        from vector_os.core.executor import TaskExecutor
        # Skill succeeds but world model doesn't have the postcondition satisfied
        skill = make_skill("pick", success=True,
                           postconditions=["gripper_holding_any"])
        registry = make_registry(skill)
        context = make_context(world_model)
        # postcondition: gripper_holding_any — but we don't update world model
        step = make_step("s0", "pick", postconditions=["gripper_holding_any"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        # Postcondition not satisfied → execution failure
        assert result.success is False

    def test_postcondition_success_when_world_updated(self, world_model):
        from vector_os.core.executor import TaskExecutor
        from vector_os.core.types import SkillResult

        # Skill that actually updates world model
        skill = MagicMock()
        skill.name = "pick"
        skill.description = "pick skill"
        skill.parameters = {}
        skill.preconditions = []
        skill.postconditions = ["gripper_holding_any"]
        skill.effects = {}

        def execute_and_update(params, context):
            context.world_model.update_robot_state(held_object="obj_001")
            return SkillResult(success=True)

        skill.execute.side_effect = execute_and_update
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "pick", postconditions=["gripper_holding_any"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is True


class TestDependencyOrder:
    def test_steps_execute_in_dependency_order(self, world_model):
        from vector_os.core.executor import TaskExecutor
        execution_order = []

        def make_tracking_skill(name):
            from vector_os.core.types import SkillResult
            skill = MagicMock()
            skill.name = name
            skill.description = f"Mock {name}"
            skill.parameters = {}
            skill.preconditions = []
            skill.postconditions = []
            skill.effects = {}

            def execute(params, context, _name=name):
                execution_order.append(_name)
                return SkillResult(success=True)

            skill.execute.side_effect = execute
            return skill

        s_detect = make_tracking_skill("detect")
        s_pick = make_tracking_skill("pick")
        s_place = make_tracking_skill("place")

        registry = make_registry(s_detect, s_pick, s_place)
        context = make_context(world_model)

        steps = [
            make_step("s2", "place", depends_on=["s1"]),
            make_step("s0", "detect"),
            make_step("s1", "pick", depends_on=["s0"]),
        ]
        plan = make_plan(steps)

        result = TaskExecutor().execute(plan, registry, context)
        assert result.success is True
        assert execution_order.index("detect") < execution_order.index("pick")
        assert execution_order.index("pick") < execution_order.index("place")

    def test_all_steps_completed_on_full_success(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("home")
        registry = make_registry(skill)
        context = make_context(world_model)
        steps = [
            make_step("s0", "home"),
            make_step("s1", "home", depends_on=["s0"]),
            make_step("s2", "home", depends_on=["s1"]),
        ]
        plan = make_plan(steps)
        result = TaskExecutor().execute(plan, registry, context)
        assert result.steps_completed == 3
        assert result.steps_total == 3


class TestExecutionTrace:
    def test_trace_records_step_results(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("home")
        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "home")])

        result = TaskExecutor().execute(plan, registry, context)
        assert len(result.trace) == 1
        trace = result.trace[0]
        assert trace.step_id == "s0"
        assert trace.skill_name == "home"
        assert trace.status == "success"

    def test_trace_records_duration(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("home")
        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "home")])

        result = TaskExecutor().execute(plan, registry, context)
        assert result.trace[0].duration_sec >= 0.0

    def test_trace_records_failure_reason(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill = make_skill("pick", success=False)
        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "pick")])

        result = TaskExecutor().execute(plan, registry, context)
        assert len(result.trace) == 1
        trace = result.trace[0]
        assert trace.status != "success"

    def test_trace_stops_at_failed_step(self, world_model):
        from vector_os.core.executor import TaskExecutor
        skill_fail = make_skill("pick", success=False)
        skill_ok = make_skill("place")
        registry = make_registry(skill_fail, skill_ok)
        context = make_context(world_model)
        steps = [
            make_step("s0", "pick"),
            make_step("s1", "place", depends_on=["s0"]),
        ]
        plan = make_plan(steps)

        result = TaskExecutor().execute(plan, registry, context)
        # Execution stopped at s0 — s1 should not have been attempted
        assert result.success is False
        assert result.steps_completed == 0
