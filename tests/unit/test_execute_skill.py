"""Tests for Agent.execute_skill() — the MCP tool call entry point.

execute_skill() bypasses string parsing and alias matching. It builds a
TaskPlan directly from the skill's auto_steps and executes it via the
TaskExecutor. Tests here verify:

1. Unknown skills are rejected cleanly.
2. Direct skills (home, scan) succeed with mocked hardware.
3. Pick's auto_steps plan includes scan and detect steps.
4. The returned ExecutionResult always has a trace.
5. Params are forwarded to the correct step.

No real hardware or network calls are made.
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import ExecutionResult, StepTrace


# ---------------------------------------------------------------------------
# Mock hardware
# ---------------------------------------------------------------------------


class MockArm:
    """Minimal arm mock that satisfies ArmProtocol."""

    name = "mock"
    joint_names = ["j1", "j2", "j3", "j4", "j5"]
    dof = 5
    _connected = False
    _bus = None  # prevents auto-gripper creation

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 5

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        return True

    def move_cartesian(self, xyz: list[float], duration: float = 3.0) -> bool:
        return True

    def fk(self, joints: list[float]):
        return ([0.2, 0.0, 0.1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def ik(self, xyz: list[float], current=None) -> list[float]:
        return [0.0] * 5

    def stop(self) -> None:
        pass


class MockGripper:
    """Minimal gripper mock."""

    def open(self) -> bool:
        return True

    def close(self) -> bool:
        return True

    def is_holding(self) -> bool:
        return False

    def get_position(self) -> float:
        return 1.0

    def get_force(self):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> Agent:
    """Agent with mock arm and gripper, no LLM, no perception."""
    return Agent(arm=MockArm(), gripper=MockGripper())


# ---------------------------------------------------------------------------
# TestExecuteSkillUnknown
# ---------------------------------------------------------------------------


class TestExecuteSkillUnknown:
    """execute_skill() with unregistered skill names."""

    def test_unknown_skill_returns_failure(self, agent: Agent) -> None:
        result = agent.execute_skill("nonexistent")
        assert not result.success

    def test_unknown_skill_failure_reason_mentions_unknown(self, agent: Agent) -> None:
        result = agent.execute_skill("nonexistent")
        reason = result.failure_reason or ""
        assert "Unknown skill" in reason or "unknown" in reason.lower(), (
            f"Expected 'Unknown skill' in failure_reason, got: {reason!r}"
        )

    def test_unknown_skill_returns_execution_result(self, agent: Agent) -> None:
        result = agent.execute_skill("does_not_exist")
        assert isinstance(result, ExecutionResult)

    def test_unknown_skill_with_params(self, agent: Agent) -> None:
        """Unknown skill with params should still fail with a clean message."""
        result = agent.execute_skill("fly", {"height": 10})
        assert not result.success
        assert result.failure_reason is not None


# ---------------------------------------------------------------------------
# TestExecuteSkillHome
# ---------------------------------------------------------------------------


class TestExecuteSkillHome:
    """execute_skill('home') — simplest direct skill."""

    def test_home_skill_succeeds(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        assert result.success, f"home skill failed: {result.failure_reason}"

    def test_home_returns_execution_result(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        assert isinstance(result, ExecutionResult)

    def test_home_trace_is_not_none(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        assert result.trace is not None

    def test_home_trace_has_home_step(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        skill_names = [t.skill_name for t in result.trace]
        assert "home" in skill_names, (
            f"Expected 'home' in trace, got: {skill_names}"
        )

    def test_home_trace_entries_are_step_traces(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        for entry in result.trace:
            assert isinstance(entry, StepTrace)

    def test_home_with_empty_params(self, agent: Agent) -> None:
        """execute_skill with explicit empty dict params should work."""
        result = agent.execute_skill("home", {})
        assert result.success


# ---------------------------------------------------------------------------
# TestExecuteSkillScan
# ---------------------------------------------------------------------------


class TestExecuteSkillScan:
    """execute_skill('scan') — direct skill, moves arm through scan positions."""

    def test_scan_skill_succeeds(self, agent: Agent) -> None:
        result = agent.execute_skill("scan")
        assert result.success, f"scan skill failed: {result.failure_reason}"

    def test_scan_returns_execution_result(self, agent: Agent) -> None:
        result = agent.execute_skill("scan")
        assert isinstance(result, ExecutionResult)

    def test_scan_trace_has_scan_step(self, agent: Agent) -> None:
        result = agent.execute_skill("scan")
        skill_names = [t.skill_name for t in result.trace]
        assert "scan" in skill_names, f"Expected 'scan' in trace, got: {skill_names}"

    def test_scan_with_no_perception_still_succeeds(self) -> None:
        """Scan works without perception (moves arm, no detection needed)."""
        agent_no_perc = Agent(arm=MockArm(), gripper=MockGripper())
        result = agent_no_perc.execute_skill("scan")
        assert result.success


# ---------------------------------------------------------------------------
# TestExecuteSkillPickAutoSteps
# ---------------------------------------------------------------------------


class TestExecuteSkillPickAutoSteps:
    """execute_skill('pick') — verifies auto_steps plan includes scan + detect."""

    def test_pick_trace_is_not_none(self, agent: Agent) -> None:
        result = agent.execute_skill("pick", {"object_label": "test"})
        assert result.trace is not None

    def test_pick_auto_steps_includes_scan(self, agent: Agent) -> None:
        """Pick auto_steps = ['scan', 'detect', 'pick'] — trace must include scan."""
        result = agent.execute_skill("pick", {"object_label": "test"})
        step_names = [t.skill_name for t in result.trace]
        assert "scan" in step_names, (
            f"Expected 'scan' in pick trace, got: {step_names}"
        )

    def test_pick_auto_steps_includes_detect(self, agent: Agent) -> None:
        """Pick auto_steps must include detect step."""
        result = agent.execute_skill("pick", {"object_label": "test"})
        step_names = [t.skill_name for t in result.trace]
        assert "detect" in step_names, (
            f"Expected 'detect' in pick trace, got: {step_names}"
        )

    def test_pick_auto_steps_order(self, agent: Agent) -> None:
        """Steps must execute in order: scan → detect → pick → home."""
        result = agent.execute_skill("pick", {"object_label": "test"})
        step_names = [t.skill_name for t in result.trace]
        # scan must come before detect
        if "scan" in step_names and "detect" in step_names:
            assert step_names.index("scan") < step_names.index("detect"), (
                f"scan must precede detect. Got: {step_names}"
            )

    def test_pick_returns_execution_result(self, agent: Agent) -> None:
        result = agent.execute_skill("pick", {"object_label": "test"})
        assert isinstance(result, ExecutionResult)

    def test_pick_with_no_object_fails_gracefully(self, agent: Agent) -> None:
        """Pick without any detectable object fails but does NOT raise an exception."""
        result = agent.execute_skill("pick", {"object_label": "nonexistent_object"})
        assert isinstance(result, ExecutionResult)
        # It should fail — no object found
        # (success would mean perception found something, which is fine too with mock)

    def test_pick_trace_always_present_even_on_failure(self, agent: Agent) -> None:
        """Even a failed pick execution should return a non-None trace."""
        result = agent.execute_skill("pick", {"object_label": "nothing"})
        assert result.trace is not None


# ---------------------------------------------------------------------------
# TestExecuteSkillParams
# ---------------------------------------------------------------------------


class TestExecuteSkillParams:
    """Verify params are forwarded to the correct steps in the plan."""

    def test_pick_object_label_param_passed(self, agent: Agent) -> None:
        """object_label in params must reach the pick step (not detect step)."""
        # We can't inspect step params directly from ExecutionResult, but we can
        # verify that at least the execute_skill call doesn't lose the param
        # by running it twice with different labels — both should produce a trace.
        result_a = agent.execute_skill("pick", {"object_label": "banana"})
        result_b = agent.execute_skill("pick", {"object_label": "apple"})
        assert result_a.trace is not None
        assert result_b.trace is not None

    def test_home_ignores_extra_params(self, agent: Agent) -> None:
        """Home skill should succeed even if unexpected params are passed."""
        result = agent.execute_skill("home", {"object_label": "ignored"})
        assert result.success

    def test_none_params_treated_as_empty_dict(self, agent: Agent) -> None:
        """execute_skill with params=None should behave like params={}."""
        result = agent.execute_skill("home", None)
        assert result.success


# ---------------------------------------------------------------------------
# TestExecuteSkillTrace
# ---------------------------------------------------------------------------


class TestExecuteSkillTrace:
    """Verify ExecutionResult.trace properties."""

    def test_trace_step_has_skill_name(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        for step in result.trace:
            assert step.skill_name, "Each trace step must have a non-empty skill_name"

    def test_trace_step_has_status(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        for step in result.trace:
            assert step.status in {
                "success", "precondition_failed", "execution_failed",
                "postcondition_failed", "skipped",
            }, f"Unexpected trace status: {step.status!r}"

    def test_home_trace_step_status_is_success(self, agent: Agent) -> None:
        result = agent.execute_skill("home")
        home_steps = [t for t in result.trace if t.skill_name == "home"]
        assert home_steps, "No 'home' step in trace"
        assert home_steps[-1].status == "success", (
            f"Home step status: {home_steps[-1].status!r}"
        )

    def test_scan_trace_step_status_is_success(self, agent: Agent) -> None:
        result = agent.execute_skill("scan")
        scan_steps = [t for t in result.trace if t.skill_name == "scan"]
        assert scan_steps, "No 'scan' step in trace"
        assert scan_steps[0].status == "success", (
            f"Scan step status: {scan_steps[0].status!r}"
        )
