# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for vector_os_nano.core.agent.Agent.

Tests cover all public Agent methods with mock hardware/LLM objects.
No real hardware or network calls are made.
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.types import ExecutionResult, TaskPlan, TaskStep, SkillResult
from vector_os_nano.core.skill import Skill, SkillContext, SkillRegistry
from vector_os_nano.core.world_model import WorldModel


# ---------------------------------------------------------------------------
# Mock hardware / LLM helpers
# ---------------------------------------------------------------------------


class MockArm:
    """Minimal ArmProtocol implementation for tests."""

    name = "mock"
    joint_names = ["j1", "j2", "j3", "j4", "j5"]
    dof = 5
    _connected = False
    _bus = None  # No bus — prevents auto-gripper creation

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 5

    def move_joints(self, pos: list[float], duration: float = 3.0) -> bool:
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
    """Minimal GripperProtocol implementation for tests."""

    def open(self) -> bool:
        return True

    def close(self) -> bool:
        return True

    def is_holding(self) -> bool:
        return True

    def get_position(self) -> float:
        return 0.5

    def get_force(self):
        return None


class MockLLM:
    """Mock LLM that returns a single-step 'home' plan."""

    def plan(
        self,
        goal: str,
        world_state: dict,
        skill_schemas: list,
        history=None,
        model_override: str | None = None,
    ) -> TaskPlan:
        return TaskPlan(
            goal=goal,
            steps=[TaskStep(step_id="s1", skill_name="home", parameters={})],
        )

    def query(self, prompt: str, image=None, model_override: str | None = None) -> str:
        return "I see a robot arm."

    def classify(self, user_message: str, model_override: str | None = None) -> str:
        return "task"

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        history=None,
        model_override: str | None = None,
    ) -> str:
        return "Done."

    def summarize(
        self,
        original_request: str,
        execution_trace: str,
        model_override: str | None = None,
    ) -> str:
        return "Task complete."


class MockLLMClarification:
    """Mock LLM that always returns a clarification request."""

    def plan(
        self,
        goal: str,
        world_state: dict,
        skill_schemas: list,
        history=None,
        model_override: str | None = None,
    ) -> TaskPlan:
        return TaskPlan(
            goal=goal,
            steps=[],
            requires_clarification=True,
            clarification_question="Which object should I pick?",
        )

    def query(self, prompt: str, image=None, model_override: str | None = None) -> str:
        return ""

    def classify(self, user_message: str, model_override: str | None = None) -> str:
        return "task"

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        history=None,
        model_override: str | None = None,
    ) -> str:
        return ""

    def summarize(
        self,
        original_request: str,
        execution_trace: str,
        model_override: str | None = None,
    ) -> str:
        return ""


class MockLLMFail:
    """Mock LLM that always returns a plan that will fail (unknown skill)."""

    def plan(
        self,
        goal: str,
        world_state: dict,
        skill_schemas: list,
        history=None,
        model_override: str | None = None,
    ) -> TaskPlan:
        return TaskPlan(
            goal=goal,
            steps=[TaskStep(step_id="s1", skill_name="nonexistent_skill", parameters={})],
        )

    def query(self, prompt: str, image=None, model_override: str | None = None) -> str:
        return ""

    def classify(self, user_message: str, model_override: str | None = None) -> str:
        return "task"

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        history=None,
        model_override: str | None = None,
    ) -> str:
        return ""

    def summarize(
        self,
        original_request: str,
        execution_trace: str,
        model_override: str | None = None,
    ) -> str:
        return ""


class AlwaysSucceedSkill:
    """Custom skill that always succeeds, for testing registration."""

    name = "always_succeed"
    description = "A test skill that always succeeds"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        return SkillResult(success=True)


# ---------------------------------------------------------------------------
# Test: Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    """Tests for Agent constructor with various combinations of arguments."""

    def test_agent_creation_minimal(self):
        """Agent(arm=MockArm()) should succeed with no other args."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        assert agent is not None

    def test_agent_creation_no_args(self):
        """Agent() with zero arguments should succeed."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        assert agent is not None

    def test_agent_creation_with_llm(self):
        """Agent(arm=MockArm(), llm=MockLLM()) should succeed."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), llm=MockLLM())
        assert agent is not None

    def test_agent_creation_with_api_key(self):
        """Agent(llm_api_key='test') should create a ClaudeProvider internally."""
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.llm.claude import ClaudeProvider

        agent = Agent(arm=MockArm(), llm_api_key="test-api-key")
        assert isinstance(agent._llm, ClaudeProvider)

    def test_agent_creation_with_gripper(self):
        """Explicit gripper should be stored on the agent."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        gripper = MockGripper()
        agent = Agent(arm=arm, gripper=gripper)
        assert agent._gripper is gripper

    def test_agent_creation_with_custom_config(self):
        """Agent accepts dict config without crashing."""
        from vector_os_nano.core.agent import Agent

        cfg = {"agent": {"max_planning_retries": 2}, "llm": {"provider": "claude"}}
        agent = Agent(arm=MockArm(), config=cfg)
        assert agent is not None

    def test_agent_creation_llm_overrides_api_key(self):
        """If both llm= and llm_api_key= are passed, llm= wins."""
        from vector_os_nano.core.agent import Agent

        mock_llm = MockLLM()
        agent = Agent(arm=MockArm(), llm=mock_llm, llm_api_key="ignored")
        assert agent._llm is mock_llm

    def test_agent_creation_no_arm(self):
        """Agent with no arm should set _arm to None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        assert agent._arm is None


# ---------------------------------------------------------------------------
# Test: execute() — direct mode (no LLM)
# ---------------------------------------------------------------------------


class TestAgentExecuteDirect:
    """Tests for _execute_direct() path — used when no LLM is configured."""

    def test_agent_execute_direct_home(self):
        """'home' command without LLM should succeed via HomeSkill."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), gripper=MockGripper())
        result = agent.execute("home")
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    def test_agent_execute_direct_scan(self):
        """'scan' command without LLM should succeed via ScanSkill."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), gripper=MockGripper())
        result = agent.execute("scan")
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    def test_agent_execute_direct_unknown(self):
        """Unknown command without LLM should return a failure result."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        result = agent.execute("fly to the moon")
        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert result.failure_reason is not None
        assert "No LLM" in result.failure_reason or "Unknown command" in result.failure_reason

    def test_agent_execute_direct_pick_with_arg(self):
        """'pick red_cup' should pass object_label to PickSkill params."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        gripper = MockGripper()
        agent = Agent(arm=arm, gripper=gripper)
        # PickSkill may fail (no perception), but it should attempt with correct param
        result = agent.execute("pick red_cup")
        assert isinstance(result, ExecutionResult)
        # Success or failure — not crashing is the key assertion here

    def test_agent_execute_direct_empty_string(self):
        """Empty instruction should return failure gracefully."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        result = agent.execute("")
        assert isinstance(result, ExecutionResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# Test: execute() — LLM planning mode
# ---------------------------------------------------------------------------


class TestAgentExecuteWithLLM:
    """Tests for execute() path with an LLM configured."""

    def test_agent_execute_with_llm(self):
        """With MockLLM, execute() should complete a 'home' plan."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), gripper=MockGripper(), llm=MockLLM())
        result = agent.execute("go home")
        assert isinstance(result, ExecutionResult)
        assert result.success is True

    def test_agent_execute_with_llm_clarification(self):
        """When LLM requests clarification, result should have clarification_question."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), llm=MockLLMClarification())
        # Use an instruction that has no alias match so the LLM classify/plan path runs
        result = agent.execute("do something ambiguous please")
        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert result.status == "clarification_needed"
        assert result.clarification_question is not None
        assert len(result.clarification_question) > 0

    def test_agent_execute_with_llm_failure(self):
        """Failed execution (unknown skill from LLM) should return failure result."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), llm=MockLLMFail())
        result = agent.execute("do something impossible")
        assert isinstance(result, ExecutionResult)
        assert result.success is False

    def test_agent_execute_returns_execution_result(self):
        """execute() always returns an ExecutionResult, never raises."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), gripper=MockGripper(), llm=MockLLM())
        result = agent.execute("any instruction")
        assert isinstance(result, ExecutionResult)


# ---------------------------------------------------------------------------
# Test: register_skill()
# ---------------------------------------------------------------------------


class TestAgentRegisterSkill:
    """Tests for register_skill() and skills property."""

    def test_agent_register_custom_skill(self):
        """Registered custom skill should appear in agent.skills."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        agent.register_skill(AlwaysSucceedSkill())
        assert "always_succeed" in agent.skills

    def test_agent_skills_property_includes_defaults(self):
        """agent.skills should include all default built-in skills."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        skill_names = agent.skills
        assert "home" in skill_names
        assert "scan" in skill_names
        assert "pick" in skill_names
        assert "place" in skill_names
        assert "detect" in skill_names

    def test_agent_register_overrides_existing(self):
        """Re-registering a skill with the same name should replace it."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        agent.register_skill(AlwaysSucceedSkill())
        agent.register_skill(AlwaysSucceedSkill())  # second register
        assert agent.skills.count("always_succeed") == 1

    def test_agent_execute_custom_skill_direct(self):
        """Custom skill registered before execute should be callable via direct mode."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        agent.register_skill(AlwaysSucceedSkill())
        result = agent.execute("always_succeed")
        assert result.success is True


# ---------------------------------------------------------------------------
# Test: world property
# ---------------------------------------------------------------------------


class TestAgentWorldProperty:
    """Tests for the agent.world property."""

    def test_agent_world_property(self):
        """agent.world should return the WorldModel instance."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        assert isinstance(agent.world, WorldModel)

    def test_agent_world_is_same_instance(self):
        """Repeated access to agent.world returns the same object."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        w1 = agent.world
        w2 = agent.world
        assert w1 is w2


# ---------------------------------------------------------------------------
# Test: skills property
# ---------------------------------------------------------------------------


class TestAgentSkillsProperty:
    """Tests for the agent.skills property."""

    def test_agent_skills_property(self):
        """agent.skills should return a list of strings."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        skill_names = agent.skills
        assert isinstance(skill_names, list)
        assert all(isinstance(n, str) for n in skill_names)

    def test_agent_skills_not_empty(self):
        """Default skills should be non-empty."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        assert len(agent.skills) > 0


# ---------------------------------------------------------------------------
# Test: home() convenience
# ---------------------------------------------------------------------------


class TestAgentHomeConvenience:
    """Tests for agent.home() shorthand."""

    def test_agent_home_convenience_returns_bool(self):
        """agent.home() should return True on success."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), gripper=MockGripper())
        result = agent.home()
        assert isinstance(result, bool)
        assert result is True

    def test_agent_home_convenience_returns_false_on_failure(self):
        """agent.home() should return False when no arm connected."""
        from vector_os_nano.core.agent import Agent

        # No arm → home skill fails with "No arm connected"
        agent = Agent(llm=MockLLMFail())
        result = agent.home()
        assert isinstance(result, bool)
        assert result is False


# ---------------------------------------------------------------------------
# Test: stop()
# ---------------------------------------------------------------------------


class TestAgentStop:
    """Tests for agent.stop()."""

    def test_agent_stop_calls_arm_stop(self):
        """agent.stop() should call arm.stop()."""
        from vector_os_nano.core.agent import Agent

        calls = []

        class TrackingArm(MockArm):
            def stop(self):
                calls.append("stop")

        agent = Agent(arm=TrackingArm())
        agent.stop()
        assert calls == ["stop"]

    def test_agent_stop_no_arm(self):
        """agent.stop() should not crash when arm is None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        agent.stop()  # Should not raise


# ---------------------------------------------------------------------------
# Test: connect() / disconnect()
# ---------------------------------------------------------------------------


class TestAgentConnectDisconnect:
    """Tests for agent.connect() and agent.disconnect()."""

    def test_agent_connect_calls_arm_connect(self):
        """agent.connect() should call arm.connect()."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        agent.connect()
        assert arm._connected is True

    def test_agent_disconnect_calls_arm_disconnect(self):
        """agent.disconnect() should call arm.disconnect()."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        agent.connect()
        agent.disconnect()
        assert arm._connected is False

    def test_agent_connect_no_arm(self):
        """agent.connect() should not crash when arm is None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        agent.connect()  # Should not raise

    def test_agent_disconnect_no_arm(self):
        """agent.disconnect() should not crash when arm is None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        agent.disconnect()  # Should not raise


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


class TestAgentContextManager:
    """Tests for Agent used as a context manager (with statement)."""

    def test_agent_context_manager_connects_and_disconnects(self):
        """with Agent() as a: should connect on enter and disconnect on exit."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        with Agent(arm=arm) as a:
            assert arm._connected is True
            assert a is not None
        assert arm._connected is False

    def test_agent_context_manager_returns_agent(self):
        """The 'as' target should be the Agent instance."""
        from vector_os_nano.core.agent import Agent

        with Agent(arm=MockArm()) as a:
            assert isinstance(a, Agent)

    def test_agent_context_manager_disconnects_on_exception(self):
        """disconnect() should be called even if body raises an exception."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        try:
            with Agent(arm=arm):
                assert arm._connected is True
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert arm._connected is False


# ---------------------------------------------------------------------------
# Test: cross-task memory (SessionMemory integration)
# ---------------------------------------------------------------------------


class MockLLMWithChat:
    """Mock LLM that routes 'hello' to chat, everything else to task planning."""

    def plan(
        self,
        goal: str,
        world_state: dict,
        skill_schemas: list,
        history=None,
        model_override: str | None = None,
    ) -> TaskPlan:
        return TaskPlan(
            goal=goal,
            steps=[TaskStep(step_id="s1", skill_name="home", parameters={})],
        )

    def query(self, prompt: str, image=None, model_override: str | None = None) -> str:
        return "I see a robot arm."

    def classify(self, user_message: str, model_override: str | None = None) -> str:
        """Return 'chat' for greetings, 'task' for everything else."""
        if user_message.lower().startswith("hello"):
            return "chat"
        return "task"

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        history=None,
        model_override: str | None = None,
    ) -> str:
        return f"Hi! Ready to help. You said: {user_message}"

    def summarize(
        self,
        original_request: str,
        execution_trace: str,
        model_override: str | None = None,
    ) -> str:
        return f"Completed: {original_request}"


class TestCrossTaskMemory:
    """Test that conversation context persists across task executions."""

    @pytest.fixture
    def agent(self):
        """Agent with MockArm, MockGripper, and MockLLMWithChat."""
        from vector_os_nano.core.agent import Agent

        return Agent(
            arm=MockArm(),
            gripper=MockGripper(),
            llm=MockLLMWithChat(),
        )

    def test_task_result_in_memory(self, agent):
        """After executing a task via LLM path, the result should be in session memory."""
        # Use an instruction that bypasses direct-skill alias matching
        agent.execute("execute a custom task")
        ctx = agent._memory.get_last_task_context()
        assert ctx is not None

    def test_task_result_metadata_contains_success(self, agent):
        """Task result metadata should record success status."""
        agent.execute("execute a custom task")
        ctx = agent._memory.get_last_task_context()
        assert ctx is not None
        assert "success" in ctx
        assert ctx["success"] is True

    def test_cross_task_history_available(self, agent):
        """After two tasks, both should appear in LLM history."""
        agent.execute("execute first task")
        agent.execute("execute second task")
        history = agent._memory.get_llm_history()
        # Should have entries from both tasks:
        # task1: user message + task_result = 2 entries
        # task2: user message + task_result = 2 entries
        assert len(history) >= 4

    def test_chat_then_task_continuity(self, agent):
        """Chat messages should persist in memory when switching to task mode."""
        # First a chat (classify returns "chat" for "hello...")
        agent.execute("hello there")
        # Then a task
        agent.execute("execute a custom task")
        history = agent._memory.get_llm_history()
        # The chat user message should still be in history
        assert any("hello there" in h["content"] for h in history)

    def test_memory_attribute_exists(self, agent):
        """Agent should expose _memory as a SessionMemory instance."""
        from vector_os_nano.core.memory import SessionMemory

        assert hasattr(agent, "_memory")
        assert isinstance(agent._memory, SessionMemory)

    def test_memory_empty_before_execute(self, agent):
        """Fresh agent should have an empty memory."""
        history = agent._memory.get_llm_history()
        assert history == []
