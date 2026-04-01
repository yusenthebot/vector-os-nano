"""Tests for Go2 simulation launch and ToolAgent Go2 prompt."""
import pytest

pytest.importorskip("convex_mpc")


class TestInitSimGo2:
    def test_init_sim_go2_returns_tuple(self):
        """_init_sim_go2 should return (None, None, None, None, base)."""
        import sys
        sys.path.insert(0, ".")
        from run import _init_sim_go2
        from vector_os_nano.core.config import load_config
        cfg = load_config()
        arm, gripper, perception, calibration, base = _init_sim_go2(cfg, gui=False)
        assert arm is None
        assert gripper is None
        assert perception is None
        assert calibration is None
        assert base is not None
        assert hasattr(base, "walk")
        assert hasattr(base, "stand")
        assert hasattr(base, "get_position")
        base.disconnect()


class TestToolAgentGo2Prompt:
    def test_system_prompt_go2_mode(self):
        """ToolAgent system prompt should include Go2 info when base is set."""
        from unittest.mock import MagicMock
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.core.tool_agent import ToolAgent

        # Create agent with mock base, no arm
        mock_base = MagicMock()
        mock_base.get_position.return_value = (2.5, 2.5, 0.27)
        mock_base.get_heading.return_value = 0.0

        agent = Agent(base=mock_base)

        # ToolAgent needs api_key but we only test prompt building
        tool_agent = ToolAgent(
            agent_ref=agent,
            api_key="test-key",
            model="test",
        )

        prompt = tool_agent._build_system_prompt()
        assert "Go2" in prompt or "go2" in prompt or "quadruped" in prompt.lower()

    def test_system_prompt_shows_position(self):
        from unittest.mock import MagicMock
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.core.tool_agent import ToolAgent

        mock_base = MagicMock()
        mock_base.get_position.return_value = (1.5, 3.0, 0.27)
        mock_base.get_heading.return_value = 1.57

        agent = Agent(base=mock_base)
        tool_agent = ToolAgent(agent_ref=agent, api_key="k", model="m")
        prompt = tool_agent._build_system_prompt()
        # Should contain position info
        assert "1.5" in prompt or "3.0" in prompt
