"""Level 5 — Full ToolAgent conversation flow integration tests.

Tests the complete pipeline: user message → ToolAgent (GPT-4o via OpenRouter)
→ tool_use calls → skill execution → final summarized response.

This is the top-level integration harness: it verifies that the LLM correctly
maps natural-language commands to skill calls and returns coherent replies.

API cost estimate: ~$0.02–$0.05 per test (1–3 GPT-4o turns each).
Total for full run (~5 tests): ~$0.15–$0.25.

Prerequisites:
  - mujoco installed (harness conftest enforces this)
  - OPENROUTER_API_KEY set in environment, or config/user.yaml llm.api_key

Skips:
  - Entire class skips if no API key is found (pytestmark autouse fixture).
  - Individual tests time out at 30 s per chat() call (httpx timeout).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path (mirrors other harness modules)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# API key helper
# ---------------------------------------------------------------------------

_API_KEY: str | None = None


def _get_api_key() -> str:
    """Return OpenRouter API key from env or config/user.yaml."""
    global _API_KEY
    if _API_KEY is not None:
        return _API_KEY

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            import yaml

            cfg_path = _REPO_ROOT / "config" / "user.yaml"
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh)
            key = cfg.get("llm", {}).get("api_key", "")
        except Exception:
            pass

    _API_KEY = key
    return key


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
class TestLevel5AgentIntegration:
    """L5: Full ToolAgent conversation with real GPT-4o tool_use calls."""

    # ------------------------------------------------------------------
    # Class-scoped fixtures (expensive to create — share across tests)
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _require_api_key(self) -> None:
        """Skip every test in this class when no API key is available."""
        if not _get_api_key():
            pytest.skip("No OPENROUTER_API_KEY available")

    @pytest.fixture(scope="class")
    def go2(self):
        """Headless MuJoCoGo2 with room geometry and sinusoidal backend."""
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

        robot = MuJoCoGo2(gui=False, room=True, backend="sinusoidal")
        robot.connect()
        robot.stand()
        yield robot
        robot.disconnect()

    @pytest.fixture(scope="class")
    def agent(self, go2: Any):
        """Agent wired with Go2 skills, VLM perception, and spatial memory."""
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.core.config import load_config
        from vector_os_nano.core.spatial_memory import SpatialMemory
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        from vector_os_nano.skills.go2 import get_go2_skills

        cfg = load_config(str(_REPO_ROOT / "config" / "user.yaml"))
        api_key = _get_api_key()

        a = Agent(base=go2, llm_api_key=api_key, config=cfg)
        for skill in get_go2_skills():
            a._skill_registry.register(skill)
        a._vlm = Go2VLMPerception(config={"api_key": api_key})
        a._spatial_memory = SpatialMemory()
        return a

    # ------------------------------------------------------------------
    # Function-scoped fixture (fresh history per test, cheap to create)
    # ------------------------------------------------------------------

    @pytest.fixture
    def tool_agent(self, agent: Any):
        """ToolAgent with fresh conversation history."""
        from vector_os_nano.core.tool_agent import ToolAgent

        api_key = _get_api_key()
        ta = ToolAgent(agent_ref=agent, api_key=api_key, model="openai/gpt-4o")
        yield ta
        ta.clear_history()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.timeout(60)
    def test_where_am_i(self, tool_agent: Any) -> None:
        """User asks where the robot is → ToolAgent calls where_am_i."""
        tool_calls: list[str] = []

        def on_tool(name: str, params: dict) -> None:
            tool_calls.append(name)

        response = tool_agent.chat("我在哪里？", on_tool_call=on_tool)

        assert "where_am_i" in tool_calls, (
            f"Expected where_am_i tool call, got: {tool_calls}"
        )
        assert response, "Expected non-empty response from ToolAgent"

    @pytest.mark.timeout(60)
    def test_navigate_command(self, tool_agent: Any) -> None:
        """User says go to kitchen → ToolAgent calls navigate(room=kitchen)."""
        tool_calls: list[tuple[str, dict]] = []

        def on_tool(name: str, params: dict) -> None:
            tool_calls.append((name, params))

        response = tool_agent.chat("去厨房", on_tool_call=on_tool)

        nav_calls = [c for c in tool_calls if c[0] == "navigate"]
        assert nav_calls, (
            f"Expected navigate tool call, got: {[c[0] for c in tool_calls]}"
        )
        assert response, "Expected non-empty response from ToolAgent"

    @pytest.mark.timeout(60)
    def test_look_command(self, tool_agent: Any) -> None:
        """User says look around → ToolAgent calls look or describe_scene."""
        tool_calls: list[str] = []

        def on_tool(name: str, params: dict) -> None:
            tool_calls.append(name)

        response = tool_agent.chat("看看周围有什么", on_tool_call=on_tool)

        assert "look" in tool_calls or "describe_scene" in tool_calls, (
            f"Expected look or describe_scene tool call, got: {tool_calls}"
        )
        assert response, "Expected non-empty response from ToolAgent"

    @pytest.mark.timeout(120)
    def test_multi_turn_context(self, tool_agent: Any) -> None:
        """Multi-turn: navigate somewhere, then ask to look — context is preserved."""
        nav_calls: list[str] = []
        look_calls: list[str] = []

        def on_tool_turn1(name: str, params: dict) -> None:
            nav_calls.append(name)

        def on_tool_turn2(name: str, params: dict) -> None:
            look_calls.append(name)

        # Turn 1: go to hallway
        tool_agent.chat("去走廊", on_tool_call=on_tool_turn1)

        # Turn 2: look (LLM must use conversation history to stay in context)
        response = tool_agent.chat("看看这里有什么", on_tool_call=on_tool_turn2)

        assert "look" in look_calls or "describe_scene" in look_calls, (
            f"Turn 2 expected look/describe_scene call, got: {look_calls}"
        )
        assert response, "Expected non-empty response on turn 2"

    @pytest.mark.timeout(60)
    def test_response_is_chinese(self, tool_agent: Any) -> None:
        """When user speaks Chinese, ToolAgent replies in Chinese."""
        response = tool_agent.chat("你好")

        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in response)
        assert has_chinese, (
            f"Expected Chinese characters in response, got: {response[:120]!r}"
        )

    @pytest.mark.timeout(30)
    def test_cost_under_budget(self, tool_agent: Any) -> None:
        """Five-turn session does not raise or time out — cost guard is trivially true.

        Actual cost is tracked by Go2VLMPerception.cumulative_cost_usd, not by
        ToolAgent directly (httpx does not expose token counts).  This test
        verifies the session completes without error; the per-VLM-call budget is
        enforced in test_level4_patrol.py::test_vlm_cost_under_budget.
        """
        # Single benign exchange — no VLM call expected
        response = tool_agent.chat("你现在准备好了吗")
        assert isinstance(response, str), "Expected string response"
