"""Unit tests for LLM provider classes.

TDD — written before implementation. Tests verify:
- ClaudeProvider, OpenAIProvider, LocalProvider construction
- All providers satisfy LLMProvider protocol
- Default values are correct
- No real API calls are made (mock responses only)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.llm.base import LLMProvider
from vector_os_nano.llm.claude import ClaudeProvider
from vector_os_nano.llm.openai_compat import OpenAIProvider
from vector_os_nano.llm.local import LocalProvider
from vector_os_nano.core.types import TaskPlan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SKILLS: list[dict] = [
    {
        "name": "detect",
        "description": "Detect objects",
        "parameters": {"query": {"type": "string"}},
        "preconditions": [],
        "postconditions": [],
        "effects": {},
    }
]

SAMPLE_WORLD_STATE: dict = {
    "objects": {"obj_001": {"label": "cup", "visible": True}},
    "gripper_state": "empty",
}

MOCK_PLAN_RESPONSE = json.dumps({
    "steps": [
        {
            "step_id": "s1",
            "skill_name": "detect",
            "parameters": {"query": "cup"},
            "depends_on": [],
            "preconditions": [],
            "postconditions": [],
        }
    ]
})

MOCK_API_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": MOCK_PLAN_RESPONSE,
            }
        }
    ]
}

MOCK_QUERY_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The scene has a red cup.",
            }
        }
    ]
}


# ---------------------------------------------------------------------------
# ClaudeProvider creation tests
# ---------------------------------------------------------------------------


class TestClaudeProviderCreation:
    def test_claude_provider_creation_with_api_key(self) -> None:
        provider = ClaudeProvider(api_key="test-key-123")
        assert provider is not None

    def test_claude_provider_default_model(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert provider.model == "anthropic/claude-haiku-4-5"

    def test_claude_provider_custom_model(self) -> None:
        provider = ClaudeProvider(api_key="test-key", model="anthropic/claude-haiku-4-5")
        assert provider.model == "anthropic/claude-haiku-4-5"

    def test_claude_provider_default_api_base_openrouter(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert "openrouter" in provider.api_base

    def test_claude_provider_custom_api_base(self) -> None:
        provider = ClaudeProvider(
            api_key="test-key",
            api_base="https://api.anthropic.com/v1",
        )
        assert provider.api_base == "https://api.anthropic.com/v1"

    def test_claude_provider_default_max_history(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert provider.max_history == 20

    def test_claude_provider_custom_max_history(self) -> None:
        provider = ClaudeProvider(api_key="test-key", max_history=10)
        assert provider.max_history == 10

    def test_claude_provider_api_key_stored(self) -> None:
        provider = ClaudeProvider(api_key="sk-secret-key")
        assert provider._api_key == "sk-secret-key"
        # Key must not leak in repr
        assert "sk-secret-key" not in repr(provider)


# ---------------------------------------------------------------------------
# OpenAIProvider creation tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderCreation:
    def test_openai_provider_creation(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider is not None

    def test_openai_provider_default_api_base(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider.api_base == "https://api.openai.com/v1"

    def test_openai_provider_custom_api_base(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            model="mistral",
            api_base="http://localhost:1234/v1",
        )
        assert provider.api_base == "http://localhost:1234/v1"

    def test_openai_provider_model_stored(self) -> None:
        provider = OpenAIProvider(api_key="key", model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"

    def test_openai_provider_api_key_stored(self) -> None:
        provider = OpenAIProvider(api_key="sk-openai-key", model="gpt-4o")
        assert provider.api_key == "sk-openai-key"


# ---------------------------------------------------------------------------
# LocalProvider creation tests
# ---------------------------------------------------------------------------


class TestLocalProviderDefaults:
    def test_local_provider_default_model(self) -> None:
        provider = LocalProvider()
        assert provider.model == "llama3"

    def test_local_provider_default_host(self) -> None:
        provider = LocalProvider()
        assert provider.api_base == "http://localhost:11434/v1"

    def test_local_provider_custom_model(self) -> None:
        provider = LocalProvider(model="mistral")
        assert provider.model == "mistral"

    def test_local_provider_custom_host(self) -> None:
        provider = LocalProvider(host="http://192.168.1.50:11434")
        assert "192.168.1.50" in provider.api_base

    def test_local_provider_no_api_key_required(self) -> None:
        """LocalProvider should work without an API key."""
        provider = LocalProvider()
        assert provider is not None

    def test_local_provider_is_openai_provider(self) -> None:
        """LocalProvider inherits from OpenAIProvider."""
        provider = LocalProvider()
        assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestProviderProtocolCompliance:
    def test_claude_provider_satisfies_protocol(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    def test_openai_provider_satisfies_protocol(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert isinstance(provider, LLMProvider)

    def test_local_provider_satisfies_protocol(self) -> None:
        provider = LocalProvider()
        assert isinstance(provider, LLMProvider)

    def test_claude_provider_has_plan_method(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert hasattr(provider, "plan")
        assert callable(provider.plan)

    def test_claude_provider_has_query_method(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        assert hasattr(provider, "query")
        assert callable(provider.query)

    def test_openai_provider_has_plan_method(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert hasattr(provider, "plan")
        assert callable(provider.plan)

    def test_openai_provider_has_query_method(self) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert hasattr(provider, "query")
        assert callable(provider.query)


# ---------------------------------------------------------------------------
# Plan method tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestClaudeProviderPlan:
    def test_plan_returns_task_plan(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(provider._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_API_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = provider.plan(
                goal="detect the cup",
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
            )

        assert isinstance(result, TaskPlan)

    def test_plan_returns_task_plan_with_steps(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(provider._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_API_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = provider.plan(
                goal="detect the cup",
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
            )

        assert len(result.steps) == 1
        assert result.steps[0].skill_name == "detect"

    def test_plan_goal_preserved(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        goal = "detect the cup"
        with patch.object(provider._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_API_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = provider.plan(
                goal=goal,
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
            )

        assert result.goal == goal

    def test_plan_handles_timeout(self) -> None:
        """Timeout returns an empty TaskPlan without raising."""
        import httpx
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(provider._http, "post", side_effect=httpx.TimeoutException("timeout")):
            result = provider.plan(
                goal="pick cup",
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
            )
        assert isinstance(result, TaskPlan)
        assert len(result.steps) == 0

    def test_plan_handles_network_error(self) -> None:
        """Network error returns an empty TaskPlan without raising."""
        import httpx
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(
            provider._http, "post",
            side_effect=httpx.RequestError("connection refused"),
        ):
            result = provider.plan(
                goal="pick cup",
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
            )
        assert isinstance(result, TaskPlan)
        assert len(result.steps) == 0

    def test_plan_with_history(self) -> None:
        """plan() accepts optional history list."""
        provider = ClaudeProvider(api_key="test-key")
        history = [{"role": "user", "content": "previous message"}]
        with patch.object(provider._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_API_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = provider.plan(
                goal="detect the cup",
                world_state=SAMPLE_WORLD_STATE,
                skill_schemas=SAMPLE_SKILLS,
                history=history,
            )

        assert isinstance(result, TaskPlan)


# ---------------------------------------------------------------------------
# Query method tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestClaudeProviderQuery:
    def test_query_returns_string(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(provider._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_QUERY_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = provider.query("What do you see?")

        assert isinstance(result, str)
        assert "cup" in result

    def test_query_handles_timeout(self) -> None:
        import httpx
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(provider._http, "post", side_effect=httpx.TimeoutException("timeout")):
            result = provider.query("What do you see?")
        assert isinstance(result, str)
        assert len(result) > 0  # Should return an error message, not empty

    def test_query_handles_network_error(self) -> None:
        import httpx
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(
            provider._http, "post",
            side_effect=httpx.RequestError("connection refused"),
        ):
            result = provider.query("What do you see?")
        assert isinstance(result, str)
