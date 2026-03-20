"""Unit tests for vector_os.llm.prompts.

TDD — written before implementation. Tests verify:
- build_planning_prompt includes skill names and world state
- build_planning_prompt returns a valid non-empty string
- build_tool_definitions converts skill schemas to OpenAI tool format
- PLANNING_SYSTEM_PROMPT is a non-empty string with required fields
"""
from __future__ import annotations

import pytest

from vector_os.llm.prompts import (
    PLANNING_SYSTEM_PROMPT,
    build_planning_prompt,
    build_tool_definitions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SKILLS: list[dict] = [
    {
        "name": "detect",
        "description": "Detect objects in the scene",
        "parameters": {
            "query": {"type": "string", "description": "Object to detect"},
        },
        "preconditions": [],
        "postconditions": ["object_visible(result_id)"],
        "effects": {"last_detection": "result_id"},
    },
    {
        "name": "pick",
        "description": "Pick up a detected object",
        "parameters": {
            "object_id": {"type": "string", "description": "Object to pick"},
        },
        "preconditions": ["gripper_empty", "object_visible(object_id)"],
        "postconditions": ["gripper_holding_any"],
        "effects": {"gripper_state": "holding"},
    },
    {
        "name": "place",
        "description": "Place the held object",
        "parameters": {
            "location": {"type": "string", "description": "Where to place"},
        },
        "preconditions": ["gripper_holding_any"],
        "postconditions": ["gripper_empty"],
        "effects": {"gripper_state": "empty"},
    },
]

SAMPLE_WORLD_STATE: dict = {
    "objects": {
        "obj_001": {"label": "red cup", "visible": True, "reachable": True},
        "obj_002": {"label": "blue ball", "visible": True, "reachable": False},
    },
    "gripper_state": "empty",
    "arm_state": "home",
}


# ---------------------------------------------------------------------------
# build_planning_prompt
# ---------------------------------------------------------------------------


class TestBuildPlanningPrompt:
    def test_build_planning_prompt_includes_skills(self) -> None:
        """Prompt must contain each skill name from skill_schemas."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        for skill in SAMPLE_SKILLS:
            assert skill["name"] in prompt, (
                f"Expected skill name '{skill['name']}' in prompt"
            )

    def test_build_planning_prompt_includes_skill_descriptions(self) -> None:
        """Prompt should include skill descriptions for LLM context."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        for skill in SAMPLE_SKILLS:
            assert skill["description"] in prompt, (
                f"Expected description for '{skill['name']}' in prompt"
            )

    def test_build_planning_prompt_includes_world_state(self) -> None:
        """Prompt must contain object labels from the world state."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        assert "red cup" in prompt
        assert "blue ball" in prompt

    def test_build_planning_prompt_format(self) -> None:
        """Prompt is a valid non-empty string."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        assert isinstance(prompt, str)
        assert len(prompt) > 100, "Prompt is suspiciously short"

    def test_build_planning_prompt_empty_skills(self) -> None:
        """Empty skill list is valid — prompt should still be a string."""
        prompt = build_planning_prompt([], SAMPLE_WORLD_STATE)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_planning_prompt_empty_world_state(self) -> None:
        """Empty world state dict is valid."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, {})
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_planning_prompt_contains_output_format(self) -> None:
        """Prompt must instruct LLM on expected JSON output format."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        assert "steps" in prompt
        assert "step_id" in prompt

    def test_build_planning_prompt_contains_clarification_instruction(self) -> None:
        """Prompt must mention requires_clarification for ambiguous goals."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        assert "clarification" in prompt.lower()

    def test_build_planning_prompt_gripper_rule(self) -> None:
        """Prompt must include the gripper constraint rule."""
        prompt = build_planning_prompt(SAMPLE_SKILLS, SAMPLE_WORLD_STATE)
        assert "gripper" in prompt.lower()


# ---------------------------------------------------------------------------
# build_tool_definitions
# ---------------------------------------------------------------------------


class TestBuildToolDefinitions:
    def test_build_tool_definitions_returns_list(self) -> None:
        tools = build_tool_definitions(SAMPLE_SKILLS)
        assert isinstance(tools, list)
        assert len(tools) == len(SAMPLE_SKILLS)

    def test_build_tool_definitions_type_function(self) -> None:
        """Each tool must have type='function'."""
        tools = build_tool_definitions(SAMPLE_SKILLS)
        for tool in tools:
            assert tool["type"] == "function"

    def test_build_tool_definitions_has_name(self) -> None:
        """Each tool must have the correct function name."""
        tools = build_tool_definitions(SAMPLE_SKILLS)
        tool_names = {t["function"]["name"] for t in tools}
        for skill in SAMPLE_SKILLS:
            assert skill["name"] in tool_names

    def test_build_tool_definitions_has_description(self) -> None:
        """Each tool function must have a non-empty description."""
        tools = build_tool_definitions(SAMPLE_SKILLS)
        for tool in tools:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 0

    def test_build_tool_definitions_has_parameters(self) -> None:
        """Each tool must have a parameters schema."""
        tools = build_tool_definitions(SAMPLE_SKILLS)
        for tool in tools:
            fn = tool["function"]
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_build_tool_definitions_empty(self) -> None:
        """Empty skills list returns empty tool list."""
        tools = build_tool_definitions([])
        assert tools == []

    def test_build_tool_definitions_parameter_names_match(self) -> None:
        """Parameter properties in tools must match skill schema keys."""
        tools = build_tool_definitions(SAMPLE_SKILLS)
        for tool, skill in zip(tools, SAMPLE_SKILLS):
            expected_params = set(skill["parameters"].keys())
            actual_params = set(tool["function"]["parameters"]["properties"].keys())
            assert expected_params == actual_params, (
                f"Tool '{skill['name']}': expected params {expected_params}, "
                f"got {actual_params}"
            )


# ---------------------------------------------------------------------------
# PLANNING_SYSTEM_PROMPT constant
# ---------------------------------------------------------------------------


class TestPlanningSystemPrompt:
    def test_planning_system_prompt_is_string(self) -> None:
        assert isinstance(PLANNING_SYSTEM_PROMPT, str)

    def test_planning_system_prompt_non_empty(self) -> None:
        assert len(PLANNING_SYSTEM_PROMPT) > 200

    def test_planning_system_prompt_has_json_format(self) -> None:
        """Prompt template must include JSON output format instructions."""
        assert "steps" in PLANNING_SYSTEM_PROMPT
        assert "skill_name" in PLANNING_SYSTEM_PROMPT

    def test_planning_system_prompt_has_placeholders(self) -> None:
        """Prompt template must have {skills_json} and {world_state_json} placeholders."""
        assert "{skills_json}" in PLANNING_SYSTEM_PROMPT
        assert "{world_state_json}" in PLANNING_SYSTEM_PROMPT

    def test_planning_system_prompt_has_clarification_block(self) -> None:
        """Prompt must include requires_clarification output option."""
        assert "requires_clarification" in PLANNING_SYSTEM_PROMPT
