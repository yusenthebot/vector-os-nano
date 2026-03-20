"""Unit tests for LLM JSON response parsing.

TDD — written before implementation. Tests verify that the parse_plan_response
function in vector_os.llm.claude correctly:
- Parses valid JSON step arrays into TaskPlan
- Handles requires_clarification responses
- Gracefully handles malformed JSON (including markdown code fences)
- Handles empty step lists
- Preserves step dependencies
"""
from __future__ import annotations

import json

import pytest

from vector_os.llm.claude import parse_plan_response
from vector_os.core.types import TaskPlan, TaskStep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PLAN_JSON = json.dumps({
    "steps": [
        {
            "step_id": "s1",
            "skill_name": "detect",
            "parameters": {"query": "red cup"},
            "depends_on": [],
            "preconditions": [],
            "postconditions": ["object_visible(s1)"],
        },
        {
            "step_id": "s2",
            "skill_name": "pick",
            "parameters": {"object_id": "s1"},
            "depends_on": ["s1"],
            "preconditions": ["gripper_empty"],
            "postconditions": ["gripper_holding_any"],
        },
    ]
})

CLARIFICATION_JSON = json.dumps({
    "requires_clarification": True,
    "clarification_question": "Which cup — the red one or the blue one?",
})

EMPTY_STEPS_JSON = json.dumps({"steps": []})

PLAN_WITH_DEPS_JSON = json.dumps({
    "steps": [
        {
            "step_id": "s1",
            "skill_name": "detect",
            "parameters": {"query": "ball"},
            "depends_on": [],
            "preconditions": [],
            "postconditions": [],
        },
        {
            "step_id": "s2",
            "skill_name": "pick",
            "parameters": {"object_id": "s1"},
            "depends_on": ["s1"],
            "preconditions": ["gripper_empty"],
            "postconditions": [],
        },
        {
            "step_id": "s3",
            "skill_name": "place",
            "parameters": {"location": "tray"},
            "depends_on": ["s2"],
            "preconditions": ["gripper_holding_any"],
            "postconditions": [],
        },
    ]
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseValidPlan:
    def test_parse_valid_plan_returns_task_plan(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert isinstance(plan, TaskPlan)

    def test_parse_valid_plan_step_count(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert len(plan.steps) == 2

    def test_parse_valid_plan_step_types(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        for step in plan.steps:
            assert isinstance(step, TaskStep)

    def test_parse_valid_plan_step_ids(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert plan.steps[0].step_id == "s1"
        assert plan.steps[1].step_id == "s2"

    def test_parse_valid_plan_skill_names(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert plan.steps[0].skill_name == "detect"
        assert plan.steps[1].skill_name == "pick"

    def test_parse_valid_plan_parameters(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert plan.steps[0].parameters == {"query": "red cup"}
        assert plan.steps[1].parameters == {"object_id": "s1"}

    def test_parse_valid_plan_not_clarification(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        assert plan.requires_clarification is False

    def test_parse_valid_plan_goal_preserved(self) -> None:
        goal = "pick up the red cup"
        plan = parse_plan_response(goal, VALID_PLAN_JSON)
        assert plan.goal == goal


class TestParseClarification:
    def test_parse_clarification_flag(self) -> None:
        plan = parse_plan_response("pick up the cup", CLARIFICATION_JSON)
        assert plan.requires_clarification is True

    def test_parse_clarification_question(self) -> None:
        plan = parse_plan_response("pick up the cup", CLARIFICATION_JSON)
        assert plan.clarification_question == "Which cup — the red one or the blue one?"

    def test_parse_clarification_no_steps(self) -> None:
        plan = parse_plan_response("pick up the cup", CLARIFICATION_JSON)
        assert len(plan.steps) == 0


class TestParseMalformedJson:
    def test_parse_malformed_json_returns_task_plan(self) -> None:
        """Malformed JSON must return a TaskPlan, not raise an exception."""
        plan = parse_plan_response("do something", "NOT VALID JSON {{{")
        assert isinstance(plan, TaskPlan)

    def test_parse_malformed_json_empty_steps(self) -> None:
        plan = parse_plan_response("do something", "NOT VALID JSON {{{")
        assert len(plan.steps) == 0

    def test_parse_malformed_json_no_clarification(self) -> None:
        plan = parse_plan_response("do something", "NOT VALID JSON {{{")
        assert plan.requires_clarification is False

    def test_parse_markdown_fenced_json(self) -> None:
        """LLMs often wrap JSON in ```json ... ``` — must strip fences."""
        fenced = f"```json\n{VALID_PLAN_JSON}\n```"
        plan = parse_plan_response("pick up the red cup", fenced)
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) == 2

    def test_parse_markdown_fenced_no_language(self) -> None:
        """Handle ``` without language specifier."""
        fenced = f"```\n{VALID_PLAN_JSON}\n```"
        plan = parse_plan_response("pick up the red cup", fenced)
        assert len(plan.steps) == 2

    def test_parse_json_with_leading_trailing_whitespace(self) -> None:
        padded = f"\n  \t  {VALID_PLAN_JSON}  \n"
        plan = parse_plan_response("pick up the red cup", padded)
        assert len(plan.steps) == 2

    def test_parse_empty_string(self) -> None:
        plan = parse_plan_response("do something", "")
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) == 0


class TestParseEmptySteps:
    def test_parse_empty_steps_valid(self) -> None:
        plan = parse_plan_response("do nothing", EMPTY_STEPS_JSON)
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) == 0
        assert plan.requires_clarification is False


class TestParseWithDependencies:
    def test_parse_with_dependencies_preserved(self) -> None:
        plan = parse_plan_response("pick up the ball and place it", PLAN_WITH_DEPS_JSON)
        assert len(plan.steps) == 3

    def test_parse_step_depends_on_preserved(self) -> None:
        plan = parse_plan_response("pick up the ball and place it", PLAN_WITH_DEPS_JSON)
        s1, s2, s3 = plan.steps
        assert s1.depends_on == []
        assert s2.depends_on == ["s1"]
        assert s3.depends_on == ["s2"]

    def test_parse_preconditions_preserved(self) -> None:
        plan = parse_plan_response("pick up the ball and place it", PLAN_WITH_DEPS_JSON)
        s2 = plan.steps[1]
        assert "gripper_empty" in s2.preconditions

    def test_parse_postconditions_preserved(self) -> None:
        plan = parse_plan_response("pick up the red cup", VALID_PLAN_JSON)
        s1 = plan.steps[0]
        assert "object_visible(s1)" in s1.postconditions
