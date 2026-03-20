"""Unit tests for built-in skill schemas and protocol compliance.

Tests verify:
- Each skill has correct name, description, parameters, pre/postconditions, effects
- All skills satisfy the Skill runtime-checkable Protocol
- get_default_skills() returns all 5 skills
"""
from __future__ import annotations

import pytest

from vector_os.core.skill import Skill
from vector_os.skills import (
    DetectSkill,
    HomeSkill,
    PickSkill,
    PlaceSkill,
    ScanSkill,
    get_default_skills,
)


# ---------------------------------------------------------------------------
# HomeSkill schema
# ---------------------------------------------------------------------------


class TestHomeSkillSchema:
    def setup_method(self):
        self.skill = HomeSkill()

    def test_name(self):
        assert self.skill.name == "home"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_empty(self):
        """HomeSkill takes no parameters."""
        assert self.skill.parameters == {}

    def test_preconditions_empty(self):
        """HomeSkill is always executable — no preconditions."""
        assert self.skill.preconditions == []

    def test_postconditions(self):
        assert "gripper_empty" in self.skill.postconditions

    def test_effects_gripper_open(self):
        assert self.skill.effects.get("gripper_state") == "open"

    def test_effects_held_object_none(self):
        assert self.skill.effects.get("held_object") is None

    def test_effects_is_moving_false(self):
        assert self.skill.effects.get("is_moving") is False

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# ScanSkill schema
# ---------------------------------------------------------------------------


class TestScanSkillSchema:
    def setup_method(self):
        self.skill = ScanSkill()

    def test_name(self):
        assert self.skill.name == "scan"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_empty(self):
        assert self.skill.parameters == {}

    def test_preconditions_empty(self):
        assert self.skill.preconditions == []

    def test_effects_is_dict(self):
        assert isinstance(self.skill.effects, dict)

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# DetectSkill schema
# ---------------------------------------------------------------------------


class TestDetectSkillSchema:
    def setup_method(self):
        self.skill = DetectSkill()

    def test_name(self):
        assert self.skill.name == "detect"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_query(self):
        assert "query" in self.skill.parameters

    def test_query_parameter_required(self):
        assert self.skill.parameters["query"]["required"] is True

    def test_query_parameter_type_string(self):
        assert self.skill.parameters["query"]["type"] == "string"

    def test_preconditions_empty(self):
        assert self.skill.preconditions == []

    def test_postconditions_empty(self):
        assert self.skill.postconditions == []

    def test_effects_empty(self):
        assert self.skill.effects == {}

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# PickSkill schema
# ---------------------------------------------------------------------------


class TestPickSkillSchema:
    def setup_method(self):
        self.skill = PickSkill()

    def test_name(self):
        assert self.skill.name == "pick"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_object_id(self):
        assert "object_id" in self.skill.parameters

    def test_parameters_has_object_label(self):
        assert "object_label" in self.skill.parameters

    def test_object_id_not_required(self):
        assert self.skill.parameters["object_id"]["required"] is False

    def test_object_label_not_required(self):
        assert self.skill.parameters["object_label"]["required"] is False

    def test_preconditions_gripper_empty(self):
        assert "gripper_empty" in self.skill.preconditions

    def test_postconditions_gripper_holding_any(self):
        assert "gripper_holding_any" in self.skill.postconditions

    def test_effects_gripper_state_holding(self):
        assert self.skill.effects.get("gripper_state") == "holding"

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# PlaceSkill schema
# ---------------------------------------------------------------------------


class TestPlaceSkillSchema:
    def setup_method(self):
        self.skill = PlaceSkill()

    def test_name(self):
        assert self.skill.name == "place"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_xyz(self):
        assert "x" in self.skill.parameters
        assert "y" in self.skill.parameters
        assert "z" in self.skill.parameters

    def test_xyz_not_required(self):
        assert self.skill.parameters["x"]["required"] is False
        assert self.skill.parameters["y"]["required"] is False
        assert self.skill.parameters["z"]["required"] is False

    def test_xyz_defaults(self):
        assert self.skill.parameters["x"]["default"] == pytest.approx(0.25)
        assert self.skill.parameters["y"]["default"] == pytest.approx(0.0)
        assert self.skill.parameters["z"]["default"] == pytest.approx(0.05)

    def test_preconditions_gripper_holding_any(self):
        assert "gripper_holding_any" in self.skill.preconditions

    def test_postconditions_gripper_empty(self):
        assert "gripper_empty" in self.skill.postconditions

    def test_effects_gripper_state_open(self):
        assert self.skill.effects.get("gripper_state") == "open"

    def test_effects_held_object_none(self):
        assert self.skill.effects.get("held_object") is None

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# get_default_skills
# ---------------------------------------------------------------------------


class TestGetDefaultSkills:
    def setup_method(self):
        self.skills = get_default_skills()

    def test_returns_five_skills(self):
        assert len(self.skills) == 5

    def test_all_satisfy_skill_protocol(self):
        for skill in self.skills:
            assert isinstance(skill, Skill), (
                f"{skill.__class__.__name__} does not satisfy Skill protocol"
            )

    def test_all_skill_names_present(self):
        names = {s.name for s in self.skills}
        assert names == {"home", "scan", "detect", "pick", "place"}

    def test_all_skills_have_descriptions(self):
        for skill in self.skills:
            assert isinstance(skill.description, str)
            assert len(skill.description) > 0, f"{skill.name} has empty description"

    def test_all_skills_have_parameters_dict(self):
        for skill in self.skills:
            assert isinstance(skill.parameters, dict), (
                f"{skill.name}.parameters must be a dict"
            )

    def test_all_skills_have_preconditions_list(self):
        for skill in self.skills:
            assert isinstance(skill.preconditions, list), (
                f"{skill.name}.preconditions must be a list"
            )

    def test_all_skills_have_postconditions_list(self):
        for skill in self.skills:
            assert isinstance(skill.postconditions, list), (
                f"{skill.name}.postconditions must be a list"
            )

    def test_all_skills_have_effects_dict(self):
        for skill in self.skills:
            assert isinstance(skill.effects, dict), (
                f"{skill.name}.effects must be a dict"
            )

    def test_unique_names(self):
        names = [s.name for s in self.skills]
        assert len(names) == len(set(names)), "Skill names must be unique"
