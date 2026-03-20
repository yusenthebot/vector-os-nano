"""Unit tests for vector_os.core.world_model — TDD RED phase."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def wm():
    from vector_os.core.world_model import WorldModel
    return WorldModel()


@pytest.fixture
def cup_obj():
    from vector_os.core.world_model import ObjectState
    return ObjectState(
        object_id="obj_001",
        label="cup",
        x=0.2,
        y=0.05,
        z=0.03,
        confidence=0.95,
        state="on_table",
        last_seen=time.time(),
    )


@pytest.fixture
def bottle_obj():
    from vector_os.core.world_model import ObjectState
    return ObjectState(
        object_id="obj_002",
        label="bottle",
        x=0.15,
        y=-0.08,
        z=0.05,
        confidence=0.85,
        state="on_table",
        last_seen=time.time(),
    )


class TestAddGetObject:
    def test_add_and_get_by_id(self, wm, cup_obj):
        wm.add_object(cup_obj)
        retrieved = wm.get_object("obj_001")
        assert retrieved is not None
        assert retrieved.object_id == "obj_001"
        assert retrieved.label == "cup"

    def test_get_nonexistent_returns_none(self, wm):
        assert wm.get_object("nonexistent") is None

    def test_add_overwrites_existing(self, wm, cup_obj):
        from vector_os.core.world_model import ObjectState
        wm.add_object(cup_obj)
        updated = ObjectState(
            object_id="obj_001",
            label="cup",
            x=0.3, y=0.1, z=0.05,
            confidence=0.99,
            state="grasped",
            last_seen=time.time(),
        )
        wm.add_object(updated)
        retrieved = wm.get_object("obj_001")
        assert retrieved.state == "grasped"
        assert retrieved.x == pytest.approx(0.3)


class TestRemoveObject:
    def test_remove_existing(self, wm, cup_obj):
        wm.add_object(cup_obj)
        wm.remove_object("obj_001")
        assert wm.get_object("obj_001") is None

    def test_remove_nonexistent_does_not_raise(self, wm):
        wm.remove_object("does_not_exist")  # Should not raise

    def test_remove_leaves_others_intact(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        wm.remove_object("obj_001")
        assert wm.get_object("obj_002") is not None


class TestGetObjects:
    def test_get_objects_empty(self, wm):
        assert wm.get_objects() == []

    def test_get_objects_returns_all(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        objs = wm.get_objects()
        assert len(objs) == 2

    def test_get_objects_by_label_cup(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        cups = wm.get_objects_by_label("cup")
        assert len(cups) == 1
        assert cups[0].label == "cup"

    def test_get_objects_by_label_missing(self, wm, cup_obj):
        wm.add_object(cup_obj)
        result = wm.get_objects_by_label("phone")
        assert result == []

    def test_get_objects_by_label_multiple(self, wm):
        from vector_os.core.world_model import ObjectState
        for i in range(3):
            obj = ObjectState(
                object_id=f"cup_{i}", label="cup",
                x=0.1 * i, y=0.0, z=0.03,
                last_seen=time.time(),
            )
            wm.add_object(obj)
        cups = wm.get_objects_by_label("cup")
        assert len(cups) == 3


class TestRobotState:
    def test_default_robot_state(self, wm):
        robot = wm.get_robot()
        assert robot.gripper_state == "open"
        assert robot.held_object is None
        assert robot.is_moving is False

    def test_update_gripper_state(self, wm):
        wm.update_robot_state(gripper_state="closed")
        assert wm.get_robot().gripper_state == "closed"

    def test_update_held_object(self, wm):
        wm.update_robot_state(held_object="obj_001", gripper_state="holding")
        robot = wm.get_robot()
        assert robot.held_object == "obj_001"
        assert robot.gripper_state == "holding"

    def test_update_is_moving(self, wm):
        wm.update_robot_state(is_moving=True)
        assert wm.get_robot().is_moving is True

    def test_update_joint_positions(self, wm):
        joints = (0.1, -1.2, 0.5, 0.8, 0.3)
        wm.update_robot_state(joint_positions=joints)
        assert wm.get_robot().joint_positions == joints

    def test_update_preserves_unset_fields(self, wm):
        """Updating one field should not reset others to defaults."""
        wm.update_robot_state(gripper_state="closed", held_object="obj_001")
        wm.update_robot_state(is_moving=True)
        robot = wm.get_robot()
        assert robot.gripper_state == "closed"
        assert robot.held_object == "obj_001"
        assert robot.is_moving is True


class TestPredicates:
    def test_gripper_empty_when_open(self, wm):
        wm.update_robot_state(gripper_state="open", held_object=None)
        assert wm.check_predicate("gripper_empty") is True

    def test_gripper_empty_false_when_holding(self, wm):
        wm.update_robot_state(gripper_state="holding", held_object="obj_001")
        assert wm.check_predicate("gripper_empty") is False

    def test_gripper_holding_any_true(self, wm):
        wm.update_robot_state(held_object="obj_001")
        assert wm.check_predicate("gripper_holding_any") is True

    def test_gripper_holding_any_false(self, wm):
        assert wm.check_predicate("gripper_holding_any") is False

    def test_gripper_holding_specific_object(self, wm):
        wm.update_robot_state(held_object="obj_001")
        assert wm.check_predicate("gripper_holding(obj_001)") is True
        assert wm.check_predicate("gripper_holding(obj_002)") is False

    def test_object_visible_true(self, wm, cup_obj):
        wm.add_object(cup_obj)  # confidence=0.95
        assert wm.check_predicate("object_visible(obj_001)") is True

    def test_object_visible_false_low_confidence(self, wm):
        from vector_os.core.world_model import ObjectState
        obj = ObjectState(
            object_id="faint", label="ghost",
            x=0.2, y=0.0, z=0.0,
            confidence=0.3,
            last_seen=time.time(),
        )
        wm.add_object(obj)
        assert wm.check_predicate("object_visible(faint)") is False

    def test_object_visible_false_missing(self, wm):
        assert wm.check_predicate("object_visible(no_such_obj)") is False

    def test_object_reachable_close(self, wm, cup_obj):
        # cup is at (0.2, 0.05) — distance ~0.206m < 0.35
        wm.add_object(cup_obj)
        assert wm.check_predicate("object_reachable(obj_001)") is True

    def test_object_reachable_far(self, wm):
        from vector_os.core.world_model import ObjectState
        far_obj = ObjectState(
            object_id="far", label="box",
            x=1.0, y=1.0, z=0.0,
            confidence=0.9,
            last_seen=time.time(),
        )
        wm.add_object(far_obj)
        assert wm.check_predicate("object_reachable(far)") is False

    def test_unknown_predicate_returns_false(self, wm):
        assert wm.check_predicate("unknown_predicate") is False


class TestSpatialRelations:
    def test_left_right_by_y_position(self, wm):
        """In robot frame, positive Y = left, negative Y = right."""
        from vector_os.core.world_model import ObjectState
        left_obj = ObjectState(
            object_id="left", label="cup",
            x=0.2, y=0.1, z=0.03,
            confidence=0.9, last_seen=time.time(),
        )
        right_obj = ObjectState(
            object_id="right", label="bottle",
            x=0.2, y=-0.1, z=0.03,
            confidence=0.9, last_seen=time.time(),
        )
        wm.add_object(left_obj)
        wm.add_object(right_obj)

        relations_left = wm.get_spatial_relations("left")
        relations_right = wm.get_spatial_relations("right")

        # left_obj is left of right_obj → right_obj should be in right_of for left_obj
        assert "right" in relations_left.get("right_of", [])
        # right_obj is right of left_obj → left_obj should be in left_of for right_obj
        assert "left" in relations_right.get("left_of", [])

    def test_spatial_relations_keys(self, wm, cup_obj):
        wm.add_object(cup_obj)
        relations = wm.get_spatial_relations("obj_001")
        assert "left_of" in relations
        assert "right_of" in relations
        assert "in_front_of" in relations
        assert "behind" in relations
        assert "near" in relations

    def test_spatial_relations_empty_world(self, wm, cup_obj):
        wm.add_object(cup_obj)
        relations = wm.get_spatial_relations("obj_001")
        # Only one object, all relations should be empty lists
        assert relations["left_of"] == []
        assert relations["right_of"] == []

    def test_spatial_relations_nonexistent_object(self, wm):
        relations = wm.get_spatial_relations("nonexistent")
        assert relations["left_of"] == []


class TestSerialization:
    def test_to_dict_is_json_serializable(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        d = wm.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_has_objects_key(self, wm, cup_obj):
        wm.add_object(cup_obj)
        d = wm.to_dict()
        assert "objects" in d
        assert len(d["objects"]) == 1

    def test_to_dict_has_robot_key(self, wm):
        d = wm.to_dict()
        assert "robot" in d
        assert "gripper_state" in d["robot"]

    def test_to_dict_roundtrip_object_count(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        d = wm.to_dict()
        assert len(d["objects"]) == 2


class TestSaveLoad:
    def test_save_and_load(self, wm, cup_obj, bottle_obj):
        from vector_os.core.world_model import WorldModel
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        wm.update_robot_state(gripper_state="closed", held_object="obj_001")

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            wm.save(path)
            loaded = WorldModel.load(path)

            assert len(loaded.get_objects()) == 2
            assert loaded.get_object("obj_001") is not None
            assert loaded.get_object("obj_001").label == "cup"
            assert loaded.get_robot().gripper_state == "closed"
            assert loaded.get_robot().held_object == "obj_001"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_creates_valid_yaml(self, wm, cup_obj):
        import yaml
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            wm.add_object(cup_obj)
            wm.save(path)
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data is not None
            assert "objects" in data
        finally:
            Path(path).unlink(missing_ok=True)


class TestSkillEffects:
    def test_apply_pick_effects(self, wm, cup_obj):
        from vector_os.core.types import SkillResult
        wm.add_object(cup_obj)
        result = SkillResult(success=True)
        wm.apply_skill_effects("pick", {"object_id": "obj_001"}, result)
        robot = wm.get_robot()
        assert robot.held_object == "obj_001"
        obj = wm.get_object("obj_001")
        assert obj.state == "grasped"

    def test_apply_place_effects(self, wm, cup_obj):
        from vector_os.core.types import SkillResult
        wm.add_object(cup_obj)
        wm.update_robot_state(held_object="obj_001", gripper_state="holding")
        result = SkillResult(success=True)
        wm.apply_skill_effects("place", {"object_id": "obj_001"}, result)
        robot = wm.get_robot()
        assert robot.held_object is None
        obj = wm.get_object("obj_001")
        assert obj.state == "placed"

    def test_apply_home_effects(self, wm):
        from vector_os.core.types import SkillResult
        wm.update_robot_state(gripper_state="closed")
        result = SkillResult(success=True)
        wm.apply_skill_effects("home", {}, result)
        assert wm.get_robot().gripper_state == "open"

    def test_apply_failed_skill_no_effect(self, wm, cup_obj):
        from vector_os.core.types import SkillResult
        wm.add_object(cup_obj)
        result = SkillResult(success=False, error_message="IK failed")
        wm.apply_skill_effects("pick", {"object_id": "obj_001"}, result)
        # No state change on failure
        assert wm.get_robot().held_object is None


class TestConfidenceDecay:
    def test_decay_reduces_confidence(self, wm):
        from vector_os.core.world_model import ObjectState
        obj = ObjectState(
            object_id="d1", label="cup",
            x=0.2, y=0.0, z=0.0,
            confidence=1.0,
            last_seen=time.time() - 10.0,  # 10 seconds ago
        )
        wm.add_object(obj)
        wm.decay_confidence(decay_rate=0.05)
        updated = wm.get_object("d1")
        assert updated.confidence < 1.0

    def test_decay_does_not_go_below_zero(self, wm):
        from vector_os.core.world_model import ObjectState
        obj = ObjectState(
            object_id="d2", label="cup",
            x=0.2, y=0.0, z=0.0,
            confidence=0.01,
            last_seen=time.time() - 1000.0,
        )
        wm.add_object(obj)
        wm.decay_confidence(decay_rate=1.0)
        updated = wm.get_object("d2")
        assert updated.confidence >= 0.0
