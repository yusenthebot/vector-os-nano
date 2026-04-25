# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for MobilePlaceSkill — TDD Wave 3, Task T9.

7 tests covering:
1. Already reachable → navigate skipped, place delegated.
2. Full call order: navigate → wait_stable → place.
3. nav_failed diagnosis on navigate_to returning False.
4. wait_stable_timeout diagnosis on _wait_stable returning False.
5. Place failure propagation (ik_unreachable from PlaceTopDownSkill).
6. no_base hardware guard.
7. Explicit target_xyz wins over receptacle_id (get_object NOT called).
"""
from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.mobile_place import MobilePlaceSkill


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_base(x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> MagicMock:
    base = MagicMock()
    base.get_position.return_value = (x, y, 0.28)
    base.get_heading.return_value = heading
    base.navigate_to.return_value = True
    return base


def _make_arm() -> MagicMock:
    arm = MagicMock()
    arm.ik_top_down.return_value = [0.1] * 6
    arm.move_joints.return_value = True
    return arm


def _make_gripper() -> MagicMock:
    gripper = MagicMock()
    gripper.open.return_value = True
    return gripper


def _make_wm(obj_xyz: tuple[float, float, float] | None = None) -> MagicMock:
    """World model. If obj_xyz given, get_object returns a SimpleNamespace."""
    wm = MagicMock()
    if obj_xyz is not None:
        obj = MagicMock()
        obj.x, obj.y, obj.z = obj_xyz
        wm.get_object.return_value = obj
    else:
        wm.get_object.return_value = None
    return wm


def _place_success() -> SkillResult:
    return SkillResult(
        success=True,
        result_data={"placed_at": [0.5, 0.0, 0.30], "diagnosis": "ok"},
    )


def _place_failure(diagnosis: str = "ik_unreachable") -> SkillResult:
    return SkillResult(
        success=False,
        error_message=f"IK failed: {diagnosis}",
        result_data={"diagnosis": diagnosis},
    )


def _make_ctx(
    base: Any = None,
    arm: Any = None,
    gripper: Any = None,
    world_model: Any = None,
    config: dict | None = None,
) -> SkillContext:
    return SkillContext(
        base=base,
        arm=arm,
        gripper=gripper,
        world_model=world_model,
        config=config or {},
    )


# Target XYZ that requires navigation: dog at origin, target far away.
_TARGET_FAR = [5.0, 5.0, 0.25]
_TARGET_FAR_TUPLE = (5.0, 5.0, 0.25)

# Target XYZ where dog is already at approach pose (within tolerance).
# Dog at (0.6, 0.0), target at (0.0, 0.0) → approach at (0.55, 0.0, yaw=pi).
# Dog is within _APPROACH_XY_TOL of approach => already_reachable.
_TARGET_NEARBY = [0.0, 0.0, 0.25]
_DOG_AT_APPROACH_X = 0.55  # clearance default 0.55


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_wait_stable(return_val: bool = True):
    """Patch _wait_stable at the module level."""
    return patch(
        "vector_os_nano.skills.mobile_place._wait_stable",
        return_value=return_val,
    )


def _patch_approach_pose(approach_xyz: tuple[float, float, float] = (0.55, 0.0, math.pi)):
    """Patch compute_approach_pose at the mobile_place module level."""
    return patch(
        "vector_os_nano.skills.mobile_place.compute_approach_pose",
        return_value=approach_xyz,
    )


# ---------------------------------------------------------------------------
# Test 1 — Already reachable: navigate skipped, place delegated
# ---------------------------------------------------------------------------


def test_mobile_place_already_reachable_skips_navigate() -> None:
    """Dog already at approach pose → navigate_to NOT called, _place.execute called."""
    base = _make_base(x=_DOG_AT_APPROACH_X, y=0.0, heading=math.pi)
    arm = _make_arm()
    gripper = _make_gripper()
    ctx = _make_ctx(base=base, arm=arm, gripper=gripper, world_model=_make_wm())

    skill = MobilePlaceSkill()
    # Patch _place.execute to return success
    skill._place = MagicMock()
    skill._place.execute.return_value = _place_success()

    # Approach pose exactly where dog stands → already reachable
    approach = (_DOG_AT_APPROACH_X, 0.0, math.pi)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute({"target_xyz": _TARGET_NEARBY}, ctx)

    assert result.success is True
    base.navigate_to.assert_not_called()
    skill._place.execute.assert_called_once()
    # skipped_navigate reflected in result_data
    assert result.result_data["mobile_place"]["skipped_navigate"] is True


# ---------------------------------------------------------------------------
# Test 2 — Full call order: navigate → wait_stable → place
# ---------------------------------------------------------------------------


def test_mobile_place_calls_navigate_then_wait_then_place_in_order() -> None:
    """navigate_to called before _place.execute; wait_stable between them."""
    call_order: list[str] = []

    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.side_effect = lambda *a, **kw: call_order.append("navigate") or True

    arm = _make_arm()
    gripper = _make_gripper()
    ctx = _make_ctx(base=base, arm=arm, gripper=gripper, world_model=_make_wm())

    skill = MobilePlaceSkill()
    skill._place = MagicMock()

    def _place_side_effect(params, context):
        call_order.append("place")
        return _place_success()

    skill._place.execute.side_effect = _place_side_effect

    def _wait_side_effect(b, max_speed, settle, timeout):
        call_order.append("wait_stable")
        return True

    # Approach far from dog → navigate triggered
    approach = (5.0 - 0.55, 5.0, 0.0)
    with _patch_approach_pose(approach):
        with patch(
            "vector_os_nano.skills.mobile_place._wait_stable",
            side_effect=_wait_side_effect,
        ):
            result = skill.execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is True
    assert call_order == ["navigate", "wait_stable", "place"], (
        f"Expected navigate → wait_stable → place, got {call_order}"
    )


# ---------------------------------------------------------------------------
# Test 3 — navigate_to returns False → nav_failed
# ---------------------------------------------------------------------------


def test_mobile_place_nav_failed_returns_nav_failed() -> None:
    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.return_value = False

    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()

    approach = (4.45, 5.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "nav_failed"
    skill._place.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4 — _wait_stable returns False → wait_stable_timeout
# ---------------------------------------------------------------------------


def test_mobile_place_wait_stable_timeout_returns_wait_stable_timeout() -> None:
    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.return_value = True

    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()

    approach = (4.45, 5.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(False):
        result = skill.execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "wait_stable_timeout"
    skill._place.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5 — Place failure propagated (ik_unreachable from _place.execute)
# ---------------------------------------------------------------------------


def test_mobile_place_propagates_place_failure() -> None:
    """_place.execute returns success=False with diagnosis=ik_unreachable → propagated."""
    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.return_value = True

    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()
    skill._place.execute.return_value = _place_failure("ik_unreachable")

    approach = (4.45, 5.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "ik_unreachable"


# ---------------------------------------------------------------------------
# Test 6 — No base → no_base diagnosis
# ---------------------------------------------------------------------------


def test_mobile_place_no_base_returns_no_base() -> None:
    ctx = _make_ctx(
        base=None,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    result = MobilePlaceSkill().execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "no_base"


# ---------------------------------------------------------------------------
# Test 7 — Explicit target_xyz wins over receptacle_id; get_object NOT called
# ---------------------------------------------------------------------------


def test_mobile_place_explicit_target_xyz_vs_receptacle_id_explicit_wins() -> None:
    """Both target_xyz and receptacle_id supplied → use xyz, skip world_model lookup."""
    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.return_value = True
    wm = _make_wm(obj_xyz=(99.0, 99.0, 99.0))  # would give wrong target if used

    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=wm,
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()
    skill._place.execute.return_value = _place_success()

    approach = (4.45, 5.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute(
            {"target_xyz": _TARGET_FAR, "receptacle_id": "tray_01"},
            ctx,
        )

    # World model lookup must NOT have been called
    wm.get_object.assert_not_called()

    # Place was delegated with the explicit xyz, not (99, 99, 99)
    assert result.success is True
    place_params = skill._place.execute.call_args[0][0]
    assert list(place_params["target_xyz"]) == _TARGET_FAR


# ---------------------------------------------------------------------------
# Test 8 — no_arm hardware guard
# ---------------------------------------------------------------------------


def test_mobile_place_no_arm_returns_no_arm() -> None:
    ctx = _make_ctx(
        base=_make_base(),
        arm=None,
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    result = MobilePlaceSkill().execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "no_arm"


# ---------------------------------------------------------------------------
# Test 9 — no_gripper hardware guard
# ---------------------------------------------------------------------------


def test_mobile_place_no_gripper_returns_no_gripper() -> None:
    ctx = _make_ctx(
        base=_make_base(),
        arm=_make_arm(),
        gripper=None,
        world_model=_make_wm(),
    )

    result = MobilePlaceSkill().execute({"target_xyz": _TARGET_FAR}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "no_gripper"


# ---------------------------------------------------------------------------
# Test 10 — receptacle_id resolves from world_model
# ---------------------------------------------------------------------------


def test_mobile_place_receptacle_id_resolves_correctly() -> None:
    """receptacle_id is resolved from world_model; target_xyz not in params."""
    base = _make_base(x=0.0, y=0.0)
    base.navigate_to.return_value = True
    wm = _make_wm(obj_xyz=(3.0, 1.0, 0.15))

    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=wm,
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()
    skill._place.execute.return_value = _place_success()

    approach = (2.45, 1.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute({"receptacle_id": "shelf_01"}, ctx)

    wm.get_object.assert_called_once_with("shelf_01")
    # Place should receive the resolved xyz
    place_params = skill._place.execute.call_args[0][0]
    assert list(place_params["target_xyz"]) == [3.0, 1.0, 0.15]
    assert result.success is True


# ---------------------------------------------------------------------------
# Test 11 — receptacle_not_found
# ---------------------------------------------------------------------------


def test_mobile_place_receptacle_not_found() -> None:
    wm = _make_wm()  # get_object returns None

    ctx = _make_ctx(
        base=_make_base(),
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=wm,
    )

    result = MobilePlaceSkill().execute({"receptacle_id": "missing_tray"}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "receptacle_not_found"


# ---------------------------------------------------------------------------
# Test 12 — missing_target (neither target_xyz nor receptacle_id)
# ---------------------------------------------------------------------------


def test_mobile_place_missing_target() -> None:
    ctx = _make_ctx(
        base=_make_base(),
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    result = MobilePlaceSkill().execute({}, ctx)

    assert result.success is False
    assert result.result_data["diagnosis"] == "missing_target"


# ---------------------------------------------------------------------------
# Test 13 — skip_navigate=True bypasses navigate_to
# ---------------------------------------------------------------------------


def test_mobile_place_skip_navigate_flag_bypasses_navigate() -> None:
    base = _make_base(x=0.0, y=0.0)
    ctx = _make_ctx(
        base=base,
        arm=_make_arm(),
        gripper=_make_gripper(),
        world_model=_make_wm(),
    )

    skill = MobilePlaceSkill()
    skill._place = MagicMock()
    skill._place.execute.return_value = _place_success()

    approach = (4.45, 5.0, 0.0)
    with _patch_approach_pose(approach), _patch_wait_stable(True):
        result = skill.execute(
            {"target_xyz": _TARGET_FAR, "skip_navigate": True},
            ctx,
        )

    base.navigate_to.assert_not_called()
    assert result.success is True
    assert result.result_data["mobile_place"]["skipped_navigate"] is True


# ---------------------------------------------------------------------------
# Test 14 — _wait_stable direct unit test
# ---------------------------------------------------------------------------


def test_wait_stable_returns_true_when_stable() -> None:
    """_wait_stable returns True when base stops moving within timeout."""
    from vector_os_nano.skills.mobile_place import _wait_stable

    base = MagicMock()
    # Position does not change → speed = 0 → stable immediately
    base.get_position.return_value = (1.0, 2.0, 0.0)

    with patch("time.sleep"), patch("time.monotonic") as mock_time:
        # Provide enough time values: deadline check, loop body, settle check
        # Sequence: start=0, deadline=5; loop ticks at 0, 0, 0, 6 (deadline exceeded)
        # First call: t=0 (deadline=5); inside loop: t=0 (pos_a), sleep, t=0 (pos_b),
        # speed=0 < max → stable_since=0.1 (next call), settle check: 0.2-0.1=0.1 < 1.0
        # Simplify: just give enough that the settle condition triggers quickly.
        tick = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 99.0]
        mock_time.side_effect = tick
        result = _wait_stable(base, max_speed=0.05, settle_duration=1.0, timeout=5.0)

    assert result is True


def test_wait_stable_returns_false_on_timeout() -> None:
    """_wait_stable returns False when base never stabilises within timeout."""
    from vector_os_nano.skills.mobile_place import _wait_stable

    base = MagicMock()
    # Position changes each call → always moving
    positions = [(float(i), 0.0, 0.0) for i in range(100)]
    base.get_position.side_effect = positions

    with patch("time.sleep"), patch("time.monotonic") as mock_time:
        # Deadline expires quickly
        mock_time.side_effect = [0.0, 0.0, 0.05, 0.05, 6.0]
        result = _wait_stable(base, max_speed=0.05, settle_duration=1.0, timeout=5.0)

    assert result is False
