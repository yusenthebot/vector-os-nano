# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for compute_approach_pose utility.

TDD RED phase: written before implementation exists.
All tests should initially fail with ImportError, then pass after GREEN phase.
"""
import math

import pytest

from vector_os_nano.skills.utils.approach_pose import compute_approach_pose


# ---------------------------------------------------------------------------
# Test 1: dog east of object — approach from east, face west (π)
# ---------------------------------------------------------------------------

def test_approach_pose_dog_east_of_object_stops_east() -> None:
    """object=(0,0,0), dog=(2,0,0) → approach stops at (0.55, 0, π)."""
    ax, ay, ayaw = compute_approach_pose(
        object_xyz=(0.0, 0.0, 0.0),
        dog_pose=(2.0, 0.0, 0.0),
    )
    assert ax == pytest.approx(0.55, abs=1e-9)
    assert ay == pytest.approx(0.0, abs=1e-9)
    assert ayaw == pytest.approx(math.pi, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2: dog north of object — approach from north, face south (-π/2)
# ---------------------------------------------------------------------------

def test_approach_pose_dog_north_of_object_stops_north() -> None:
    """object=(0,0,0), dog=(0,2,0) → approach stops at (0, 0.55, -π/2)."""
    ax, ay, ayaw = compute_approach_pose(
        object_xyz=(0.0, 0.0, 0.0),
        dog_pose=(0.0, 2.0, 0.0),
    )
    assert ax == pytest.approx(0.0, abs=1e-9)
    assert ay == pytest.approx(0.55, abs=1e-9)
    # faces south: atan2(0 - 0.55, 0 - 0) = atan2(-0.55, 0) = -π/2
    assert ayaw == pytest.approx(-math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: yaw faces object — 4 cardinal directions around object=(5,5,0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dog_xy",
    [
        (7.0, 5.0),   # east of object
        (3.0, 5.0),   # west of object
        (5.0, 7.0),   # north of object
        (5.0, 3.0),   # south of object
    ],
)
def test_approach_pose_yaw_faces_object(dog_xy: tuple[float, float]) -> None:
    """Approach yaw must point from approach_xy toward object=(5,5)."""
    obj_x, obj_y = 5.0, 5.0
    dog_x, dog_y = dog_xy
    ax, ay, ayaw = compute_approach_pose(
        object_xyz=(obj_x, obj_y, 0.0),
        dog_pose=(dog_x, dog_y, 0.0),
    )
    expected_yaw = math.atan2(obj_y - ay, obj_x - ax)
    assert ayaw == pytest.approx(expected_yaw, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: clearance is the exact Euclidean distance from approach to object
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "clearance,dog_pose",
    [
        (0.55, (3.0, 0.0, 0.0)),
        (0.30, (0.0, -4.0, 0.0)),
        (1.00, (1.5, 2.5, 0.5)),
    ],
)
def test_approach_pose_clearance_is_exact_distance(
    clearance: float, dog_pose: tuple[float, float, float]
) -> None:
    """||approach_xy - object_xy|| == clearance to within 1e-9."""
    obj_xyz = (0.0, 0.0, 0.0)
    ax, ay, _ = compute_approach_pose(
        object_xyz=obj_xyz,
        dog_pose=dog_pose,
        clearance=clearance,
    )
    dist = math.sqrt((ax - obj_xyz[0]) ** 2 + (ay - obj_xyz[1]) ** 2)
    assert dist == pytest.approx(clearance, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: degenerate case — dog == object raises ValueError
# ---------------------------------------------------------------------------

def test_approach_pose_degenerate_dog_equals_object_raises_value_error() -> None:
    """dog XY == object XY must raise ValueError."""
    with pytest.raises(ValueError):
        compute_approach_pose(
            object_xyz=(0.0, 0.0, 0.0),
            dog_pose=(0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------
# Test 6: custom clearance propagated correctly
# ---------------------------------------------------------------------------

def test_approach_pose_custom_clearance_propagated() -> None:
    """Same east geometry but clearance=0.3 → approach_x=0.3; clearance=1.0 → approach_x=1.0."""
    _, _, _ = compute_approach_pose(
        object_xyz=(0.0, 0.0, 0.0),
        dog_pose=(2.0, 0.0, 0.0),
        clearance=0.3,
    )
    ax_03, _, _ = compute_approach_pose(
        object_xyz=(0.0, 0.0, 0.0),
        dog_pose=(2.0, 0.0, 0.0),
        clearance=0.3,
    )
    ax_10, _, _ = compute_approach_pose(
        object_xyz=(0.0, 0.0, 0.0),
        dog_pose=(2.0, 0.0, 0.0),
        clearance=1.0,
    )
    assert ax_03 == pytest.approx(0.3, abs=1e-9)
    assert ax_10 == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 7 (bonus): approach_direction="from_normal" raises NotImplementedError
# ---------------------------------------------------------------------------

def test_approach_pose_from_normal_not_implemented() -> None:
    """approach_direction='from_normal' is reserved for v2.3 and must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        compute_approach_pose(
            object_xyz=(1.0, 1.0, 0.0),
            dog_pose=(3.0, 3.0, 0.0),
            approach_direction="from_normal",
        )
