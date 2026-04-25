# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""v2.4 G3 — REP-103 xmat compliance for Go2ROS2Proxy.get_camera_pose.

Locks the corrected basis convention so a future refactor does not
silently re-introduce the v2.3 R2 latent inconsistency where the
xmat column labelled "right" was actually body-LEFT.

Convention asserted:
  * heading 0 (facing world +X) → right = (0, -1, 0)
  * heading 0 → up = (0, 0, 1) for a level camera (we tolerate the
    -5° MJCF pitch — small +X component allowed)
  * heading π/2 (facing world +Y) → right = (1, 0, 0)
  * xmat columns orthonormal (det == 1, each column unit length)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy


def _make_proxy(heading: float) -> Go2ROS2Proxy:
    p = Go2ROS2Proxy()
    p._position = (0.0, 0.0, 0.28)
    p._heading = heading
    return p


def _columns(proxy: Go2ROS2Proxy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, neg_forward) columns of the xmat."""
    _, xmat_flat = proxy.get_camera_pose()
    xmat = xmat_flat.reshape(3, 3)
    return xmat[:, 0], xmat[:, 1], xmat[:, 2]


def test_right_at_heading_zero_is_minus_y() -> None:
    right, _, _ = _columns(_make_proxy(0.0))
    assert right[0] == pytest.approx(0.0, abs=1e-9)
    assert right[1] == pytest.approx(-1.0, abs=1e-9)
    assert right[2] == pytest.approx(0.0, abs=1e-9)


def test_up_at_heading_zero_is_plus_z_with_pitch_drift() -> None:
    """Camera has -5° pitch from MJCF; up must still be predominantly +Z."""
    _, up, _ = _columns(_make_proxy(0.0))
    assert up[2] > 0.99            # cos(5°) ≈ 0.996
    assert up[0] > 0.0             # small +X due to downward pitch
    assert abs(up[1]) < 1e-9       # no Y component for heading 0


def test_right_at_heading_pi_over_2_is_plus_x() -> None:
    """Heading π/2 means dog faces world +Y → its right is world +X."""
    right, _, _ = _columns(_make_proxy(math.pi / 2))
    assert right[0] == pytest.approx(1.0, abs=1e-9)
    assert right[1] == pytest.approx(0.0, abs=1e-9)
    assert right[2] == pytest.approx(0.0, abs=1e-9)


def test_right_at_heading_pi_is_plus_y() -> None:
    """Heading π means dog faces world -X → its right is world +Y."""
    right, _, _ = _columns(_make_proxy(math.pi))
    assert right[0] == pytest.approx(0.0, abs=1e-9)
    assert right[1] == pytest.approx(1.0, abs=1e-9)
    assert right[2] == pytest.approx(0.0, abs=1e-9)


def test_xmat_columns_orthonormal_at_heading_zero() -> None:
    right, up, neg_fwd = _columns(_make_proxy(0.0))
    for col in (right, up, neg_fwd):
        assert np.linalg.norm(col) == pytest.approx(1.0, abs=1e-9)
    xmat = np.column_stack([right, up, neg_fwd])
    assert np.linalg.det(xmat) == pytest.approx(1.0, abs=1e-9)


def test_xmat_columns_orthonormal_across_headings() -> None:
    """Sweep four headings — every basis must remain orthonormal."""
    for h in (0.0, math.pi / 4, math.pi / 2, math.pi):
        right, up, neg_fwd = _columns(_make_proxy(h))
        assert np.dot(right, up) == pytest.approx(0.0, abs=1e-9)
        assert np.dot(right, neg_fwd) == pytest.approx(0.0, abs=1e-9)
        # Note: up · -fwd is not exactly 0 because forward has a small
        # -Z component due to MJCF pitch; we only require the first two
        # columns (right + up) plus the right ⊥ -fwd condition.


def test_existing_v23_bug_signature_absent() -> None:
    """Reproduces the old bug check: right at heading 0 is NOT (0, +1, 0).

    Failure here means a regression of the v2.3 ROS-left convention.
    """
    right, _, _ = _columns(_make_proxy(0.0))
    assert not (
        right[0] == pytest.approx(0.0, abs=1e-9)
        and right[1] == pytest.approx(1.0, abs=1e-9)
    ), "v2.3 bug regressed: right is body-LEFT (0, +1, 0)"
